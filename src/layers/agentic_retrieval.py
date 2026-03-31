"""Agentic RAG: 迭代检索 + 跨片段推理。

每轮：全部已积累片段 → LLM 推理 → 够了就停 / 不够就搜新的。
片段总数限制在 max_fragments 以内，避免注意力稀释。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config
    from src.layers.l3_episodic import L3EpisodicMemory
    from src.models import L3Result

logger = logging.getLogger(__name__)

EVAL_SYSTEM = "你是一个信息检索与推理评估模块。请严格按 JSON 格式输出。"

EVAL_PROMPT = """\
系统从小说原文中检索到了一些片段，请尝试仅根据这些片段回答用户的问题。

## 对话场景
角色：{character_name}（被扮演的角色）
对话对象：{identity}
用户的消息是对{character_name}说的。

## 用户问题
{user_message}

## 已检索到的全部片段（共{fragment_count}个）
{all_fragments}

## 任务
1. 将用户的问题拆解为子问题。例如"结婚时各自多大"需要知道：
   a. 分离时各自几岁？
   b. 分离了多久？
   c. 计算：分离年龄 + 分离时长 = 结婚年龄
2. 检查已有片段能回答哪些子问题（注意仔细阅读每个片段）
3. 如果所有子问题都能回答（直接或推理）→ sufficient。推理得出的结论也算回答，不需要原文逐字确认
4. 如果某个子问题缺信息，针对那个子问题生成检索词

## 搜索机制说明
new_queries 走语义向量匹配（意思相近就能找到）。
new_keywords 走原文精确子串匹配（原文必须逐字包含这个词才能命中）。
因此 new_keywords 必须是小说原文中真实存在的短词。
如果不确定具体数字，穷举候选：如["三年", "四年", "五年", "六年", "七年"]。

## 输出格式（纯 JSON）
能回答时：
{{"sufficient": true, "reasoning": "分步推理：子问题1→答案，子问题2→答案，最终结论"}}

不能回答时：
{{"sufficient": false, "solved": "已解决的子问题", "missing": "缺什么", "new_queries": ["语义检索"], "new_keywords": ["精确子串"]}}"""


class AgenticRetrieval:
    """迭代式检索 + 跨片段推理。"""

    def __init__(self, config: Config, l3: L3EpisodicMemory, max_iterations: int = 3):
        self._config = config
        self._l3 = l3
        self._max_iterations = max_iterations
        self._max_fragments = 15  # 评估时最多传几个片段
        self._provider = config.get("agentic.provider", config.get_runtime_provider("step_a"))

    async def retrieve(
        self,
        user_message: str,
        initial_query: str,
        initial_keywords: list[str] | None = None,
        topic_is_past: bool = False,
        character_name: str = "",
        identity: str = "",
        known_facts: str = "",
    ) -> L3Result:
        from src.models import L3Result

        all_scene_ids: list[str] = []
        all_fragments: dict[str, str] = {}

        # 第 1 轮检索
        result = self._l3.retrieve(
            query=initial_query, filter_tags=None,
            topic_is_past=topic_is_past, llm_keywords=initial_keywords,
        )
        self._collect(result, all_scene_ids, all_fragments)

        if not all_fragments:
            return result

        agent_reasoning = ""

        for iteration in range(self._max_iterations - 1):
            # 每轮都看全部片段（限制数量）
            top_frags = {sid: all_fragments[sid] for sid in all_scene_ids[:self._max_fragments]}
            fragments_text = self._format_fragments(top_frags)

            # 知识库事实 + 检索片段一起给评估 LLM
            eval_context = ""
            if known_facts:
                eval_context = f"## 已知事实（优先级最高）\n{known_facts}\n\n"
            eval_context += fragments_text

            eval_result = await self._evaluate(
                user_message, eval_context, len(top_frags),
                character_name, identity,
            )

            if eval_result is None or eval_result.get("sufficient", True):
                agent_reasoning = eval_result.get("reasoning", "") if eval_result else ""
                logger.info(f"[agentic] 第{iteration+1}轮: sufficient. {agent_reasoning[:200]}")
                break

            solved = eval_result.get("solved", "")
            missing = eval_result.get("missing", "")
            new_queries = eval_result.get("new_queries", [])
            new_keywords = eval_result.get("new_keywords", [])
            logger.info(f"[agentic] 第{iteration+1}轮: 已知[{solved[:80]}] 缺[{missing[:120]}] → kw: {new_keywords[:10]}")

            if not new_queries and not new_keywords:
                break

            # 自动扩展关键词：当缺时间/年龄信息时，注入数字候选
            pre_count = len(new_keywords)
            new_keywords = self._expand_keywords(new_keywords, missing)
            if len(new_keywords) > pre_count:
                logger.info(f"[agentic] 关键词扩展: +{len(new_keywords)-pre_count}个, 总计: {new_keywords[:15]}")

            # 搜索
            found_new = False
            for q in new_queries[:2]:
                new_result = self._l3.retrieve(
                    query=q, filter_tags=None,
                    topic_is_past=topic_is_past, llm_keywords=new_keywords,
                )
                if self._collect(new_result, all_scene_ids, all_fragments):
                    found_new = True

            if not found_new:
                logger.info(f"[agentic] 无新片段，停止")
                break

        return self._assemble_result(all_scene_ids, all_fragments, agent_reasoning)

    @staticmethod
    def _expand_keywords(keywords: list[str], missing: str) -> list[str]:
        """根据缺失信息类型，自动扩展关键词。"""
        expanded = list(keywords)
        missing_lower = missing.lower()

        # 缺时间/年数/时长 → 注入数字+年
        time_triggers = ["几年", "多久", "多少年", "时间", "时长", "年数", "年限", "间隔"]
        if any(t in missing_lower for t in time_triggers):
            year_kws = ["一年", "两年", "三年", "四年", "五年", "六年", "七年", "八年", "九年", "十年"]
            for yk in year_kws:
                if yk not in expanded:
                    expanded.append(yk)

        # 缺年龄 → 注入数字+岁
        age_triggers = ["几岁", "年龄", "多大", "岁数"]
        if any(t in missing_lower for t in age_triggers):
            age_kws = [f"{n}岁" for n in ["十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十", "二十一"]]
            for ak in age_kws:
                if ak not in expanded:
                    expanded.append(ak)

        return expanded

    def _assemble_result(self, all_scene_ids, all_fragments, agent_reasoning):
        from src.models import L3Result
        sep = "\n\n---SCENE---\n\n"
        parts = []
        if agent_reasoning:
            parts.append(f"★以下是系统根据多个原文片段的综合推理，请据此回答★\n【推理结论】{agent_reasoning}")
        for sid in all_scene_ids:
            if sid in all_fragments:
                parts.append(all_fragments[sid])
        return L3Result(prompt_text=sep.join(parts), scenes_used=all_scene_ids)

    def _collect(self, result, all_ids: list[str], all_frags: dict[str, str]) -> bool:
        if not result.prompt_text:
            return False
        parts = result.prompt_text.split("\n\n---SCENE---\n\n")
        added = False
        for sid, part in zip(result.scenes_used, parts):
            if sid not in all_frags:
                all_ids.append(sid)
                all_frags[sid] = part
                added = True
        return added

    def _format_fragments(self, frags: dict[str, str]) -> str:
        parts = []
        for i, (sid, text) in enumerate(frags.items()):
            parts.append(f"[片段{i+1}] ({sid})\n{text[:500]}")
        return "\n\n".join(parts)

    async def _evaluate(self, user_message: str, all_fragments: str,
                        fragment_count: int, character_name: str, identity: str) -> dict | None:
        from src.llm.client import LLMClient

        prompt = EVAL_PROMPT.replace("{character_name}", character_name or "角色")
        prompt = prompt.replace("{identity}", identity or "未知")
        prompt = prompt.replace("{user_message}", user_message)
        prompt = prompt.replace("{all_fragments}", all_fragments)
        prompt = prompt.replace("{fragment_count}", str(fragment_count))

        client = LLMClient(self._config)
        try:
            response = await client.complete(
                provider=self._provider,
                system_prompt=EVAL_SYSTEM,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=8192,
            )
        except Exception as e:
            logger.error(f"[agentic] 评估失败: {e}")
            return None
        finally:
            await client.close()

        return self._parse(response)

    @staticmethod
    def _parse(response: str) -> dict | None:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:
            return None
