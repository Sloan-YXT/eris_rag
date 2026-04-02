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
请根据下方所有可用信息回答用户的问题。可用信息包括：对话场景、已知事实、检索到的原文片段。

## 对话场景
角色：{character_name}（被扮演的角色，对话中的"你"）
对话对象：{identity}（发消息的人，对话中的"我"）

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
3. 判断回答是否完整。用以下检查清单：
   - 事件的起因、经过、结果都有了吗？
   - 涉及人物关系变化时，后续发展（怀孕、结婚、分手、和好等）查了吗？
   - 如果你是听这件事的人，还会追问什么？那就是还缺的信息

4. 你是这部作品的专家，你对作品的理解比片段更全面。
   - 如果你知道答案，直接回答，不需要在片段中找到逐字确认
   - 片段是证据，用来验证你的判断，不是唯一信息来源
   - 不要简单列举片段中出现的名称，要理解它们之间的关系和本质（如：多个名称可能是同一事物的不同形态）
5. sufficient 的标准：reasoning 中必须包含对用户问题的**明确、具体的回答**。
   - ⚠ 如果你的 reasoning 里出现"无法确定""具体不知""不清楚"等字眼，那就不是 sufficient
6. 只有当片段+你的知识都推不出具体答案时，才 insufficient 并继续搜索

## 搜索机制说明
new_queries 走语义向量匹配（意思相近就能找到）。
new_keywords 走原文精确子串匹配（原文必须逐字包含这个词才能命中）。
因此 new_keywords 必须是小说原文中真实存在的短词。
如果不确定具体内容，用你自己的知识推测最可能的答案，把推测的具体名词加入 new_keywords 去原文中验证。
验证通过（原文确实包含）才算找到，验证不通过就继续搜别的方向。
这样做的目的是：你的知识可能知道答案但不确定，用 RAG 原文来确认或否定你的猜测。

## 输出格式（纯 JSON）
sufficient=true 的条件：reasoning 里必须包含用户问题的具体答案（数字、人名、事件等）。
如果你的结论是"无法确定""不清楚""具体不知"，那必须输出 sufficient=false 继续搜索。
片段没有直接写明答案时，用片段线索+你自己的知识推算，推算结果也算具体答案。

能给出具体答案时：
{{"sufficient": true, "reasoning": "分步推理：子问题1→答案，子问题2→答案，最终结论"}}

无法给出具体答案时：
{{"sufficient": false, "solved": "已解决的子问题", "missing": "缺什么", "new_queries": ["语义检索"], "new_keywords": ["精确子串"]}}"""


class AgenticRetrieval:
    """迭代式检索 + 跨片段推理。"""

    def __init__(self, config: Config, l3: L3EpisodicMemory, max_iterations: int = 3):
        self._config = config
        self._l3 = l3
        self._max_iterations = max_iterations
        self._max_fragments = config.get("agentic.max_fragments", 15)
        self._search_top_k = config.get("agentic.search_top_k", 15)
        self._provider_first = config.get("agentic.provider_first", config.get_runtime_provider("step_a"))
        self._provider_rest = config.get("agentic.provider_rest", config.get_runtime_provider("step_a"))
        self._force_second_round = config.get("agentic.force_second_round", True)
        self._eval_timeout = config.get("agentic.eval_timeout", 60)

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

        # 第 1 轮检索（大 top_k，让 agent 看到更多候选）
        result = self._l3.retrieve(
            query=initial_query, filter_tags=None,
            topic_is_past=topic_is_past, llm_keywords=initial_keywords,
            top_k=self._search_top_k,
        )
        self._collect(result, all_scene_ids, all_fragments)

        if not all_fragments:
            return result

        agent_reasoning = ""
        last_solved = ""
        last_missing = ""
        prev_reasoning = ""  # 上一轮的推理结论，传给下一轮

        for iteration in range(self._max_iterations - 1):
            # 每轮都看全部片段（限制数量）
            top_frags = {sid: all_fragments[sid] for sid in all_scene_ids[:self._max_fragments]}
            fragments_text = self._format_fragments(top_frags)

            # 知识库事实 + 上一轮推理 + 检索片段一起给评估 LLM
            eval_context = ""
            if known_facts:
                eval_context = f"## 已知事实（优先级最高）\n{known_facts}\n\n"
            if prev_reasoning:
                eval_context += f"## 上一轮推理结论（请在此基础上继续，不要推翻已确认的事实）\n{prev_reasoning}\n\n"
            eval_context += fragments_text

            provider = self._provider_first if iteration == 0 else self._provider_rest
            logger.info(f"[agentic] 第{iteration+1}轮: 使用 provider={provider}, 片段数={len(top_frags)}")
            # 调试：打印前几个片段的ID和前100字
            if iteration == 0:
                for i, (sid, text) in enumerate(top_frags.items()):
                    if i < 20:
                        logger.debug(f"[agentic] 片段{i+1} [{sid}] ({len(text)}字): {text[:150].replace(chr(10), ' ')}")
            eval_result = await self._evaluate(
                user_message, eval_context, len(top_frags),
                character_name, identity, provider=provider,
            )

            # LLM 调用失败 → 尝试 fallback provider，再失败才放弃
            if eval_result is None:
                if provider != self._provider_rest:
                    logger.warning(f"[agentic] 第{iteration+1}轮: {provider} 失败，fallback 到 {self._provider_rest}")
                    eval_result = await self._evaluate(
                        user_message, eval_context, len(top_frags),
                        character_name, identity, provider=self._provider_rest,
                    )
                if eval_result is None:
                    logger.warning(f"[agentic] 第{iteration+1}轮: 评估失败，用已有片段")
                    break

            is_sufficient = eval_result.get("sufficient", True)

            if is_sufficient:
                agent_reasoning = eval_result.get("reasoning", "")
                # 首轮强制续搜：保留结论，继续搜索
                if iteration == 0 and self._force_second_round:
                    logger.info(f"[agentic] 第1轮: 强制续搜（首轮不允许sufficient）. {agent_reasoning[:200]}")
                    prev_reasoning = agent_reasoning
                    eval_result = {
                        "sufficient": False,
                        "solved": agent_reasoning,
                        "missing": "首轮强制续搜，验证是否有遗漏信息",
                        "new_queries": [user_message],
                        "new_keywords": initial_keywords or [],
                    }
                    is_sufficient = False  # 继续走下面的 solved/missing/queries 提取
                else:
                    logger.info(f"[agentic] 第{iteration+1}轮: sufficient. {agent_reasoning[:200]}")
                    break

            solved = eval_result.get("solved", "")
            missing = eval_result.get("missing", "")
            new_queries = eval_result.get("new_queries", [])
            new_keywords = eval_result.get("new_keywords", [])
            logger.info(f"[agentic] 第{iteration+1}轮: 已知[{solved[:80]}] 缺[{missing[:120]}] → kw: {new_keywords[:10]}")

            # 更新上一轮推理，传给下一轮
            prev_reasoning = f"已知：{solved}\n缺失：{missing}" if solved else prev_reasoning

            # 保存最后一轮的已知信息作为 fallback
            last_solved = solved
            last_missing = missing

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
                    top_k=self._search_top_k,
                )
                if self._collect(new_result, all_scene_ids, all_fragments):
                    found_new = True

            if not found_new:
                logger.info(f"[agentic] 无新片段，停止")
                break

        # 迭代耗尽未 sufficient 时，用最后一轮的已知+缺失作为 fallback
        is_fallback = False
        if not agent_reasoning and last_solved:
            agent_reasoning = f"已知：{last_solved}\n未确认：{last_missing}"
            is_fallback = True
            logger.info(f"[agentic] 迭代耗尽，使用 fallback: {agent_reasoning[:200]}")

        return self._assemble_result(all_scene_ids, all_fragments, agent_reasoning, is_fallback)

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

        # 缺结束点/离开时间 → 注入结束事件关键词
        end_triggers = ["结束", "离开", "回归", "重逢", "结束点", "何时离开", "何时回"]
        if any(t in missing_lower for t in end_triggers):
            end_kws = ["离开", "回归", "重逢", "归来", "返回", "结束", "告别", "出发"]
            for ek in end_kws:
                if ek not in expanded:
                    expanded.append(ek)

        # 缺年龄 → 注入数字+岁
        age_triggers = ["几岁", "年龄", "多大", "岁数"]
        if any(t in missing_lower for t in age_triggers):
            age_kws = [f"{n}岁" for n in ["十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十", "二十一"]]
            for ak in age_kws:
                if ak not in expanded:
                    expanded.append(ak)

        return expanded

    def _assemble_result(self, all_scene_ids, all_fragments, agent_reasoning, is_fallback=False):
        from src.models import L3Result
        sep = "\n\n---SCENE---\n\n"
        parts = []
        if agent_reasoning:
            if is_fallback:
                parts.append(f"★系统未能找到完整答案，以下是部分分析，请结合下方原文片段自行推理★\n{agent_reasoning}")
            else:
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
            parts.append(f"[片段{i+1}] ({sid})\n{text}")
        return "\n\n".join(parts)

    async def _evaluate(self, user_message: str, all_fragments: str,
                        fragment_count: int, character_name: str, identity: str,
                        provider: str = "") -> dict | None:
        from src.llm.client import LLMClient

        prompt = EVAL_PROMPT.replace("{character_name}", character_name or "角色")
        prompt = prompt.replace("{identity}", identity or "未知")
        prompt = prompt.replace("{user_message}", user_message)
        prompt = prompt.replace("{all_fragments}", all_fragments)
        prompt = prompt.replace("{fragment_count}", str(fragment_count))

        client = LLMClient(self._config, timeout=self._eval_timeout)
        try:
            response = await client.complete(
                provider=provider or self._provider_first,
                system_prompt=EVAL_SYSTEM,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=8192,
            )
        except Exception as e:
            logger.error(f"[agentic] 评估失败: {type(e).__name__}: {e}")
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
