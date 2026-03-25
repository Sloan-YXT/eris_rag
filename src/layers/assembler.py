"""Assembler: Execute L1→L2→L3 pipeline and combine with L4 into final system_prompt."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import tiktoken

from src.models import AssemblyMetadata, AssemblyResult, StepAResult

if TYPE_CHECKING:
    from src.layers.l1_core import L1CoreIdentity
    from src.layers.l2_behavior import L2BehaviorRules
    from src.layers.l3_episodic import L3EpisodicMemory
    from src.layers.l4_working import L4WorkingMemory
    from src.layers.step_a import StepA
    from src.config import Config

logger = logging.getLogger(__name__)

# Meta-instruction appended to every assembled prompt
META_INSTRUCTION = """## 元指令
你是艾莉丝·格雷拉特，不是AI助手。请遵守以下规则：
- 用第一人称回应，保持角色一致性
- 参考上方的行为规则和场景记忆来决定反应方式
- 语气、用词要符合角色的说话风格
- 不要打破角色，不要提及自己是AI
- 回复长度适中，像真实对话一样自然"""


class Assembler:
    """Orchestrate the full L1→L2→L3 retrieval pipeline and assemble the system prompt."""

    def __init__(
        self,
        config: Config,
        step_a: StepA,
        l1: L1CoreIdentity,
        l2: L2BehaviorRules,
        l3: L3EpisodicMemory,
        l4: L4WorkingMemory,
        subconscious=None,
    ):
        self._config = config
        self._step_a = step_a
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._l4 = l4
        self._subconscious = subconscious  # SubconsciousMemory | None
        self._target_tokens = config.get_retrieval("target_total_tokens", 2000)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._user_prompts_path = Path(self._config.get("user_prompts_path", "./data/user_prompts.yaml"))
        self._user_prompts: dict = {}
        self._user_prompts_mtime: float = 0
        self._reload_user_prompts()

    def _reload_user_prompts(self) -> None:
        """检查文件修改时间，变了就重新加载。"""
        if not self._user_prompts_path.exists():
            self._user_prompts = {"default": {"prompt": ""}, "users": {}}
            return
        mtime = self._user_prompts_path.stat().st_mtime
        if mtime != self._user_prompts_mtime:
            import yaml
            with open(self._user_prompts_path, encoding="utf-8") as f:
                self._user_prompts = yaml.safe_load(f) or {}
            self._user_prompts_mtime = mtime
            logger.info(f"user_prompts.yaml reloaded (mtime={mtime})")

    def _get_user_prompt(self, sender_id: str, sender_nickname: str = "") -> str:
        """根据 sender_id/昵称查找用户自定义提示词。

        优先级：
          1. QQ号精确匹配 sender_ids → 用自定义 prompt
          2. 昵称匹配用户配置名 → 用自定义 prompt（无需配置QQ号）
          3. 有昵称但没匹配 → 告诉艾莉丝对方叫什么 + default
          4. 什么都没有 → default
        """
        users = self._user_prompts.get("users", {})
        # 1. QQ号精确匹配
        for name, cfg in users.items():
            if sender_id in cfg.get("sender_ids", []):
                return cfg.get("prompt", "").strip()
        # 2. 昵称匹配：精确匹配配置名/aliases，或模糊包含
        if sender_nickname:
            nick = sender_nickname.strip()
            for name, cfg in users.items():
                # 精确匹配
                if nick == name:
                    return cfg.get("prompt", "").strip()
                # aliases 匹配（支持包含关系：昵称包含alias 或 alias包含昵称）
                for alias in cfg.get("aliases", []):
                    if alias in nick or nick in alias:
                        return cfg.get("prompt", "").strip()
        # 3. 有昵称 → 直接告诉 LLM 对方是谁，让它自己判断
        if sender_nickname:
            return f"对方是{sender_nickname}。"
        # 4. 什么都没有
        default = self._user_prompts.get("default", {}).get("prompt", "").strip()
        return default

    def _resolve_identity(self, sender_id: str, sender_nickname: str) -> str:
        """解析对方身份名，用于 L3 关系检索。

        返回角色名（如 "洛琪希"），或昵称，或空字符串。
        """
        users = self._user_prompts.get("users", {})
        # QQ号匹配
        for name, cfg in users.items():
            if sender_id in cfg.get("sender_ids", []):
                return name
        # 昵称匹配：精确 + 模糊包含
        if sender_nickname:
            nick = sender_nickname.strip()
            for name, cfg in users.items():
                if nick == name:
                    return name
                for alias in cfg.get("aliases", []):
                    if alias in nick or nick in alias:
                        return name
        # 未匹配返回昵称本身
        return sender_nickname.strip()

    def assemble(
        self,
        user_message: str,
        conversation_context: list[str] | None = None,
        sender_id: str = "default",
        sender_nickname: str = "",
    ) -> AssemblyResult:
        """Sync entry point."""
        identity = self._resolve_identity(sender_id, sender_nickname)
        step_a_result = self._step_a.analyze(user_message, conversation_context)
        return self._assemble_inner(step_a_result, user_message, conversation_context, sender_id, sender_nickname, identity)

    async def assemble_async(
        self,
        user_message: str,
        conversation_context: list[str] | None = None,
        sender_id: str = "default",
        sender_nickname: str = "",
    ) -> AssemblyResult:
        """Async entry point."""
        identity = self._resolve_identity(sender_id, sender_nickname)
        step_a_result = await self._step_a.analyze_async(user_message, conversation_context)
        return self._assemble_inner(step_a_result, user_message, conversation_context, sender_id, sender_nickname, identity)

    def _assemble_inner(
        self,
        step_a_result: StepAResult,
        user_message: str,
        conversation_context: list[str] | None,
        sender_id: str,
        sender_nickname: str = "",
        identity: str = "",
    ) -> AssemblyResult:
        """Run the L1→L2→L3→L4 pipeline and assemble the system prompt."""
        self._reload_user_prompts()  # 热加载
        triggers = step_a_result.triggers
        topic_is_past = step_a_result.topic_is_past
        emotion_hint = step_a_result.emotion_hint

        logger.debug(f"StepA: triggers={triggers}, past={topic_is_past}, emotion={emotion_hint}, identity={identity}")

        # L1: Module selection
        l1_result = self._l1.select(triggers) if self._l1.is_loaded else None

        # L2: Rule matching (filtered by active L1 domains)
        domain_filter = l1_result.active_domains if l1_result else None
        l2_result = self._l2.match(triggers, domain_filter=domain_filter) if self._l2.is_loaded else None

        # L3: Scene retrieval
        # 内容检索：search_queries 或 用户原话
        # 关系检索：如果知道对方身份，单独检索互动场景
        search_queries = step_a_result.search_queries
        content_query = search_queries[0] if search_queries else user_message

        # 内容检索
        content_result = self._l3.retrieve(
            query=content_query,
            filter_tags=None,
            topic_is_past=topic_is_past,
        )

        if identity:
            # 关系检索：额外独立检索，不占内容名额
            relation_top_k = self._config.get_retrieval("l3_relation_top_k", 2)
            relation_result = self._l3.retrieve(
                query=f"艾莉丝和{identity}",
                filter_tags=None,
                topic_is_past=topic_is_past,
                top_k=relation_top_k,
            )
            # 去重后追加
            seen = set(content_result.scenes_used)
            extra_parts = []
            extra_ids = []
            if relation_result.prompt_text:
                for sid, part in zip(
                    relation_result.scenes_used,
                    relation_result.prompt_text.split("\n\n"),
                ):
                    if sid not in seen:
                        seen.add(sid)
                        extra_ids.append(sid)
                        extra_parts.append(part)

            from src.models import L3Result
            merged_text = content_result.prompt_text or ""
            if extra_parts:
                merged_text += "\n\n" + "\n\n".join(extra_parts)
            l3_result = L3Result(
                prompt_text=merged_text,
                scenes_used=content_result.scenes_used + extra_ids,
            )
        else:
            l3_result = content_result

        # L4: Update session state
        self._l4.update(
            user_id=sender_id,
            emotion_hint=emotion_hint,
            triggers=triggers,
            user_message=user_message,
        )
        l4_text = self._l4.format_state(sender_id)

        # 潜意识记忆检索（向量搜索，和 L3 并行但独立）
        subconscious_text = ""
        if self._subconscious and self._subconscious.enabled:
            # 用当前对话内容作为 query
            sc_query = user_message
            if conversation_context:
                sc_query = " ".join(conversation_context[-3:]) + " " + user_message
            subconscious_text = self._subconscious.retrieve(sender_id, sc_query)

        if subconscious_text:
            l4_text = l4_text + "\n" + subconscious_text if l4_text else subconscious_text

        user_prompt = self._get_user_prompt(sender_id, sender_nickname)

        # Assemble with token budget enforcement
        system_prompt = self._build_prompt(
            l1_text=l1_result.prompt_text if l1_result else "",
            l2_text=l2_result.prompt_text if l2_result else "",
            l3_text=l3_result.prompt_text if l3_result else "",
            l4_text=l4_text,
            user_prompt=user_prompt,
        )

        metadata = AssemblyMetadata(
            l1_modules_used=l1_result.modules_used if l1_result else [],
            l2_rules_used=l2_result.rules_used if l2_result else [],
            l3_scenes_used=l3_result.scenes_used,
            total_tokens=self._count_tokens(system_prompt),
        )

        return AssemblyResult(system_prompt=system_prompt, metadata=metadata)

    def _build_prompt(
        self,
        l1_text: str,
        l2_text: str,
        l3_text: str,
        l4_text: str,
        user_prompt: str = "",
    ) -> str:
        """Build the system prompt with token budget enforcement.

        Priority (never trimmed first):
          1. L1 core identity (highest priority)
          2. User custom prompt (用户身份/关系定义)
          3. L4 session state (情绪/话题)
          4. Meta-instruction
        Trimmed if over budget:
          5. L3 scenes (trimmed first)
          6. L2 rules (trimmed second)
        """
        # Fixed parts (always included)
        fixed_parts = []
        if l1_text:
            fixed_parts.append(f"## 核心身份\n{l1_text}")
        if user_prompt:
            fixed_parts.append(f"## 对话对象\n{user_prompt}")
        if l4_text:
            fixed_parts.append(f"## 当前状态\n{l4_text}")
        fixed_parts.append(META_INSTRUCTION)
        # 全局自定义要求（元指令后、行为规则前）
        custom = self._user_prompts.get("custom_instructions", "").strip()
        if custom:
            fixed_parts.append(f"## 用户自定义要求\n{custom}")

        fixed_text = "\n\n".join(fixed_parts)
        fixed_tokens = self._count_tokens(fixed_text)

        remaining_budget = self._target_tokens - fixed_tokens
        if remaining_budget <= 0:
            return fixed_text

        # Fit L2 and L3 into remaining budget
        l2_section = f"## 行为规则\n{l2_text}" if l2_text else ""
        l3_section = f"## 场景记忆\n{l3_text}" if l3_text else ""

        l2_tokens = self._count_tokens(l2_section)
        l3_tokens = self._count_tokens(l3_section)

        # Both fit
        if l2_tokens + l3_tokens <= remaining_budget:
            parts = [fixed_text]
            if l2_section:
                parts.insert(-1, l2_section)  # Before meta-instruction
            if l3_section:
                parts.insert(-1, l3_section)
            return "\n\n".join(parts)

        # Need to trim — trim L3 first
        l3_trimmed = self._trim_section(l3_section, remaining_budget - l2_tokens)
        if l2_tokens + self._count_tokens(l3_trimmed) <= remaining_budget:
            parts = [p for p in [fixed_text, l2_section, l3_trimmed] if p]
            return "\n\n".join(parts)

        # Still over — trim L2 as well
        l2_trimmed = self._trim_section(l2_section, remaining_budget - self._count_tokens(l3_trimmed))
        parts = [p for p in [fixed_text, l2_trimmed, l3_trimmed] if p]
        return "\n\n".join(parts)

    def _trim_section(self, section: str, max_tokens: int) -> str:
        """Trim a section by removing trailing paragraphs until it fits."""
        if max_tokens <= 0:
            return ""

        if self._count_tokens(section) <= max_tokens:
            return section

        paragraphs = section.split("\n\n")
        while len(paragraphs) > 1 and self._count_tokens("\n\n".join(paragraphs)) > max_tokens:
            paragraphs.pop()

        result = "\n\n".join(paragraphs)
        # Final hard truncation if single paragraph is still too long
        if self._count_tokens(result) > max_tokens:
            tokens = self._tokenizer.encode(result)[:max_tokens]
            result = self._tokenizer.decode(tokens)

        return result

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text))
