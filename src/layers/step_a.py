"""Step A: Intent analysis — jieba (local) or LLM (API), configurable."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.models import StepAResult

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

# Tags that indicate the user is talking about the past
PAST_TAGS = {"回忆", "对比过去"}

# LLM prompt for intent analysis
_LLM_SYSTEM = "你是一个意图分析模块。请严格按指定 JSON 格式输出，不要输出任何其他内容。"

_LLM_TEMPLATE = """\
分析以下用户消息，提取情境标签，并生成用于检索小说原文的关键词。

角色背景：艾莉丝·格雷拉特，《无职转生》中的角色，剑士，红发贵族少女。

## 可用标签（只能从中选择）

{tag_list}

## 用户消息

{user_message}

## 对话上下文（最近几轮）

{context}

## 输出要求

输出纯 JSON，格式如下：
```json
{{"triggers": ["标签1", "标签2"], "topic_is_past": false, "emotion_hint": "", "search_queries": ["检索词1", "检索词2"]}}
```

- triggers: 从上方可用标签中选择所有匹配的标签（可以多选，通常 1-4 个）
- topic_is_past: 用户是否在谈论过去/回忆（true/false）
- emotion_hint: 最突出的情绪标签，没有明显情绪则留空字符串
- search_queries: 2-3 个用于检索小说原文的关键词短语，要覆盖用户消息的核心意图。例如用户说"你送给鲁迪的第一个生日礼物"，应生成["艾莉丝 鲁迪乌斯 生日 礼物", "庆生会 魔杖", "傲慢水龙王"]

输出纯 JSON，不要 markdown 标记。"""


class StepA:
    """Intent analysis: extract situation triggers from user message.

    Supports two backends:
      - "local": jieba keyword matching (free, fast, ~50ms, less accurate)
      - any provider name (e.g. "gemini"): LLM API call (accurate, ~500ms, costs API)

    Configured via `api.runtime.step_a` in config.yaml.
    """

    def __init__(self, config: Config):
        self._config = config
        self._mode = config.get_runtime_provider("step_a")  # "local" or provider name
        self._keyword_dict: dict[str, str] = {}
        self._past_keywords: set[str] = set()
        self._all_tags: list[str] = []
        self._loaded = False
        self._taxonomy_mtime: float = 0

    def _check_reload(self) -> None:
        """热加载：taxonomy 文件变了就重新加载。"""
        path = self._config.taxonomy_path
        if not path.exists():
            return
        mtime = path.stat().st_mtime
        if mtime != self._taxonomy_mtime:
            self.load()

    def load(self, taxonomy_path: str | Path | None = None) -> None:
        """Load taxonomy. For local mode also registers keywords with jieba."""
        path = Path(taxonomy_path) if taxonomy_path else self._config.taxonomy_path

        if not path.exists():
            logger.warning(f"Taxonomy file not found: {path}")
            return

        with open(path, encoding="utf-8") as f:
            taxonomy = yaml.safe_load(f)

        self._all_tags = taxonomy.get("situations", [])
        self._keyword_dict = taxonomy.get("keyword_dict", {})
        self._past_keywords.clear()

        for keyword, tag in self._keyword_dict.items():
            if tag in PAST_TAGS:
                self._past_keywords.add(keyword)

        # Register keywords with jieba regardless of mode (cheap, helps fallback)
        try:
            import jieba
            for keyword in self._keyword_dict:
                jieba.add_word(keyword)
        except ImportError:
            if self._mode == "local":
                logger.error("jieba not installed but step_a mode is 'local'")
                return

        self._loaded = True
        path = Path(taxonomy_path) if taxonomy_path else self._config.taxonomy_path
        if path.exists():
            self._taxonomy_mtime = path.stat().st_mtime
        logger.info(f"StepA loaded, mode={self._mode}, {len(self._keyword_dict)} keywords, {len(self._all_tags)} tags")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Public API ───────────────────────────────────────────

    def analyze(self, user_message: str, conversation_context: list[str] | None = None) -> StepAResult:
        """Synchronous entry — dispatches to local or wraps async LLM call."""
        self._check_reload()
        if self._mode == "local":
            return self._analyze_local(user_message, conversation_context)
        else:
            # Run the async LLM path in a sync wrapper
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already inside an event loop (e.g. FastAPI) — use local as hot path,
                # the server endpoint can call analyze_async directly
                return self._analyze_local(user_message, conversation_context)
            else:
                return asyncio.run(self.analyze_async(user_message, conversation_context))

    async def analyze_async(self, user_message: str, conversation_context: list[str] | None = None) -> StepAResult:
        """Async entry — uses LLM when configured, falls back to local on error."""
        self._check_reload()
        if self._mode == "local":
            return self._analyze_local(user_message, conversation_context)

        try:
            return await self._analyze_llm(user_message, conversation_context)
        except Exception as e:
            logger.warning(f"LLM step_a failed ({e}), falling back to local")
            return self._analyze_local(user_message, conversation_context)

    # ── Local (jieba) backend ────────────────────────────────

    def _analyze_local(self, user_message: str, conversation_context: list[str] | None = None) -> StepAResult:
        """Jieba tokenization + keyword dict + substring fallback."""
        import jieba

        tokens = list(jieba.cut(user_message))
        triggers: set[str] = set()
        topic_is_past = False

        # Token-level matching
        for token in tokens:
            token = token.strip()
            if token in self._keyword_dict:
                tag = self._keyword_dict[token]
                triggers.add(tag)
                if token in self._past_keywords:
                    topic_is_past = True

        # Substring fallback: check if any keyword appears as a substring
        # This catches cases where jieba merges keywords (e.g. "别生气" not split into "生气")
        for keyword, tag in self._keyword_dict.items():
            if len(keyword) >= 2 and keyword in user_message and tag not in triggers:
                triggers.add(tag)
                if keyword in self._past_keywords:
                    topic_is_past = True

        # Check conversation context for past-topic signals
        if conversation_context:
            for msg in conversation_context[-2:]:
                for keyword in self._past_keywords:
                    if keyword in msg:
                        topic_is_past = True
                        triggers.add(self._keyword_dict[keyword])

        emotion_hint = self._detect_emotion(triggers)

        return StepAResult(
            triggers=sorted(triggers),
            topic_is_past=topic_is_past,
            emotion_hint=emotion_hint,
        )

    # ── LLM backend ─────────────────────────────────────────

    async def _analyze_llm(self, user_message: str, conversation_context: list[str] | None = None) -> StepAResult:
        """Call LLM API to extract triggers — more accurate than jieba."""
        from src.llm.client import LLMClient

        tag_list = ", ".join(self._all_tags)
        context_str = "\n".join(conversation_context[-3:]) if conversation_context else "(无)"

        prompt = _LLM_TEMPLATE.format(
            tag_list=tag_list,
            user_message=user_message,
            context=context_str,
        )

        client = LLMClient(self._config)
        try:
            response = await client.complete(
                provider=self._mode,
                system_prompt=_LLM_SYSTEM,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=4096,
            )
        finally:
            await client.close()

        return self._parse_llm_response(response)

    def _parse_llm_response(self, response: str) -> StepAResult:
        """Parse LLM JSON output into StepAResult, with validation."""
        text = response.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        data = json.loads(text)

        # Validate triggers against known tags
        valid_tags = set(self._all_tags)
        triggers = [t for t in data.get("triggers", []) if t in valid_tags]
        topic_is_past = bool(data.get("topic_is_past", False))
        emotion_hint = data.get("emotion_hint", "")
        if emotion_hint and emotion_hint not in valid_tags:
            emotion_hint = ""
        search_queries = [str(q) for q in data.get("search_queries", []) if q]

        return StepAResult(
            triggers=sorted(triggers),
            topic_is_past=topic_is_past,
            emotion_hint=emotion_hint,
            search_queries=search_queries,
        )

    # ── Shared helpers ───────────────────────────────────────

    @staticmethod
    def _detect_emotion(triggers: set[str]) -> str:
        """Pick the most salient emotion from triggers."""
        emotion_priority = [
            "愤怒", "恐惧", "悲伤", "哭泣", "孤独", "无力感",
            "心动", "嫉妒", "羞耻", "焦虑",
            "喜悦", "感谢",
        ]
        for emotion in emotion_priority:
            if emotion in triggers:
                return emotion
        return ""
