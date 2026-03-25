"""AstrBot Star plugin: Bridge between AstrBot and the local Eris RAG server."""

import logging
import traceback

import httpx
from astrbot.api import logger as astrbot_logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register

logger = astrbot_logger.get_logger("eris_rag")


@register("eris_rag", "Eris RAG Plugin", "1.0.0", "Eris personality enhancement via local RAG server")
class ErisRAGPlugin(Star):

    def __init__(self, context: Context):
        super().__init__(context)
        self._rag_url: str = ""
        self._enabled: bool = False
        self._timeout: float = 5.0
        self._query_prefix: str = "/ask"
        self._client: httpx.AsyncClient | None = None

    async def initialize(self):
        """Load configuration and create HTTP client."""
        conf = self.context.get_config()
        self._rag_url = conf.get("rag_server_url", "http://localhost:8787").rstrip("/")
        self._enabled = conf.get("enabled", True)
        self._timeout = conf.get("timeout_ms", 5000) / 1000.0
        self._query_prefix = conf.get("query_command_prefix", "/ask")
        self._client = httpx.AsyncClient(timeout=self._timeout)
        logger.info(f"Eris RAG plugin initialized: url={self._rag_url}, enabled={self._enabled}")

    async def terminate(self):
        """Cleanup HTTP client."""
        if self._client:
            await self._client.aclose()

    @filter.on_llm_request()
    async def on_llm_request(self, event, req):
        """Intercept LLM requests to inject RAG-enhanced system prompt."""
        if not self._enabled or not self._client:
            return

        try:
            # Extract user message from the event
            user_msg = self._extract_user_message(event)
            if not user_msg:
                return

            # Check for direct query command
            if user_msg.startswith(self._query_prefix):
                # Will be handled in on_message instead
                return

            # Get conversation context
            context = self._extract_context(event)
            sender_id = self._extract_sender_id(event)

            # Call RAG server /retrieve
            response = await self._client.post(
                f"{self._rag_url}/retrieve",
                json={
                    "user_message": user_msg,
                    "conversation_context": context,
                    "sender_id": sender_id,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Inject enhanced system prompt
            enhanced_prompt = data.get("enhanced_system_prompt", "")
            if enhanced_prompt:
                req.system_prompt = enhanced_prompt
                logger.debug(
                    f"Injected RAG prompt ({data.get('metadata', {}).get('total_tokens', 0)} tokens)"
                )

        except httpx.TimeoutException:
            logger.warning("RAG server timeout — using original prompt")
        except Exception:
            logger.warning(f"RAG request failed — using original prompt\n{traceback.format_exc()}")

    @filter.command_group("ask")
    async def ask_command(self, event, ctx):
        """Handle /ask <query> for direct novel detail lookup."""
        if not self._enabled or not self._client:
            return

        query_text = ctx.get_args_text().strip()
        if not query_text:
            yield event.plain_result("用法: /ask <问题>\n例: /ask 艾莉丝什么时候开始练剑")
            return

        try:
            response = await self._client.post(
                f"{self._rag_url}/query",
                json={"query": query_text, "top_k": 3, "format": "raw"},
            )
            response.raise_for_status()
            data = response.json()

            raw_text = data.get("raw_text", "")
            if raw_text:
                yield event.plain_result(f"📖 相关小说片段:\n\n{raw_text}")
            else:
                yield event.plain_result("未找到相关内容。")

        except httpx.TimeoutException:
            yield event.plain_result("RAG 服务器超时，请稍后再试。")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            yield event.plain_result("查询失败，请检查 RAG 服务器状态。")

    @staticmethod
    def _extract_user_message(event) -> str:
        """Extract the user's text message from an AstrBot event."""
        if hasattr(event, "message_str"):
            return event.message_str
        if hasattr(event, "get_messages"):
            messages = event.get_messages()
            if messages:
                return messages[-1].get("content", "")
        return ""

    @staticmethod
    def _extract_context(event) -> list[str]:
        """Extract recent conversation context."""
        if hasattr(event, "get_messages"):
            messages = event.get_messages()
            return [m.get("content", "") for m in messages[-5:] if m.get("content")]
        return []

    @staticmethod
    def _extract_sender_id(event) -> str:
        """Extract sender identifier."""
        if hasattr(event, "get_sender_id"):
            return str(event.get_sender_id())
        if hasattr(event, "session_id"):
            return str(event.session_id)
        return "default"
