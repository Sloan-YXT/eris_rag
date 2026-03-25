"""潜意识记忆系统：per-user 向量检索 + 异步写入。

每个用户一个 JSON 文件（持久化）+ 一个 ChromaDB collection（向量检索）。
记忆提取在响应返回后异步执行，不阻塞请求。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb

if TYPE_CHECKING:
    from src.config import Config
    from src.embedding.embed_model import EmbeddingModel

logger = logging.getLogger(__name__)

# 提取 prompt：让 LLM 判断是否值得存入潜意识
EXTRACT_SYSTEM = "你是一个记忆管理模块。请严格按 JSON 格式输出。"

EXTRACT_PROMPT = """\
你是艾莉丝·格雷拉特。以下是你和对方刚刚的对话。
请判断这轮对话中是否发生了对你来说**人生级别的重要事件**。

重要的标准：
- 关于家人的重大变化（怀孕、出生、受伤、离世）
- 关系的重大转变（告白、结婚、决裂、重逢）
- 改变你行为或认知的事件
- 对方做出的重要承诺或决定
- 对方人生中已经发生的重要变化（辞职了、考上研了、搬家了、毕业了、失恋了等）

**不算重要的：**
- 日常闲聊、开玩笑、普通问候
- 只是计划或打算做的事（"我想辞职"不算，"我已经辞职了"才算）

## 对方身份
{identity}

## 本轮对话
{conversation}

## 输出
如果有重要记忆，输出：
{{"important": true, "memory": "用一句话概括这个重要事件"}}

如果没有：
{{"important": false}}

输出纯 JSON。"""


class SubconsciousMemory:
    """Per-user 潜意识记忆：向量检索 + JSON 持久化。"""

    def __init__(self, config: Config, embed_model: EmbeddingModel):
        self._config = config
        self._embed_model = embed_model
        self._enabled = config.get("memory.enabled", False)
        self._storage_dir = Path(config.get("memory.storage_dir", "./data/memories"))
        self._collection_prefix = config.get("memory.vectordb_prefix", "subconscious")
        self._provider = config.get("memory.provider", "deepseek-v3")
        self._top_k = config.get_retrieval("subconscious_top_k", 3)

        # ChromaDB client（复用 L3 的 persist_dir）
        persist_dir = config.vectordb_persist_dir
        self._chroma = chromadb.PersistentClient(path=persist_dir)

        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── 检索（同步，在 L4 阶段调用）──────────────────────────

    def retrieve(self, sender_id: str, query: str) -> str:
        """检索与当前对话相关的潜意识记忆。

        Args:
            sender_id: 用户 QQ 号
            query: 当前对话上下文（用于语义匹配）

        Returns:
            格式化的记忆文本，直接注入 system_prompt。
        """
        if not self._enabled or not sender_id or sender_id == "default":
            return ""

        collection_name = f"{self._collection_prefix}_{sender_id}"
        try:
            collection = self._chroma.get_collection(collection_name)
        except Exception:
            return ""  # 该用户还没有潜意识记忆

        if collection.count() == 0:
            return ""

        query_embedding = self._embed_model.encode_query(query).tolist()
        top_k = min(self._top_k, collection.count())

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        if not results["ids"] or not results["ids"][0]:
            return ""

        parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            date = meta.get("created", "")
            parts.append(f"【潜意识】{doc}（{date}）")

        return "\n".join(parts)

    # ── 写入（异步，在响应返回后执行）────────────────────────

    async def extract_and_store(
        self,
        sender_id: str,
        identity: str,
        user_message: str,
        bot_reply: str,
        conversation_context: list[str] | None = None,
    ) -> bool:
        """异步提取潜意识记忆并存储。

        Returns:
            是否存入了新记忆。
        """
        if not self._enabled or not sender_id or sender_id == "default":
            return False

        # 构建对话文本
        conv_parts = []
        if conversation_context:
            for i, msg in enumerate(conversation_context[-4:]):
                role = "对方" if i % 2 == 0 else "你"
                conv_parts.append(f"{role}: {msg}")
        conv_parts.append(f"对方: {user_message}")
        if bot_reply:
            conv_parts.append(f"你: {bot_reply}")
        conversation_text = "\n".join(conv_parts)

        identity_text = identity if identity else "未知身份"

        prompt = EXTRACT_PROMPT.replace("{identity}", identity_text)
        prompt = prompt.replace("{conversation}", conversation_text)

        # 调 LLM
        from src.llm.client import LLMClient
        client = LLMClient(self._config)
        try:
            response = await client.complete(
                provider=self._provider,
                system_prompt=EXTRACT_SYSTEM,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=4096,
            )
        except Exception as e:
            logger.error(f"潜意识提取失败: {e}")
            return False
        finally:
            await client.close()

        # 解析
        result = self._parse_response(response)
        if not result or not result.get("important"):
            return False

        memory_text = result.get("memory", "").strip()
        if not memory_text:
            return False

        # 存储 + 重建向量
        self._store_memory(sender_id, memory_text)
        self._rebuild_collection(sender_id)

        logger.info(f"潜意识记忆存入 [{sender_id}]: {memory_text}")
        return True

    # ── 持久化 ───────────────────────────────────────────────

    def _get_storage_path(self, sender_id: str) -> Path:
        return self._storage_dir / f"{sender_id}.json"

    def _load_memories(self, sender_id: str) -> list[dict]:
        path = self._get_storage_path(sender_id)
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("memories", [])

    def _store_memory(self, sender_id: str, memory_text: str) -> None:
        memories = self._load_memories(sender_id)
        memories.append({
            "id": len(memories) + 1,
            "text": memory_text,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        })
        path = self._get_storage_path(sender_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sender_id": sender_id, "memories": memories},
                      f, ensure_ascii=False, indent=2)

    # ── 向量库管理 ────────────────────────────────────────────

    def _rebuild_collection(self, sender_id: str) -> None:
        """重建某个用户的潜意识向量 collection。"""
        collection_name = f"{self._collection_prefix}_{sender_id}"

        # 删除旧的
        try:
            self._chroma.delete_collection(collection_name)
        except Exception:
            pass

        memories = self._load_memories(sender_id)
        if not memories:
            return

        # 创建新的
        collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 每条记忆 = 一个向量（短文本，不需要分块）
        texts = [m["text"] for m in memories]
        ids = [f"mem_{m['id']}" for m in memories]
        metadatas = [{"created": m["created"]} for m in memories]
        embeddings = self._embed_model.encode_documents(texts).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"潜意识向量库重建 [{sender_id}]: {len(memories)} 条")

    def rebuild_all(self) -> None:
        """重建所有用户的潜意识向量库（手动工具用）。"""
        for json_file in self._storage_dir.glob("*.json"):
            sender_id = json_file.stem
            self._rebuild_collection(sender_id)
            logger.info(f"重建 {sender_id}")

    # ── 工具 ──────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response: str) -> dict | None:
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
