"""知识库：用户手写的知识条目，独立 collection，检索时优先于小说原文。

每条知识 = 一行文本 = 一个向量。修改 knowledge_base.txt 后运行 rebuild 即可。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb

if TYPE_CHECKING:
    from src.config import Config
    from src.embedding.embed_model import EmbeddingModel

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"


class KnowledgeBase:
    """手写知识库的向量检索。"""

    def __init__(self, config: Config, embed_model: EmbeddingModel):
        self._config = config
        self._embed_model = embed_model
        self._kb_path = Path(config.get("knowledge_base_path", "./data/knowledge_base.txt"))
        self._kb_mtime: float = 0

        persist_dir = config.vectordb_persist_dir
        self._chroma = chromadb.PersistentClient(path=persist_dir)

        # 启动时检查是否需要重建
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """如果 collection 不存在或文件更新了，自动重建。"""
        if not self._kb_path.exists():
            return
        mtime = self._kb_path.stat().st_mtime
        try:
            col = self._chroma.get_collection(COLLECTION_NAME)
            # 检查是否需要重建（文件修改时间变了）
            if col.count() > 0 and mtime == self._kb_mtime:
                return
        except Exception:
            pass
        self._kb_mtime = mtime
        self.rebuild()

    @property
    def count(self) -> int:
        try:
            return self._chroma.get_collection(COLLECTION_NAME).count()
        except Exception:
            return 0

    def rebuild(self) -> int:
        """从 knowledge_base.txt 重建向量 collection。"""
        entries = self._load_entries()
        if not entries:
            logger.info("知识库为空，跳过重建")
            return 0

        # 删旧建新
        try:
            self._chroma.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [f"kb_{i}" for i in range(len(entries))]
        embeddings = self._embed_model.encode_documents(entries).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=entries,
        )

        if self._kb_path.exists():
            self._kb_mtime = self._kb_path.stat().st_mtime

        logger.info(f"知识库重建完成: {len(entries)} 条")
        return len(entries)

    def retrieve(self, query: str, top_k: int = 2, llm_keywords: list[str] | None = None) -> list[str]:
        """检索知识库，返回相关条目。

        同时做语义检索和关键词精确匹配，合并去重。
        """
        try:
            collection = self._chroma.get_collection(COLLECTION_NAME)
        except Exception:
            return []

        if collection.count() == 0:
            return []

        results: list[str] = []
        seen: set[str] = set()

        # 关键词精确匹配优先（LLM keywords）
        for kw in (llm_keywords or []):
            if len(kw) < 2:
                continue
            try:
                kw_results = collection.get(
                    where_document={"$contains": kw},
                    include=["documents"],
                    limit=top_k,
                )
                for doc in kw_results.get("documents", []):
                    if doc not in seen:
                        seen.add(doc)
                        results.append(doc)
            except Exception:
                continue

        # 语义检索补充
        if len(results) < top_k:
            query_embedding = self._embed_model.encode_query(query).tolist()
            sem_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents"],
            )
            if sem_results["documents"] and sem_results["documents"][0]:
                for doc in sem_results["documents"][0]:
                    if doc not in seen:
                        seen.add(doc)
                        results.append(doc)

        return results[:top_k]

    def _load_entries(self) -> list[str]:
        """从 txt 文件加载知识条目，忽略空行和注释。"""
        if not self._kb_path.exists():
            return []
        lines = self._kb_path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    def check_reload(self) -> None:
        """热加载：文件变了自动重建。"""
        if not self._kb_path.exists():
            return
        mtime = self._kb_path.stat().st_mtime
        if mtime != self._kb_mtime:
            self.rebuild()
