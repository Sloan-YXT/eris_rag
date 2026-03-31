"""L3 Episodic Memory: ChromaDB 向量库，基于原文分块的语义检索。

embedding 输入 = Chunk.raw_text（小说原文），不是 LLM 生成的浓缩文本。
metadata 用于过滤（situation_tags, period 等），由 Phase 2 标注器填充。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
import yaml

from src.models import Chunk, L3Result

if TYPE_CHECKING:
    from src.config import Config
    from src.embedding.embed_model import EmbeddingModel
    from src.embedding.reranker import Reranker

logger = logging.getLogger(__name__)


class L3EpisodicMemory:
    """原文分块的向量存储与检索。"""

    def __init__(
        self,
        config: Config,
        embed_model: EmbeddingModel,
        reranker: Reranker | None = None,
    ):
        self._config = config
        self._embed_model = embed_model
        self._reranker = reranker

        self._candidates = config.get_retrieval("l3_candidates", 6)
        self._top_k = config.get_retrieval("l3_top_k", 3)

        # Period weights
        self._period_weights: dict[str, dict[str, float]] = {}
        self._load_period_weights()

        # ChromaDB
        persist_dir = config.vectordb_persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=config.vectordb_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def _load_period_weights(self) -> None:
        taxonomy_path = self._config.taxonomy_path
        if taxonomy_path.exists():
            self._taxonomy_mtime = taxonomy_path.stat().st_mtime
            with open(taxonomy_path, encoding="utf-8") as f:
                taxonomy = yaml.safe_load(f)
            self._period_weights = taxonomy.get("period_weights", {})

    def _check_reload_weights(self) -> None:
        """热加载 period_weights。"""
        taxonomy_path = self._config.taxonomy_path
        if taxonomy_path.exists():
            mtime = taxonomy_path.stat().st_mtime
            if mtime != getattr(self, "_taxonomy_mtime", 0):
                self._load_period_weights()
                logger.info("taxonomy period_weights reloaded")

    @property
    def scene_count(self) -> int:
        return self._collection.count()

    # ── 入库 ─────────────────────────────────────────────────

    def ingest_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """将原文分块 embed 并存入 ChromaDB。

        Returns:
            新增条目数（跳过已存在的）。
        """
        existing_ids = set(self._collection.get()["ids"]) if self._collection.count() > 0 else set()
        new_chunks = [c for c in chunks if c.id not in existing_ids]

        if not new_chunks:
            logger.info("没有新的 chunk 需要入库")
            return 0

        logger.info(f"入库 {len(new_chunks)} 个 chunks（跳过 {len(chunks) - len(new_chunks)} 个已存在）")

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            texts = [c.raw_text for c in batch]
            embeddings = self._embed_model.encode_documents(texts, batch_size=batch_size)

            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings.tolist(),
                documents=texts,  # 存原文，检索后直接用
                metadatas=[self._chunk_to_metadata(c) for c in batch],
            )

        logger.info(f"入库完成，总计: {self._collection.count()}")
        return len(new_chunks)

    def update_metadata(self, chunk_id: str, metadata_update: dict) -> None:
        """更新单个 chunk 的 metadata（用于 Phase 2 标注）。"""
        existing = self._collection.get(ids=[chunk_id], include=["metadatas"])
        if not existing["ids"]:
            return
        current_meta = existing["metadatas"][0]
        current_meta.update(metadata_update)
        self._collection.update(ids=[chunk_id], metadatas=[current_meta])

    def update_metadata_batch(self, ids: list[str], metadatas: list[dict]) -> None:
        """批量更新 metadata。"""
        self._collection.update(ids=ids, metadatas=metadatas)

    # ── 检索 ─────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filter_tags: list[str] | None = None,
        topic_is_past: bool = False,
        top_k: int | None = None,
        n_candidates: int | None = None,
        llm_keywords: list[str] | None = None,
    ) -> L3Result:
        """检索与 query 相关的原文片段。

        Args:
            query: 用户消息或检索查询。
            filter_tags: 从 L2 传来的 situation_tags，用于 metadata 过滤。
            topic_is_past: 如果为 true，反转 period_weight 以偏向早期。
            top_k: 最终返回数量。
            n_candidates: rerank 前的粗排数量。

        Returns:
            L3Result，prompt_text 中包含格式化的原文片段。
        """
        self._check_reload_weights()
        top_k = top_k or self._top_k
        n_candidates = n_candidates or self._candidates

        if self._collection.count() == 0:
            return L3Result()

        query_embedding = self._embed_model.encode_query(query).tolist()

        # Metadata 过滤
        where_filter = self._build_tag_filter(filter_tags) if filter_tags else None

        # 路线1: 语义检索（先尝试 tag 过滤，无结果则去掉过滤）
        results = None
        if where_filter:
            try:
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_candidates,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
                # 过滤后无结果 → 回退无过滤
                if not results["ids"] or not results["ids"][0]:
                    results = None
            except Exception:
                results = None

        if results is None:
            try:
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_candidates,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return L3Result()

        # 路线2: 关键词精确匹配
        keyword_hits = self._keyword_search(query, n_candidates, llm_keywords=llm_keywords)

        # 合并语义结果 + 关键词结果
        seen_ids: set[str] = set()
        ids = []
        documents = []
        metadatas = []
        distances = []

        if results["ids"] and results["ids"][0]:
            for doc_id, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0],
                results["metadatas"][0], results["distances"][0],
            ):
                seen_ids.add(doc_id)
                ids.append(doc_id)
                documents.append(doc)
                metadatas.append(meta)
                distances.append(dist)

        keyword_hit_ids: set[str] = set()  # 标记关键词精确命中的 ID
        for kw_hit in keyword_hits:
            if kw_hit["scene_id"] not in seen_ids:
                seen_ids.add(kw_hit["scene_id"])
                keyword_hit_ids.add(kw_hit["scene_id"])
                ids.append(kw_hit["scene_id"])
                documents.append(kw_hit["text"])
                metadatas.append({
                    "volume": kw_hit["volume"],
                    "chapter": kw_hit["chapter"],
                    "period": kw_hit.get("period", ""),
                    "significance": kw_hit.get("significance", 0.0),
                })
                distances.append(0.5)

        if not ids:
            return L3Result()

        # Period weight 加权
        weight_key = "topic_is_past" if topic_is_past else "default"
        weights = self._period_weights.get(weight_key, {})

        scored = []
        for idx, (doc_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
            similarity = 1.0 - dist
            period = meta.get("period", "")
            period_weight = weights.get(period, 1.0)
            sig = meta.get("significance", 0.0)
            sig_boost = 1.0 + sig * 0.5
            weighted_score = similarity * period_weight * sig_boost
            scored.append((idx, doc_id, doc, meta, weighted_score))

        scored.sort(key=lambda x: x[4], reverse=True)

        # Rerank — 关键词精确命中的结果强制保留
        # 先把关键词命中的提出来，剩下的交给 reranker 排序
        keyword_entries = [(s[1], s[2], s[3], s[4]) for s in scored if s[1] in keyword_hit_ids]
        non_keyword = [s for s in scored if s[1] not in keyword_hit_ids]

        # 关键词命中优先：有命中时占大多数位置，语义结果补充
        kw_slots = min(len(keyword_entries), max(1, top_k - 1))
        remaining_slots = top_k - kw_slots

        if self._reranker and self._reranker.is_loaded and len(non_keyword) > remaining_slots:
            candidate_docs = [s[2] for s in non_keyword]
            reranked = self._reranker.rank(query, candidate_docs, top_k=remaining_slots)
            reranked_entries = [(non_keyword[orig_idx][1], non_keyword[orig_idx][2],
                                non_keyword[orig_idx][3], score)
                               for orig_idx, score in reranked]
        else:
            reranked_entries = [(s[1], s[2], s[3], s[4]) for s in non_keyword[:remaining_slots]]

        # 关键词命中排前面
        final = keyword_entries[:kw_slots] + reranked_entries
        kw_id_set = {e[0] for e in keyword_entries[:kw_slots]}

        # 格式化输出
        prompt_parts = []
        scene_ids = []
        for doc_id, doc, meta, score in final:
            scene_ids.append(doc_id)
            period = meta.get("period", "")
            target_period = self._config.target_period
            formatted = self._format_chunk(doc, meta, period, target_period)
            # 关键词命中的场景加标记，点明匹配到的关键词
            if doc_id in kw_id_set:
                # 找出这个 chunk 里命中了哪些关键词
                import re as _re
                query_kws = _re.findall(r"[\u4e00-\u9fff]{2,}", query)
                matched = [kw for kw in query_kws if kw in doc]
                if matched:
                    kw_str = "、".join(matched[:3])
                    formatted = f"★以下原文提到了「{kw_str}」，这是事实，请据此回答★\n" + formatted
                else:
                    formatted = "★以下内容与用户的问题直接相关★\n" + formatted
            prompt_parts.append(formatted)

        prompt_text = "\n\n---SCENE---\n\n".join(prompt_parts) if prompt_parts else ""
        return L3Result(prompt_text=prompt_text, scenes_used=scene_ids)

    def retrieve_raw(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """独立检索模式：语义搜索 + 关键词回退，返回原始数据。"""
        if self._collection.count() == 0:
            return []

        query_embedding = self._embed_model.encode_query(query).tolist()

        # 路线1: 语义检索
        semantic_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # 路线2: 关键词精确匹配（从 query 中提取 2-3 字中文片段）
        import re as _re
        fallback_kws = [s for s in _re.findall(r"[\u4e00-\u9fff]{2,4}", query) if len(s) <= 4]
        keyword_results = self._keyword_search(query, top_k, llm_keywords=fallback_kws)

        # 合并去重，语义优先
        seen_ids: set[str] = set()
        output: list[dict] = []

        if semantic_results["ids"] and semantic_results["ids"][0]:
            for doc_id, doc, meta, dist in zip(
                semantic_results["ids"][0], semantic_results["documents"][0],
                semantic_results["metadatas"][0], semantic_results["distances"][0],
            ):
                seen_ids.add(doc_id)
                output.append({
                    "scene_id": doc_id,
                    "volume": meta.get("volume", 0),
                    "chapter": meta.get("chapter", 0),
                    "summary": doc[:100],
                    "text": doc,
                    "score": 1.0 - dist,
                    "match_type": "semantic",
                })

        kw_items: list[dict] = []
        for item in keyword_results:
            if item["scene_id"] not in seen_ids:
                seen_ids.add(item["scene_id"])
                kw_items.append(item)

        # Rerank — 关键词命中强制保留
        semantic_items = [r for r in output]  # 已有的语义结果
        kw_slots = min(len(kw_items), max(1, top_k // 2))
        remaining_slots = top_k - kw_slots

        if self._reranker and self._reranker.is_loaded and len(semantic_items) > remaining_slots:
            docs = [r["text"] for r in semantic_items]
            reranked = self._reranker.rank(query, docs, top_k=remaining_slots)
            semantic_items = [semantic_items[idx] for idx, _ in reranked]
        else:
            semantic_items = semantic_items[:remaining_slots]

        output = kw_items[:kw_slots] + semantic_items
        return output

    # ── 关键词检索 ───────────────────────────────────────────

    def _keyword_search(self, query: str, top_k: int, llm_keywords: list[str] | None = None) -> list[dict]:
        """用 where_document $contains 做原文精确匹配。

        使用 LLM 提取的 keywords（专有名词），直接 $contains 查询。
        过滤主角名等超高频词（匹配所有 chunk 无检索价值）。
        """
        # 主角名出现在几乎所有 chunk 里，搜它等于没搜
        _HIGH_FREQ = {"艾莉丝", "艾丽丝", "鲁迪乌斯", "鲁迪", "格雷拉特",
                      "洛琪希", "希露菲", "基列奴"}
        keywords = [k for k in (llm_keywords or []) if len(k) >= 2 and k not in _HIGH_FREQ]
        if not keywords:
            return []

        seen_ids: set[str] = set()
        output: list[dict] = []

        # 用 query embedding 做语义排序（关键词匹配 + 语义相关性）
        query_embedding = self._embed_model.encode_query(query).tolist()

        for kw in keywords:
            try:
                # query + where_document：精确包含关键词 且 按语义相关性排序
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    where_document={"$contains": kw},
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                continue

            if not results["ids"] or not results["ids"][0]:
                continue

            for doc_id, doc, meta in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0],
            ):
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                output.append({
                    "scene_id": doc_id,
                    "volume": meta.get("volume", 0),
                    "chapter": meta.get("chapter", 0),
                    "period": meta.get("period", ""),
                    "significance": meta.get("significance", 0.0),
                    "summary": doc[:100],
                    "text": doc,
                    "score": 0.5,  # 关键词命中基础分
                    "match_type": "keyword",
                })

        return output

    # ── 内部工具 ─────────────────────────────────────────────

    # 文本匹配判断是否涉及艾莉丝
    _ERIS_NAMES = {"艾莉丝", "エリス", "Eris", "艾丽丝"}

    @classmethod
    def _chunk_to_metadata(cls, chunk: Chunk) -> dict:
        """构建 ChromaDB metadata。has_eris 入库时直接文本匹配判断。"""
        has_eris = any(name in chunk.raw_text for name in cls._ERIS_NAMES)
        return {
            "volume": chunk.volume,
            "chapter": chunk.chapter,
            "period": chunk.period,
            "period_weight": chunk.period_weight,
            "char_offset": chunk.char_offset,
            "has_eris": has_eris,
            "situation_tags": "",      # Phase 2 标注后填充
            "significance": 0.0,       # Phase 2 核心记忆标注后填充 (0-1)
        }

    @staticmethod
    def _build_tag_filter(tags: list[str]) -> dict | None:
        if not tags:
            return None
        if len(tags) == 1:
            return {"situation_tags": {"$contains": tags[0]}}
        return {
            "$or": [{"situation_tags": {"$contains": tag}} for tag in tags]
        }

    @staticmethod
    def _format_chunk(
        document: str,
        metadata: dict,
        chunk_period: str,
        target_period: str,
    ) -> str:
        """按 period 选择输出模板。当前时期 → 行为参考；早期 → 你的记忆。"""
        period_order = ["少女期", "魔大陆流浪期", "剑之圣地期", "回归后"]
        chunk_idx = period_order.index(chunk_period) if chunk_period in period_order else -1
        target_idx = period_order.index(target_period) if target_period in period_order else len(period_order) - 1

        vol = metadata.get("volume", "?")

        if chunk_idx >= target_idx or chunk_idx == -1:
            header = f"【行为参考 · 第{vol}卷】"
            emotion = metadata.get("emotion", "")
            if emotion:
                header += f"（{emotion}）"
            return f"{header}\n{document}"
        else:
            header = f"【你的记忆 · {chunk_period} · 第{vol}卷】"
            return f"{header}\n{document}"
