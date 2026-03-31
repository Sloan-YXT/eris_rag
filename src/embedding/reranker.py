"""BGE-reranker wrapper — 用 sentence-transformers CrossEncoder 加载。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker (默认 BGE-reranker-v2-m3)。"""

    def __init__(self, config: Config, use_train_device: bool = False):
        self._model_name = config.reranker_model
        if use_train_device:
            self._device = config.get("reranker.train_device", config.reranker_device)
        else:
            self._device = config.reranker_device
        self._model = None

    def load(self) -> None:
        """加载 reranker 模型。"""
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker {self._model_name} on {self._device}")
        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
        )
        logger.info("Reranker loaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """对 documents 按与 query 的相关性排序。

        Returns:
            List of (original_index, score) 按分数降序。
        """
        if self._model is None:
            raise RuntimeError("Reranker not loaded — call load() first")

        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        indexed_scores = list(enumerate(float(s) for s in scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores
