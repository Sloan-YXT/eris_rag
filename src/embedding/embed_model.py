"""BGE-large-zh embedding model wrapper with GPU support and batch encoding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper around sentence-transformers for BGE-large-zh-v1.5."""

    def __init__(self, config: Config, use_train_device: bool = False):
        self._model_name = config.embedding_model
        if use_train_device:
            self._device = config.get("embedding.train_device", config.embedding_device)
        else:
            self._device = config.embedding_device
        self._batch_size = config.get("embedding.batch_size", 64)
        self._model = None  # SentenceTransformer instance, loaded lazily

    def load(self) -> None:
        """Load model onto the configured device."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model {self._model_name} on {self._device}")
        self._model = SentenceTransformer(self._model_name, device=self._device, trust_remote_code=True)
        logger.info("Embedding model loaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int:
        if self._model is None:
            raise RuntimeError("Embedding model not loaded")
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Override default batch size.
            normalize: L2-normalize embeddings (required for cosine similarity).

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        if self._model is None:
            raise RuntimeError("Embedding model not loaded — call load() first")

        bs = batch_size or self._batch_size
        # BGE models benefit from the "query: " / "passage: " prefix convention,
        # but for Chinese BGE-large-zh it's not required. We keep texts as-is.
        embeddings = self._model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=len(texts) > bs,
            normalize_embeddings=normalize,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns 1-D array."""
        return self.encode([query], normalize=True)[0]

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode a batch of documents. Returns 2-D array."""
        return self.encode(documents, batch_size=batch_size, normalize=True)
