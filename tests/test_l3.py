"""Tests for L3 Episodic Memory: chunk 存储和检索。"""

import pytest
from unittest.mock import MagicMock

import numpy as np

from src.models import Chunk
from src.layers.l3_episodic import L3EpisodicMemory


def make_chunk(chunk_id: str, volume: int = 1, period: str = "回归后", **kwargs) -> Chunk:
    defaults = {
        "id": chunk_id,
        "raw_text": f"这是测试文本 {chunk_id}，艾莉丝在做某件事。",
        "volume": volume,
        "chapter": 1,
        "period": period,
        "period_weight": 1.0,
    }
    defaults.update(kwargs)
    return Chunk(**defaults)


@pytest.fixture
def mock_embed_model():
    model = MagicMock()
    model.is_loaded = True
    model.dimension = 1024

    def fake_encode_query(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(1024).astype(np.float32)

    def fake_encode_documents(texts, batch_size=None):
        embeddings = []
        for t in texts:
            np.random.seed(hash(t) % 2**32)
            embeddings.append(np.random.randn(1024).astype(np.float32))
        return np.array(embeddings)

    model.encode_query = fake_encode_query
    model.encode_documents = fake_encode_documents
    return model


class TestL3EpisodicMemory:
    def test_ingest_chunks(self, tmp_config, mock_embed_model):
        l3 = L3EpisodicMemory(tmp_config, mock_embed_model)
        chunks = [make_chunk("c01"), make_chunk("c02"), make_chunk("c03")]
        count = l3.ingest_chunks(chunks)
        assert count == 3
        assert l3.scene_count == 3

    def test_no_duplicate_ingestion(self, tmp_config, mock_embed_model):
        l3 = L3EpisodicMemory(tmp_config, mock_embed_model)
        chunks = [make_chunk("c01"), make_chunk("c02")]
        l3.ingest_chunks(chunks)
        count = l3.ingest_chunks(chunks)
        assert count == 0
        assert l3.scene_count == 2

    def test_retrieve_basic(self, tmp_config, mock_embed_model):
        l3 = L3EpisodicMemory(tmp_config, mock_embed_model)
        chunks = [
            make_chunk("c01", raw_text="艾莉丝在练剑"),
            make_chunk("c02", raw_text="艾莉丝在吃饭"),
            make_chunk("c03", raw_text="艾莉丝在战斗"),
        ]
        l3.ingest_chunks(chunks)

        result = l3.retrieve("练剑")
        assert len(result.scenes_used) > 0
        assert result.prompt_text != ""

    def test_retrieve_empty_db(self, tmp_config, mock_embed_model):
        l3 = L3EpisodicMemory(tmp_config, mock_embed_model)
        result = l3.retrieve("anything")
        assert result.scenes_used == []
        assert result.prompt_text == ""

    def test_retrieve_raw_standalone(self, tmp_config, mock_embed_model):
        l3 = L3EpisodicMemory(tmp_config, mock_embed_model)
        chunks = [make_chunk("c01", raw_text="艾莉丝的日常")]
        l3.ingest_chunks(chunks)

        results = l3.retrieve_raw("日常")
        assert len(results) > 0
        assert "scene_id" in results[0]

    def test_period_template_current(self, tmp_config, mock_embed_model):
        l3 = L3EpisodicMemory(tmp_config, mock_embed_model)
        chunks = [make_chunk("c01", period="回归后", volume=18, raw_text="回归后的艾莉丝")]
        l3.ingest_chunks(chunks)

        result = l3.retrieve("test")
        if result.prompt_text:
            assert "行为参考" in result.prompt_text

    def test_period_template_past(self, tmp_config, mock_embed_model):
        l3 = L3EpisodicMemory(tmp_config, mock_embed_model)
        chunks = [make_chunk("c01", period="少女期", volume=2, raw_text="少女时期的艾莉丝")]
        l3.ingest_chunks(chunks)

        result = l3.retrieve("test")
        if result.prompt_text:
            assert "你的记忆" in result.prompt_text

    def test_retrieve_with_filter_tags(self, tmp_config, mock_embed_model):
        l3 = L3EpisodicMemory(tmp_config, mock_embed_model)
        chunks = [
            make_chunk("c01", raw_text="战斗场景"),
            make_chunk("c02", raw_text="日常场景"),
        ]
        l3.ingest_chunks(chunks)

        result = l3.retrieve("test", filter_tags=["战斗"])
        assert isinstance(result.scenes_used, list)
