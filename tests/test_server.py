"""Tests for the FastAPI server endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.models import AssemblyMetadata, AssemblyResult, L3Result


@pytest.fixture
def client():
    """Create a test client with fully mocked components (no lifespan)."""
    import src.server as srv

    # Save originals
    orig_lifespan = srv.app.router.lifespan_context

    # Replace lifespan with no-op
    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    srv.app.router.lifespan_context = noop_lifespan

    # Set up mocked globals
    srv._start_time = 1000.0
    srv._config = MagicMock()

    # Mock assembler with async support
    mock_metadata = AssemblyMetadata(
        l1_modules_used=["core"],
        l2_rules_used=["speech_style"],
        l3_scenes_used=["s01"],
        total_tokens=500,
    )
    mock_result = AssemblyResult(
        system_prompt="Test system prompt",
        metadata=mock_metadata,
    )
    mock_assembler = MagicMock()
    mock_assembler.assemble_async = AsyncMock(return_value=mock_result)
    srv._assembler = mock_assembler

    # Mock L3
    mock_l3 = MagicMock()
    mock_l3.scene_count = 100
    mock_l3.retrieve_raw.return_value = [
        {"scene_id": "s01", "volume": 1, "chapter": 1, "summary": "Test", "text": "Test text", "score": 0.9}
    ]
    srv._l3 = mock_l3

    # Mock status flags
    mock_embed = MagicMock()
    mock_embed.is_loaded = True
    srv._embed_model = mock_embed

    mock_reranker = MagicMock()
    mock_reranker.is_loaded = True
    srv._reranker = mock_reranker

    mock_l1 = MagicMock()
    mock_l1.is_loaded = True
    srv._l1 = mock_l1

    mock_l2 = MagicMock()
    mock_l2.is_loaded = True
    srv._l2 = mock_l2

    with TestClient(srv.app) as c:
        yield c

    # Restore
    srv.app.router.lifespan_context = orig_lifespan


class TestServerEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["embedding_loaded"] is True
        assert data["scene_count"] == 100

    def test_retrieve(self, client):
        resp = client.post("/retrieve", json={
            "user_message": "今天练剑了吗",
            "sender_id": "test_user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "enhanced_system_prompt" in data
        assert data["enhanced_system_prompt"] == "Test system prompt"
        assert data["metadata"]["total_tokens"] == 500

    def test_query_raw(self, client):
        resp = client.post("/query", json={
            "query": "艾莉丝的剑术",
            "top_k": 3,
            "format": "raw",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "raw_text" in data
        assert len(data["results"]) > 0

    def test_query_structured(self, client):
        resp = client.post("/query", json={
            "query": "艾莉丝",
            "top_k": 2,
            "format": "structured",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) > 0
        assert "scene_id" in data["results"][0]
