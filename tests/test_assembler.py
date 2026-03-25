"""Tests for the Assembler: full pipeline assembly and token budget."""

import pytest
from unittest.mock import MagicMock

import numpy as np

from src.layers.assembler import Assembler
from src.layers.l1_core import L1CoreIdentity
from src.layers.l2_behavior import L2BehaviorRules
from src.layers.l3_episodic import L3EpisodicMemory
from src.layers.l4_working import L4WorkingMemory
from src.layers.step_a import StepA
from src.models import Chunk


def make_mock_embed_model():
    model = MagicMock()
    model.is_loaded = True
    model.dimension = 1024
    model.encode_query = lambda t: np.random.randn(1024).astype(np.float32)
    model.encode_documents = lambda texts, **kw: np.random.randn(len(texts), 1024).astype(np.float32)
    return model


class TestAssembler:
    def test_basic_assembly(self, tmp_config, sample_modules_yaml, sample_rules_yaml):
        step_a = StepA(tmp_config)
        step_a.load()

        l1 = L1CoreIdentity(tmp_config)
        l1.load()

        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        embed = make_mock_embed_model()
        l3 = L3EpisodicMemory(tmp_config, embed)

        chunk = Chunk(
            id="test_c01", raw_text="艾莉丝在日常练剑的测试场景",
            volume=18, chapter=1, period="回归后",
        )
        l3.ingest_chunks([chunk])

        l4 = L4WorkingMemory()

        assembler = Assembler(tmp_config, step_a, l1, l2, l3, l4)
        result = assembler.assemble("今天练剑了吗？")

        assert result.system_prompt != ""
        assert "核心身份" in result.system_prompt or "元指令" in result.system_prompt
        assert result.metadata.total_tokens > 0

    def test_assembly_without_l1_l2(self, tmp_config):
        """Assembly should work with just L3 (degraded mode)."""
        step_a = StepA(tmp_config)
        step_a.load()

        l1 = L1CoreIdentity(tmp_config)  # Not loaded
        l2 = L2BehaviorRules(tmp_config)  # Not loaded

        embed = make_mock_embed_model()
        l3 = L3EpisodicMemory(tmp_config, embed)
        l4 = L4WorkingMemory()

        assembler = Assembler(tmp_config, step_a, l1, l2, l3, l4)
        result = assembler.assemble("hello")

        # Should still produce meta-instruction at minimum
        assert "元指令" in result.system_prompt

    def test_token_budget(self, tmp_config, sample_modules_yaml, sample_rules_yaml):
        step_a = StepA(tmp_config)
        step_a.load()

        l1 = L1CoreIdentity(tmp_config)
        l1.load()

        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        embed = make_mock_embed_model()
        l3 = L3EpisodicMemory(tmp_config, embed)
        l4 = L4WorkingMemory()

        assembler = Assembler(tmp_config, step_a, l1, l2, l3, l4)
        result = assembler.assemble("test")

        # Should be under the token budget
        assert result.metadata.total_tokens <= 2500  # Allow some overhead

    def test_l4_state_updates(self, tmp_config, sample_modules_yaml, sample_rules_yaml):
        step_a = StepA(tmp_config)
        step_a.load()

        l1 = L1CoreIdentity(tmp_config)
        l1.load()

        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        embed = make_mock_embed_model()
        l3 = L3EpisodicMemory(tmp_config, embed)
        l4 = L4WorkingMemory()

        assembler = Assembler(tmp_config, step_a, l1, l2, l3, l4)

        # First turn
        assembler.assemble("你好", sender_id="user1")
        state = l4.get_state("user1")
        assert state.conversation_turns == 1

        # Second turn
        assembler.assemble("今天天气不错", sender_id="user1")
        state = l4.get_state("user1")
        assert state.conversation_turns == 2
