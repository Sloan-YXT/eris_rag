"""Tests for Step A: intent analysis (local jieba mode + LLM mode parsing)."""

import json
import pytest

from src.layers.step_a import StepA


class TestStepALocal:
    """Tests for local (jieba + substring fallback) mode."""

    def test_basic_trigger_extraction(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        result = step_a.analyze("今天练剑了吗？")
        assert "练剑" in result.triggers
        assert "日常闲聊" in result.triggers

    def test_emotion_detection(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        result = step_a.analyze("真是生气")
        assert "愤怒" in result.triggers
        assert result.emotion_hint == "愤怒"

    def test_substring_fallback(self, tmp_config):
        """Keywords merged by jieba should still match via substring fallback."""
        step_a = StepA(tmp_config)
        step_a.load()

        # jieba merges "别生气" into one token, but "生气" should match via substring
        result = step_a.analyze("你别生气了")
        assert "愤怒" in result.triggers

    def test_topic_is_past(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        result = step_a.analyze("你还记得以前的事吗？")
        assert result.topic_is_past is True
        assert "回忆" in result.triggers

    def test_no_triggers(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        result = step_a.analyze("嗯")
        assert isinstance(result.triggers, list)
        assert result.topic_is_past is False

    def test_multiple_triggers(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        result = step_a.analyze("我受伤了好害怕")
        assert "受伤" in result.triggers
        assert "恐惧" in result.triggers
        assert result.emotion_hint == "恐惧"

    def test_context_past_detection(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        result = step_a.analyze("那后来呢？", conversation_context=["你还记得以前在布埃纳村的事吗？"])
        assert result.topic_is_past is True


class TestStepALLMParsing:
    """Tests for LLM response parsing (no actual API calls)."""

    def test_parse_valid_json(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        response = json.dumps({
            "triggers": ["战斗", "愤怒"],
            "topic_is_past": False,
            "emotion_hint": "愤怒",
        }, ensure_ascii=False)

        result = step_a._parse_llm_response(response)
        assert "战斗" in result.triggers
        assert "愤怒" in result.triggers
        assert result.emotion_hint == "愤怒"

    def test_parse_with_markdown_fence(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        response = '```json\n{"triggers": ["练剑"], "topic_is_past": false, "emotion_hint": ""}\n```'
        result = step_a._parse_llm_response(response)
        assert "练剑" in result.triggers

    def test_parse_filters_invalid_tags(self, tmp_config):
        step_a = StepA(tmp_config)
        step_a.load()

        response = json.dumps({
            "triggers": ["战斗", "不存在的标签", "愤怒"],
            "topic_is_past": False,
            "emotion_hint": "不存在的情绪",
        }, ensure_ascii=False)

        result = step_a._parse_llm_response(response)
        assert "不存在的标签" not in result.triggers
        assert "战斗" in result.triggers
        assert result.emotion_hint == ""  # Invalid emotion filtered out
