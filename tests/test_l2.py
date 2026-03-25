"""Tests for L2 Behavior Rules: tag matching and rule selection."""

import pytest

from src.layers.l2_behavior import L2BehaviorRules


class TestL2BehaviorRules:
    def test_load_rules(self, tmp_config, sample_rules_yaml):
        l2 = L2BehaviorRules(tmp_config)
        l2.load()
        assert l2.is_loaded
        assert l2.rule_count == 3

    def test_always_rules_included(self, tmp_config, sample_rules_yaml):
        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        result = l2.match([])  # No triggers
        assert "speech_style" in result.rules_used

    def test_combat_trigger_match(self, tmp_config, sample_rules_yaml):
        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        result = l2.match(["战斗", "练剑"])
        assert "speech_style" in result.rules_used
        assert "beh_combat_01" in result.rules_used

    def test_vulnerability_trigger(self, tmp_config, sample_rules_yaml):
        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        result = l2.match(["受伤", "恐惧"])
        assert "beh_vuln_01" in result.rules_used

    def test_tags_for_l3_propagation(self, tmp_config, sample_rules_yaml):
        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        result = l2.match(["战斗"])
        # Should include tags from matched rules
        assert "战斗" in result.tags_for_l3 or "练剑" in result.tags_for_l3

    def test_domain_filter(self, tmp_config, sample_rules_yaml):
        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        result = l2.match(["战斗", "受伤"], domain_filter=["combat"])
        matched_non_always = [r for r in result.rules_used if r != "speech_style"]
        for rid in matched_non_always:
            # Only combat domain rules should be matched
            assert "combat" in rid

    def test_prompt_text_format(self, tmp_config, sample_rules_yaml):
        l2 = L2BehaviorRules(tmp_config)
        l2.load()

        result = l2.match(["战斗"])
        assert "来吧" in result.prompt_text or "战斗" in result.prompt_text

    def test_missing_file(self, tmp_config):
        l2 = L2BehaviorRules(tmp_config)
        l2.load()  # Should not raise
        assert not l2.is_loaded
