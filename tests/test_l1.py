"""Tests for L1 Core Identity: module selection."""

import pytest

from src.layers.l1_core import L1CoreIdentity


class TestL1CoreIdentity:
    def test_load_modules(self, tmp_config, sample_modules_yaml):
        l1 = L1CoreIdentity(tmp_config)
        l1.load()
        assert l1.is_loaded
        assert l1.module_count == 3

    def test_core_always_selected(self, tmp_config, sample_modules_yaml):
        l1 = L1CoreIdentity(tmp_config)
        l1.load()

        result = l1.select([])  # No triggers at all
        assert "core" in result.modules_used
        assert "core" in result.active_domains

    def test_trigger_activates_module(self, tmp_config, sample_modules_yaml):
        l1 = L1CoreIdentity(tmp_config)
        l1.load()

        result = l1.select(["战斗", "练剑"])
        assert "core" in result.modules_used
        assert "combat" in result.modules_used
        assert "vulnerability" not in result.modules_used

    def test_vulnerability_trigger(self, tmp_config, sample_modules_yaml):
        l1 = L1CoreIdentity(tmp_config)
        l1.load()

        result = l1.select(["受伤", "恐惧"])
        assert "core" in result.modules_used
        assert "vulnerability" in result.modules_used

    def test_multiple_modules_activated(self, tmp_config, sample_modules_yaml):
        l1 = L1CoreIdentity(tmp_config)
        l1.load()

        result = l1.select(["战斗", "受伤"])
        assert "core" in result.modules_used
        assert "combat" in result.modules_used
        assert "vulnerability" in result.modules_used

    def test_prompt_text_combined(self, tmp_config, sample_modules_yaml):
        l1 = L1CoreIdentity(tmp_config)
        l1.load()

        result = l1.select(["战斗"])
        assert "艾莉丝" in result.prompt_text
        assert "剑" in result.prompt_text

    def test_missing_directory(self, tmp_config):
        l1 = L1CoreIdentity(tmp_config)
        l1.load()  # Should not raise, just warn
        assert not l1.is_loaded
