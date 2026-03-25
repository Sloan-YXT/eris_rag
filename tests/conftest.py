"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


@pytest.fixture
def tmp_config(tmp_path):
    """Create a minimal config for testing (no GPU, no real models)."""
    config_data = {
        "server": {"host": "127.0.0.1", "port": 18787},
        "api": {
            "providers": {
                "gemini": {"api_key": "test-key", "model": "test-model", "base_url": "http://localhost"},
            },
            "preprocess": {"map_extract": "gemini"},
            "runtime": {"step_a": "local"},
        },
        "embedding": {"model_name": "BAAI/bge-large-zh-v1.5", "device": "cpu", "batch_size": 8},
        "reranker": {"model_name": "BAAI/bge-reranker-v2-m3", "device": "cpu"},
        "vectordb": {"persist_dir": str(tmp_path / "vectordb"), "collection_name": "test_scenes"},
        "character": {
            "name": "eris",
            "data_dir": str(tmp_path / "data" / "characters" / "eris"),
            "novel_dir": str(tmp_path / "data" / "novel"),
            "target_period": "回归后",
        },
        "retrieval": {"l3_candidates": 4, "l3_top_k": 2, "l2_top_k": 3, "target_total_tokens": 2000},
        "preprocess": {"map_batch_size": 2, "map_delay_seconds": 0},
        "taxonomy_path": str(tmp_path / "tags_taxonomy.yaml"),
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # Create a minimal taxonomy
    taxonomy = {
        "domains": {
            "core": {"description": "test", "activate_on": ["always"]},
            "vulnerability": {"description": "test", "activate_on": ["受伤", "恐惧", "哭泣"]},
            "combat": {"description": "test", "activate_on": ["战斗", "练剑"]},
            "relationship": {"description": "test", "activate_on": ["心动", "嫉妒"]},
            "pride": {"description": "test", "activate_on": ["被夸奖", "被侮辱"]},
            "growth": {"description": "test", "activate_on": ["回忆"]},
        },
        "situations": ["愤怒", "恐惧", "喜悦", "受伤", "战斗", "练剑", "心动", "被夸奖", "被侮辱", "回忆", "日常闲聊"],
        "keyword_dict": {
            "练剑": "练剑",
            "剑": "练剑",
            "受伤": "受伤",
            "伤": "受伤",
            "生气": "愤怒",
            "害怕": "恐惧",
            "开心": "喜悦",
            "喜欢": "心动",
            "以前": "回忆",
            "厉害": "被夸奖",
            "笨": "被侮辱",
            "今天": "日常闲聊",
        },
        "volume_periods": {
            "少女期": [1, 2, 3],
            "魔大陆流浪期": [4, 5, 6],
            "剑之圣地期": [7, 8, 9],
            "回归后": [16, 17, 18],
        },
        "period_weights": {
            "default": {"少女期": 0.85, "魔大陆流浪期": 1.0, "剑之圣地期": 1.1, "回归后": 1.3},
            "topic_is_past": {"少女期": 1.3, "魔大陆流浪期": 1.2, "剑之圣地期": 1.0, "回归后": 0.85},
        },
    }
    with open(tmp_path / "tags_taxonomy.yaml", "w", encoding="utf-8") as f:
        yaml.dump(taxonomy, f, allow_unicode=True)

    # Import Config after writing the file
    from src.config import Config
    return Config(str(config_path))


@pytest.fixture
def sample_rules_yaml(tmp_path, tmp_config):
    """Create sample L2 rules YAML."""
    rules_dir = Path(tmp_config.character_data_dir) / "l2_rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    rules_data = {
        "rules": [
            {
                "id": "speech_style",
                "domain": "core",
                "situation_tags": ["always"],
                "condition": "任何对话时",
                "behavior": "说话直接、不拐弯抹角",
                "motivation": "性格使然",
                "language_examples": ["哼，真是的", "你在说什么啊"],
                "exclusions": ["使用敬语"],
            },
            {
                "id": "beh_combat_01",
                "domain": "combat",
                "situation_tags": ["战斗", "练剑", "遇强敌"],
                "condition": "当面对战斗或练剑时",
                "behavior": "眼睛发亮，立刻进入战斗状态",
                "motivation": "对剑术的热爱",
                "language_examples": ["来吧！", "让我试试你的实力"],
            },
            {
                "id": "beh_vuln_01",
                "domain": "vulnerability",
                "situation_tags": ["受伤", "恐惧", "哭泣"],
                "condition": "当受伤或感到恐惧时",
                "behavior": "试图隐藏脆弱",
                "motivation": "不想被人看到软弱的一面",
            },
        ]
    }

    rules_path = rules_dir / "rules.yaml"
    with open(rules_path, "w", encoding="utf-8") as f:
        yaml.dump(rules_data, f, allow_unicode=True)

    return rules_path


@pytest.fixture
def sample_modules_yaml(tmp_path, tmp_config):
    """Create sample L1 module YAML files."""
    modules_dir = Path(tmp_config.character_data_dir) / "l1_modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    modules = [
        {
            "id": "core",
            "domain": "core",
            "activate_on": ["always"],
            "prompt_text": "你是艾莉丝·格雷拉特。你直率、好胜、内心善良。",
        },
        {
            "id": "vulnerability",
            "domain": "vulnerability",
            "activate_on": ["受伤", "恐惧", "哭泣", "孤独"],
            "prompt_text": "在独处或受伤时，你会流露脆弱。",
        },
        {
            "id": "combat",
            "domain": "combat",
            "activate_on": ["战斗", "练剑", "遇强敌"],
            "prompt_text": "你热爱剑术，是剑王级别的剑士。",
        },
    ]

    for m in modules:
        with open(modules_dir / f"{m['id']}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(m, f, allow_unicode=True)

    return modules_dir
