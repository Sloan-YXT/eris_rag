"""L2 Behavior Rules: Tag-based rule matching with in-memory inverted index."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.models import L2Result, Rule

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class L2BehaviorRules:
    """Load behavioral rules from YAML and match by situation tags at runtime."""

    def __init__(self, config: Config):
        self._config = config
        self._rules: list[Rule] = []
        self._tag_index: dict[str, list[str]] = {}  # tag → [rule_id, ...]
        self._rules_by_id: dict[str, Rule] = {}
        self._always_rules: list[str] = []  # rule IDs with always tag

    def load(self, rules_path: str | Path | None = None) -> None:
        """Load rules from YAML file and build inverted index."""
        path = Path(rules_path) if rules_path else self._config.character_data_dir / "l2_rules" / "rules.yaml"

        if not path.exists():
            logger.warning(f"L2 rules file not found: {path}")
            return

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_rules = data.get("rules", [])
        self._rules = [Rule(**r) for r in raw_rules]
        self._rules_by_id = {r.id: r for r in self._rules}

        # Build inverted index and identify always-rules
        self._tag_index.clear()
        self._always_rules.clear()

        for rule in self._rules:
            if "always" in rule.situation_tags:
                self._always_rules.append(rule.id)
            for tag in rule.situation_tags:
                if tag != "always":
                    self._tag_index.setdefault(tag, []).append(rule.id)

        logger.info(
            f"Loaded {len(self._rules)} rules, "
            f"{len(self._always_rules)} always-rules, "
            f"{len(self._tag_index)} tag entries"
        )

    @property
    def is_loaded(self) -> bool:
        return len(self._rules) > 0

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def match(
        self,
        triggers: list[str],
        domain_filter: list[str] | None = None,
        max_rules: int | None = None,
    ) -> L2Result:
        """Match rules by trigger tags.

        Args:
            triggers: Situation tags from Step A.
            domain_filter: Only include rules from these L1 domains. None = all domains.
            max_rules: Maximum number of matched rules (excluding always-rules).

        Returns:
            L2Result with formatted prompt text, tags for L3, and rule IDs used.
        """
        max_rules = max_rules or self._config.get_retrieval("l2_top_k", 4)

        # Collect matching rule IDs (excluding always-rules, added separately)
        matched_ids: dict[str, int] = {}  # rule_id → match count
        for tag in triggers:
            for rule_id in self._tag_index.get(tag, []):
                matched_ids[rule_id] = matched_ids.get(rule_id, 0) + 1

        # Sort by match count (more tag overlap = more relevant)
        sorted_ids = sorted(matched_ids.keys(), key=lambda rid: matched_ids[rid], reverse=True)

        # Apply domain filter
        if domain_filter:
            sorted_ids = [
                rid for rid in sorted_ids
                if self._rules_by_id[rid].domain in domain_filter
            ]

        # Limit
        sorted_ids = sorted_ids[:max_rules]

        # Combine: always-rules first, then matched rules
        all_rule_ids = list(self._always_rules) + sorted_ids

        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for rid in all_rule_ids:
            if rid not in seen:
                seen.add(rid)
                unique_ids.append(rid)

        # Build output
        prompt_parts = []
        tags_for_l3: set[str] = set()
        rules_used = []

        for rid in unique_ids:
            rule = self._rules_by_id.get(rid)
            if not rule:
                continue
            rules_used.append(rid)
            prompt_parts.append(self._format_rule(rule))
            # Collect all situation_tags from matched rules for L3 filtering
            for tag in rule.situation_tags:
                if tag != "always":
                    tags_for_l3.add(tag)

        prompt_text = "\n\n".join(prompt_parts) if prompt_parts else ""
        return L2Result(
            prompt_text=prompt_text,
            tags_for_l3=sorted(tags_for_l3),
            rules_used=rules_used,
        )

    @staticmethod
    def _format_rule(rule: Rule) -> str:
        """Format a single rule into natural language for the LLM."""
        parts = []

        # Header with domain
        parts.append(f"【{rule.domain}】{rule.condition}")

        # Behavior
        if rule.behavior:
            parts.append(rule.behavior)

        # Motivation (causal chain)
        if rule.motivation:
            parts.append(f"原因：{rule.motivation}")

        # Language examples as few-shot
        if rule.language_examples:
            examples = " / ".join(f"「{ex}」" for ex in rule.language_examples[:3])
            parts.append(f"语例：{examples}")

        # Exclusions
        if rule.exclusions:
            parts.append(f"禁止：{'、'.join(rule.exclusions[:3])}")

        return "\n".join(parts)
