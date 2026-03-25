"""L1 Core Identity: Dynamic module selection based on trigger tags."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.models import L1Result, Module

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class L1CoreIdentity:
    """Load identity modules from YAML and select dynamically at runtime."""

    def __init__(self, config: Config):
        self._config = config
        self._modules: list[Module] = []
        self._modules_by_id: dict[str, Module] = {}

    def load(self, modules_dir: str | Path | None = None) -> None:
        """Load all L1 module YAML files from directory."""
        directory = Path(modules_dir) if modules_dir else self._config.character_data_dir / "l1_modules"

        if not directory.exists():
            logger.warning(f"L1 modules directory not found: {directory}")
            return

        self._modules.clear()
        self._modules_by_id.clear()

        for yaml_path in sorted(directory.glob("*.yaml")):
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                module = Module(**data)
                self._modules.append(module)
                self._modules_by_id[module.id] = module

        logger.info(f"Loaded {len(self._modules)} L1 identity modules: {[m.id for m in self._modules]}")

    @property
    def is_loaded(self) -> bool:
        return len(self._modules) > 0

    @property
    def module_count(self) -> int:
        return len(self._modules)

    def select(self, triggers: list[str]) -> L1Result:
        """Select active modules based on trigger tags.

        - 'core' module (activate_on: ["always"]) is always included.
        - Other modules are included if any of their activate_on tags
          overlap with the provided triggers.

        Args:
            triggers: Situation tags from Step A analysis.

        Returns:
            L1Result with combined prompt text, active domains, and module IDs.
        """
        trigger_set = set(triggers)
        selected: list[Module] = []
        active_domains: list[str] = []
        modules_used: list[str] = []

        for module in self._modules:
            if "always" in module.activate_on:
                # Always include (core module)
                selected.append(module)
                active_domains.append(module.domain)
                modules_used.append(module.id)
            elif trigger_set & set(module.activate_on):
                # Trigger overlap
                selected.append(module)
                active_domains.append(module.domain)
                modules_used.append(module.id)

        prompt_text = "\n\n".join(m.prompt_text for m in selected if m.prompt_text)

        return L1Result(
            prompt_text=prompt_text,
            active_domains=active_domains,
            modules_used=modules_used,
        )
