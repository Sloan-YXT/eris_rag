"""REDUCE phase: Behavioral rules → Core identity modules."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from src.config import Config
from src.llm.client import LLMClient, load_prompt_template
from src.models import Module, Rule

logger = logging.getLogger(__name__)

CORE_MODULES_TEMPLATE = Path(__file__).parent / "prompts" / "reduce_core_modules.md"


async def reduce_core_modules(
    rules: list[Rule],
    config: Config,
    output_dir: str | Path | None = None,
) -> list[Module]:
    """REDUCE round 3: Synthesize rules into core identity modules.

    Returns:
        List of Module objects (one per domain).
    """
    provider = config.get_preprocess_provider("reduce_core_modules")
    template = load_prompt_template(CORE_MODULES_TEMPLATE)

    rules_json = json.dumps(
        [r.model_dump() for r in rules],
        ensure_ascii=False, indent=2,
    )
    prompt = template.replace("{all_rules_json}", rules_json)

    client = LLMClient(config)
    try:
        response = await client.complete(
            provider=provider,
            system_prompt="你是一个角色心理分析专家。请将行为规则浓缩为核心身份模块，使用因果链而非标签。",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=8192,
        )
    finally:
        await client.close()

    raw_modules = _parse_json_response(response)
    modules = [Module(**m) for m in raw_modules]

    # Save each module as separate YAML
    output_dir = Path(output_dir) if output_dir else config.character_data_dir / "l1_modules"
    output_dir.mkdir(parents=True, exist_ok=True)

    for module in modules:
        path = output_dir / f"{module.id}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(module.model_dump(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved {len(modules)} identity modules to {output_dir}")
    return modules


def _parse_json_response(response: str) -> list[dict]:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    data = json.loads(text)
    return data if isinstance(data, list) else [data]
