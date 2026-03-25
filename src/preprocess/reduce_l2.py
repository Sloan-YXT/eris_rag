"""L2 行为规则生成：从 RAG 检索结果中归纳行为规则。

流程：对每个 domain → 用 domain 相关 tags 检索 L3 → LLM 归纳规则。
替代旧方案的两轮 REDUCE（逐卷 → 全局）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.llm.client import LLMClient, load_prompt_template
from src.models import Rule

if TYPE_CHECKING:
    from src.config import Config
    from src.layers.l3_episodic import L3EpisodicMemory

logger = logging.getLogger(__name__)

RULES_PROMPT_PATH = Path(__file__).parent / "prompts" / "reduce_global_rules.md"


async def reduce_rules_from_rag(
    l3: L3EpisodicMemory,
    config: Config,
    taxonomy_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[Rule]:
    """对每个 domain，检索相关原文 → LLM 归纳行为规则。

    Returns:
        全部 Rule 对象。
    """
    taxonomy_path = Path(taxonomy_path) if taxonomy_path else config.taxonomy_path
    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    domains: dict[str, dict] = taxonomy.get("domains", {})
    template = load_prompt_template(RULES_PROMPT_PATH)
    provider = config.get_preprocess_provider("reduce_global_rules")
    client = LLMClient(config)
    all_rules: list[Rule] = []

    try:
        for domain_name, domain_info in domains.items():
            activate_tags = domain_info.get("activate_on", [])
            if "always" in activate_tags:
                activate_tags = []  # core domain 用通用查询

            logger.info(f"生成 domain={domain_name} 的行为规则 (tags={activate_tags})")

            # 检索相关 chunks
            chunks_text = await _retrieve_domain_chunks(
                l3, domain_name, activate_tags, config,
            )

            if not chunks_text:
                logger.warning(f"  domain={domain_name}: 没有检索到相关 chunks，跳过")
                continue

            # 构建 prompt
            prompt = template.replace("{domain}", domain_name)
            prompt = prompt.replace("{domain_description}", domain_info.get("description", ""))
            prompt = prompt.replace("{chunks_text}", chunks_text)

            try:
                response = await client.complete(
                    provider=provider,
                    system_prompt="你是一个角色行为分析专家。请从小说原文片段中归纳行为规则，以成长回归后的艾莉丝为基准。",
                    user_prompt=prompt,
                    temperature=0.3,
                    max_tokens=8192,
                )
                raw_rules = _parse_json_response(response)
                for r in raw_rules:
                    r["domain"] = domain_name
                    all_rules.append(Rule(**r))
                logger.info(f"  domain={domain_name}: 生成 {len(raw_rules)} 条规则")
            except Exception as e:
                logger.error(f"  domain={domain_name} 规则生成失败: {e}")

            await asyncio.sleep(1.0)

        # 单独生成 speech_style 规则
        logger.info("生成 speech_style 规则")
        speech_chunks = await _retrieve_speech_chunks(l3)
        if speech_chunks:
            speech_rule = await _generate_speech_rule(client, provider, template, speech_chunks)
            if speech_rule:
                all_rules.append(speech_rule)

    finally:
        await client.close()

    # 保存
    output_path = Path(output_path) if output_path else config.character_data_dir / "l2_rules" / "rules.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rules_data = [r.model_dump() for r in all_rules]
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump({"rules": rules_data}, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # tag 反向索引
    tag_index: dict[str, list[str]] = {}
    for rule in all_rules:
        for tag in rule.situation_tags:
            tag_index.setdefault(tag, []).append(rule.id)
    index_path = output_path.parent / "tag_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(tag_index, f, ensure_ascii=False, indent=2)

    logger.info(f"保存 {len(all_rules)} 条规则到 {output_path}")
    return all_rules


async def _retrieve_domain_chunks(
    l3: L3EpisodicMemory,
    domain: str,
    tags: list[str],
    config: Config,
    max_chunks: int = 50,
) -> str:
    """为一个 domain 检索相关的原文片段，分散采样不同卷。"""
    queries = []
    if tags:
        for tag in tags[:5]:
            queries.append(f"艾莉丝 {tag}")
    else:
        # core domain：通用查询
        queries = ["艾莉丝的性格", "艾莉丝的行为", "艾莉丝说话"]

    all_results: dict[str, dict] = {}
    for query in queries:
        results = l3.retrieve_raw(query=query, top_k=20)
        for r in results:
            if r["scene_id"] not in all_results:
                all_results[r["scene_id"]] = r

    if not all_results:
        return ""

    # 按卷分散采样：每卷最多取 max_chunks / 卷数 个
    by_volume: dict[int, list[dict]] = {}
    for r in all_results.values():
        by_volume.setdefault(r["volume"], []).append(r)

    sampled = []
    per_vol = max(2, max_chunks // max(len(by_volume), 1))
    for vol in sorted(by_volume.keys()):
        vol_items = sorted(by_volume[vol], key=lambda x: x["score"], reverse=True)
        sampled.extend(vol_items[:per_vol])

    sampled = sampled[:max_chunks]

    # 拼接为文本
    parts = []
    for r in sampled:
        parts.append(f"[第{r['volume']}卷]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


async def _retrieve_speech_chunks(l3: L3EpisodicMemory) -> str:
    """检索包含艾莉丝对话的片段。"""
    queries = ["艾莉丝说", "「", "艾莉丝的语气"]
    all_results: dict[str, dict] = {}
    for query in queries:
        results = l3.retrieve_raw(query=query, top_k=20)
        for r in results:
            if r["scene_id"] not in all_results:
                all_results[r["scene_id"]] = r

    # 只保留确实包含对话标记的
    filtered = [r for r in all_results.values() if "「" in r["text"]]
    filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)[:30]

    parts = []
    for r in filtered:
        parts.append(f"[第{r['volume']}卷]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


async def _generate_speech_rule(
    client: LLMClient,
    provider: str,
    template: str,
    chunks_text: str,
) -> Rule | None:
    """生成 speech_style 规则。"""
    prompt = template.replace("{domain}", "speech_style")
    prompt = prompt.replace("{domain_description}", "说话方式、用词习惯、句式特征、语气变化")
    prompt = prompt.replace("{chunks_text}", chunks_text)

    try:
        response = await client.complete(
            provider=provider,
            system_prompt="你是一个角色语言分析专家。请从对话片段中归纳说话风格规则。",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=4096,
        )
        raw_rules = _parse_json_response(response)
        if raw_rules:
            r = raw_rules[0]
            r["id"] = "speech_style"
            r["domain"] = "core"
            r["situation_tags"] = ["always"]
            return Rule(**r)
    except Exception as e:
        logger.error(f"speech_style 规则生成失败: {e}")
    return None


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
