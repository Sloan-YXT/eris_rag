"""Phase 2: RAG + LLM 元数据标注。

向量库建好后，用检索 + LLM 为每个 chunk 添加 situation_tags / has_eris / emotion。
替代旧的 map_extract.py（那个是先让 LLM 从原文提取场景，逻辑是反的）。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.llm.client import LLMClient, load_prompt_template

if TYPE_CHECKING:
    from src.config import Config
    from src.layers.l3_episodic import L3EpisodicMemory

logger = logging.getLogger(__name__)

ANNOTATE_PROMPT_PATH = Path(__file__).parent / "prompts" / "annotate_tags.md"


async def annotate_tags(
    l3: L3EpisodicMemory,
    config: Config,
    taxonomy_path: str | Path | None = None,
) -> dict[str, int]:
    """对向量库中的 chunk 执行 tag 标注。

    流程（对每个 situation_tag）：
    1. 用 tag_queries 检索 L3 → top-30 chunks
    2. 批量喂给 LLM 判断哪些 chunk 确实与该 tag 相关
    3. 更新 ChromaDB metadata

    Returns:
        标注统计：{tag: 被标注的chunk数}
    """
    taxonomy_path = Path(taxonomy_path) if taxonomy_path else config.taxonomy_path
    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    tag_queries: dict[str, list[str]] = taxonomy.get("tag_queries", {})
    if not tag_queries:
        logger.warning("taxonomy 中没有 tag_queries，跳过标注")
        return {}

    template = load_prompt_template(ANNOTATE_PROMPT_PATH)
    provider = config.get_preprocess_provider("annotate_tags")
    client = LLMClient(config)
    stats: dict[str, int] = {}

    try:
        for tag, queries in tag_queries.items():
            logger.info(f"标注 tag: {tag} (queries: {queries})")

            # 用每个 query 检索，合并去重
            candidate_ids: dict[str, tuple[str, dict]] = {}  # id → (document, metadata)
            for query in queries:
                results = l3.retrieve_raw(query=query, top_k=30)
                for r in results:
                    if r["scene_id"] not in candidate_ids:
                        candidate_ids[r["scene_id"]] = (r["text"], {
                            "volume": r["volume"],
                            "chapter": r["chapter"],
                        })

            if not candidate_ids:
                stats[tag] = 0
                continue

            # 构建 prompt：让 LLM 判断哪些 chunk 与该 tag 相关
            chunks_text = ""
            id_list = list(candidate_ids.keys())
            for idx, cid in enumerate(id_list):
                doc_text, meta = candidate_ids[cid]
                # 截取前 300 字符以控制 prompt 大小
                preview = doc_text[:300]
                chunks_text += f"\n[{idx}] (id={cid}, 第{meta['volume']}卷)\n{preview}\n"

            prompt = template.replace("{tag}", tag)
            prompt = prompt.replace("{chunks_text}", chunks_text)
            prompt = prompt.replace("{chunk_count}", str(len(id_list)))

            try:
                response = await client.complete(
                    provider=provider,
                    system_prompt="你是一个文本分析助手。请严格按 JSON 格式输出。",
                    user_prompt=prompt,
                    temperature=0.1,
                    max_tokens=8192,
                )
                confirmed_indices = _parse_annotation_response(response)
                confirmed_ids = [id_list[i] for i in confirmed_indices if i < len(id_list)]
            except Exception as e:
                logger.error(f"标注 tag={tag} 失败: {e}")
                confirmed_ids = []

            # 更新 metadata：在现有 situation_tags 中追加该 tag
            for cid in confirmed_ids:
                existing = l3._collection.get(ids=[cid], include=["metadatas"])
                if existing["ids"]:
                    meta = existing["metadatas"][0]
                    current_tags = meta.get("situation_tags", "")
                    tag_set = set(current_tags.split(",")) if current_tags else set()
                    tag_set.discard("")
                    tag_set.add(tag)
                    meta["situation_tags"] = ",".join(sorted(tag_set))
                    l3._collection.update(ids=[cid], metadatas=[meta])

            stats[tag] = len(confirmed_ids)
            logger.info(f"  tag={tag}: {len(confirmed_ids)}/{len(id_list)} chunks 确认")
            await asyncio.sleep(1.0)

    finally:
        await client.close()

    return stats


async def annotate_significance(
    l3: L3EpisodicMemory,
    config: Config,
    taxonomy_path: str | Path | None = None,
) -> int:
    """标注核心记忆：给关键成长节点的 chunk 打 significance 分数。

    用 significance_queries 检索 → LLM 判断哪些是人生转折点 → 打分 0-1。

    Returns:
        被标注的 chunk 数。
    """
    taxonomy_path = Path(taxonomy_path) if taxonomy_path else config.taxonomy_path
    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    sig_queries = taxonomy.get("significance_queries", [])
    if not sig_queries:
        logger.warning("taxonomy 中没有 significance_queries，跳过")
        return 0

    provider = config.get_preprocess_provider("annotate_tags")
    client = LLMClient(config)

    # 检索候选
    candidate_ids: dict[str, tuple[str, dict]] = {}
    for query in sig_queries:
        results = l3.retrieve_raw(query=query, top_k=20)
        for r in results:
            if r["scene_id"] not in candidate_ids:
                candidate_ids[r["scene_id"]] = (r["text"], {"volume": r["volume"]})

    if not candidate_ids:
        await client.close()
        return 0

    id_list = list(candidate_ids.keys())
    chunks_text = ""
    for idx, cid in enumerate(id_list):
        doc_text, meta = candidate_ids[cid]
        preview = doc_text[:300]
        chunks_text += f"\n[{idx}] (id={cid}, 第{meta['volume']}卷)\n{preview}\n"

    prompt = f"""以下是从《无职转生》中检索到的 {len(id_list)} 个文本片段。
请判断哪些片段描述了艾莉丝人生中的**关键转折点**（如重大创伤、重要决定、性格转变的契机等）。

对每个确认的片段，给出 0.5-1.0 的分数：
- 1.0 = 绝对的人生转折（如失去家人、与重要的人分别/重逢）
- 0.7 = 重要时刻（如战斗中的觉悟、关键选择）
- 0.5 = 有一定意义的成长瞬间

{chunks_text}

输出 JSON 对象，key 是片段编号，value 是分数。不相关的片段不要包含。
示例：{{"0": 1.0, "3": 0.7, "5": 0.5}}
输出纯 JSON。"""

    try:
        response = await client.complete(
            provider=provider,
            system_prompt="你是一个叙事分析专家。请判断哪些片段是角色的关键成长节点。",
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=8192,
        )
        scores = _parse_significance_response(response)
    except Exception as e:
        logger.error(f"significance 标注失败: {e}")
        scores = {}
    finally:
        await client.close()

    # 更新 metadata
    count = 0
    for idx_str, score in scores.items():
        idx = int(idx_str)
        if idx < len(id_list):
            cid = id_list[idx]
            existing = l3._collection.get(ids=[cid], include=["metadatas"])
            if existing["ids"]:
                meta = existing["metadatas"][0]
                meta["significance"] = min(1.0, max(0.0, float(score)))
                l3._collection.update(ids=[cid], metadatas=[meta])
                count += 1

    logger.info(f"significance 标注: {count} chunks")
    return count


def _parse_significance_response(response: str) -> dict:
    """解析 significance 评分的 JSON 响应。"""
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return json.loads(text)


def _parse_annotation_response(response: str) -> list[int]:
    """解析 LLM 返回的 JSON，提取确认的 chunk 索引列表。"""
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    data = json.loads(text)

    if isinstance(data, list):
        return [int(x) for x in data if isinstance(x, (int, float))]
    if isinstance(data, dict):
        return [int(x) for x in data.get("confirmed", data.get("indices", []))]
    return []
