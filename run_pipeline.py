"""预处理流水线 CLI — 分步执行，随时可停。

用法:
  python run_pipeline.py chunk          # Phase 1: 分块 + embed + 入库
  python run_pipeline.py check          # 检查向量库状态 + 试几个检索
  python run_pipeline.py search 查询内容  # 搜索向量库
  python run_pipeline.py annotate       # Phase 2: tag 标注 + significance 标注
  python run_pipeline.py reduce_l2      # Phase 3: 生成行为规则
  python run_pipeline.py reduce_l1      # Phase 4: 生成核心模块
  python run_pipeline.py all            # 全部按顺序
  python run_pipeline.py rebuild_memory  # 重建所有用户的潜意识向量库
"""

import asyncio
import json
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pipeline")


def get_config():
    from src.config import Config
    return Config()


def phase_chunk(config):
    """Phase 1: 原文分块 → embed → 入 ChromaDB。"""
    from src.preprocess.chunker import parse_novel_to_chunks
    from src.embedding.embed_model import EmbeddingModel
    from src.layers.l3_episodic import L3EpisodicMemory

    novel_files = sorted(config.novel_dir.glob("*.txt"))
    if not novel_files:
        logger.error(f"没有找到小说文件: {config.novel_dir}")
        logger.error("请把小说 TXT 放到该目录下")
        return

    logger.info(f"找到 {len(novel_files)} 个文件: {[f.name for f in novel_files]}")

    # 分块
    all_chunks = []
    for f in novel_files:
        logger.info(f"解析 {f.name}...")
        chunks = parse_novel_to_chunks(
            str(f), str(config.taxonomy_path),
            target_size=config.get("chunking.target_size", 384),
            max_size=config.get("chunking.max_size", 480),
            overlap=config.get("chunking.overlap", 64),
        )
        all_chunks.extend(chunks)

    logger.info(f"分块完成: {len(all_chunks)} 个 chunks")

    # 统计
    sizes = [len(c.raw_text) for c in all_chunks]
    logger.info(f"  大小分布: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    volumes = set(c.volume for c in all_chunks)
    logger.info(f"  覆盖卷: {sorted(volumes)}")
    periods = {}
    for c in all_chunks:
        periods[c.period] = periods.get(c.period, 0) + 1
    logger.info(f"  各时期 chunks: {periods}")

    # 加载 embedding 模型
    embed_model = EmbeddingModel(config)
    embed_model.load()

    # 入库
    l3 = L3EpisodicMemory(config, embed_model)
    t0 = time.time()
    count = l3.ingest_chunks(all_chunks)
    elapsed = time.time() - t0
    logger.info(f"入库完成: {count} 新增, 总计 {l3.scene_count}, 耗时 {elapsed:.1f}s")

    # has_eris 统计
    eris_count = 0
    batch_size = 100
    all_ids = [c.id for c in all_chunks]
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        result = l3._collection.get(ids=batch_ids, include=["metadatas"])
        for meta in result["metadatas"]:
            if meta.get("has_eris"):
                eris_count += 1
    logger.info(f"涉及艾莉丝的 chunks: {eris_count}/{len(all_chunks)}")


def phase_check(config):
    """检查向量库状态 + 试几个检索。"""
    from src.embedding.embed_model import EmbeddingModel
    from src.layers.l3_episodic import L3EpisodicMemory

    embed_model = EmbeddingModel(config)
    embed_model.load()
    l3 = L3EpisodicMemory(config, embed_model)

    logger.info(f"向量库 chunks 数量: {l3.scene_count}")
    if l3.scene_count == 0:
        logger.error("向量库为空，请先运行 chunk")
        return

    test_queries = [
        "艾莉丝练剑",
        "艾莉丝和鲁迪乌斯",
        "艾莉丝害怕",
        "艾莉丝的贵族身份",
        "战斗场景",
    ]

    for q in test_queries:
        results = l3.retrieve_raw(query=q, top_k=2)
        logger.info(f"\n{'='*60}")
        logger.info(f"查询: {q}")
        for r in results:
            logger.info(f"  [{r['scene_id']}] 第{r['volume']}卷 (score={r['score']:.3f})")
            preview = r['text'][:150].replace('\n', ' ')
            logger.info(f"  {preview}...")


def phase_search(config, query: str, top_k: int = 5):
    """搜索向量库（语义 + 关键词混合，reranker 精排）。"""
    from src.embedding.embed_model import EmbeddingModel
    from src.embedding.reranker import Reranker
    from src.layers.l3_episodic import L3EpisodicMemory

    embed_model = EmbeddingModel(config)
    embed_model.load()

    reranker = Reranker(config)
    reranker.load()

    l3 = L3EpisodicMemory(config, embed_model, reranker=reranker)

    if l3.scene_count == 0:
        print("向量库为空，请先运行 chunk")
        return

    results = l3.retrieve_raw(query=query, top_k=top_k)
    print(f"\n搜索: {query}  (共 {l3.scene_count} chunks)\n")
    for i, r in enumerate(results):
        mt = r.get('match_type', '?')
        print(f"--- [{i+1}] {r['scene_id']} | 第{r['volume']}卷 | score={r['score']:.3f} ({mt}) ---")
        print(r['text'][:500])
        print()


async def phase_annotate(config):
    """Phase 2: tag 标注 + significance 标注。"""
    from src.embedding.embed_model import EmbeddingModel
    from src.layers.l3_episodic import L3EpisodicMemory
    from src.preprocess.annotator import annotate_significance, annotate_tags

    embed_model = EmbeddingModel(config)
    embed_model.load()
    l3 = L3EpisodicMemory(config, embed_model)

    if l3.scene_count == 0:
        logger.error("向量库为空，请先运行 chunk")
        return

    logger.info("Phase 2a: tag 标注...")
    stats = await annotate_tags(l3, config)
    logger.info(f"tag 标注结果: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    logger.info("Phase 2b: significance 标注...")
    sig_count = await annotate_significance(l3, config)
    logger.info(f"significance 标注: {sig_count} chunks")


async def phase_reduce_l2(config):
    """Phase 3: 从 RAG 检索结果生成行为规则。"""
    from src.embedding.embed_model import EmbeddingModel
    from src.layers.l3_episodic import L3EpisodicMemory
    from src.preprocess.reduce_l2 import reduce_rules_from_rag

    embed_model = EmbeddingModel(config)
    embed_model.load()
    l3 = L3EpisodicMemory(config, embed_model)

    if l3.scene_count == 0:
        logger.error("向量库为空，请先运行 chunk")
        return

    rules = await reduce_rules_from_rag(l3, config)
    logger.info(f"生成 {len(rules)} 条规则")
    for r in rules:
        logger.info(f"  [{r.id}] {r.domain}: {r.condition}")

    output = config.character_data_dir / "l2_rules" / "rules.yaml"
    logger.info(f"保存到 {output}")


async def phase_reduce_l1(config):
    """Phase 4: 从 L2 规则生成核心模块。"""
    import yaml
    from src.models import Rule
    from src.preprocess.reduce_l1 import reduce_core_modules

    rules_path = config.character_data_dir / "l2_rules" / "rules.yaml"
    if not rules_path.exists():
        logger.error(f"L2 规则文件不存在: {rules_path}")
        logger.error("请先运行 reduce_l2")
        return

    with open(rules_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rules = [Rule(**r) for r in data.get("rules", [])]

    modules = await reduce_core_modules(rules, config)
    logger.info(f"生成 {len(modules)} 个模块")
    for m in modules:
        logger.info(f"  [{m.id}] {m.domain}: {m.prompt_text[:80]}...")

    output_dir = config.character_data_dir / "l1_modules"
    logger.info(f"保存到 {output_dir}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    stage = sys.argv[1]
    config = get_config()

    if stage == "chunk":
        phase_chunk(config)
    elif stage == "check":
        phase_check(config)
    elif stage == "annotate":
        asyncio.run(phase_annotate(config))
    elif stage == "reduce_l2":
        asyncio.run(phase_reduce_l2(config))
    elif stage == "reduce_l1":
        asyncio.run(phase_reduce_l1(config))
    elif stage == "search":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        if not query:
            print("用法: python run_pipeline.py search 查询内容")
            sys.exit(1)
        phase_search(config, query)
    elif stage == "rebuild_memory":
        from src.embedding.embed_model import EmbeddingModel
        from src.layers.subconscious import SubconsciousMemory
        embed_model = EmbeddingModel(config)
        embed_model.load()
        sc = SubconsciousMemory(config, embed_model)
        sc.rebuild_all()
        logger.info("潜意识向量库重建完成")
    elif stage == "rebuild_kb":
        from src.embedding.embed_model import EmbeddingModel
        from src.layers.knowledge_base import KnowledgeBase
        embed_model = EmbeddingModel(config)
        embed_model.load()
        kb = KnowledgeBase(config, embed_model)
        count = kb.rebuild()
        logger.info(f"知识库重建完成: {count} 条")
    elif stage == "all":
        phase_chunk(config)
        phase_check(config)
        asyncio.run(phase_annotate(config))
        asyncio.run(phase_reduce_l2(config))
        asyncio.run(phase_reduce_l1(config))
    else:
        print(f"未知阶段: {stage}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
