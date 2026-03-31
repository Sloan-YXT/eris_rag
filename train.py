"""人格树构建脚本 — 从小说原文生成完整的角色认知模型。

用法:
  python train.py                    # 交互式引导，逐步确认
  python train.py --all              # 全部自动跑（跳过确认）
  python train.py --step chunk       # 只跑某一步
  python train.py --step annotate
  python train.py --step reduce_l2
  python train.py --step reduce_l1
  python train.py --rebuild-memory   # 重建潜意识向量库

前置条件:
  1. 小说 TXT 文件放在 config.yaml 中 character.novel_dir 指定的目录
  2. config.yaml 中配置好 LLM API key
  3. pip install -r requirements.txt
"""

import asyncio
import argparse
import json
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train")


def get_config():
    from src.config import Config
    return Config()


def confirm(msg: str) -> bool:
    """交互确认，返回 True 继续，False 跳过。"""
    resp = input(f"\n{msg} [Y/n] ").strip().lower()
    return resp in ("", "y", "yes")


# ══════════════════════════════════════════════════════════
# Step 1: 分块 + embed + 入库
# ══════════════════════════════════════════════════════════

def step_chunk(config):
    """Phase 1: 小说原文 → 固定窗口分块 → BGE embed → ChromaDB。"""
    from src.preprocess.chunker import parse_novel_to_chunks
    from src.embedding.embed_model import EmbeddingModel
    from src.layers.l3_episodic import L3EpisodicMemory

    novel_files = sorted(config.novel_dir.glob("*.txt"))
    if not novel_files:
        logger.error(f"没有找到小说文件: {config.novel_dir}")
        logger.error("请把小说 TXT 放到该目录下")
        return False

    logger.info(f"找到 {len(novel_files)} 个文件: {[f.name for f in novel_files]}")

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

    if not all_chunks:
        logger.error("分块结果为空，请检查小说文件格式")
        return False

    sizes = [len(c.raw_text) for c in all_chunks]
    logger.info(f"分块完成: {len(all_chunks)} 个 chunks")
    logger.info(f"  大小: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    periods = {}
    for c in all_chunks:
        periods[c.period] = periods.get(c.period, 0) + 1
    logger.info(f"  各时期: {periods}")

    embed_model = EmbeddingModel(config, use_train_device=True)
    embed_model.load()

    l3 = L3EpisodicMemory(config, embed_model)
    t0 = time.time()
    count = l3.ingest_chunks(all_chunks)
    elapsed = time.time() - t0
    logger.info(f"入库完成: {count} 新增, 总计 {l3.scene_count}, 耗时 {elapsed:.1f}s")
    return True


# ══════════════════════════════════════════════════════════
# Step 2: tag 标注 + significance 标注
# ══════════════════════════════════════════════════════════

async def step_annotate(config):
    """Phase 2: 用 RAG + LLM 为每个 chunk 添加 situation_tags 和 significance。"""
    from src.embedding.embed_model import EmbeddingModel
    from src.layers.l3_episodic import L3EpisodicMemory
    from src.preprocess.annotator import annotate_tags, annotate_significance

    embed_model = EmbeddingModel(config, use_train_device=True)
    embed_model.load()
    l3 = L3EpisodicMemory(config, embed_model)

    if l3.scene_count == 0:
        logger.error("向量库为空，请先运行 chunk")
        return False

    logger.info(f"向量库: {l3.scene_count} chunks")

    logger.info("Phase 2a: tag 标注...")
    stats = await annotate_tags(l3, config)
    total = sum(stats.values())
    logger.info(f"tag 标注完成: {total} 个标注")
    for tag, count in stats.items():
        logger.info(f"  {tag}: {count}")

    logger.info("Phase 2b: significance 标注...")
    sig_count = await annotate_significance(l3, config)
    logger.info(f"significance 标注: {sig_count} chunks")
    return True


# ══════════════════════════════════════════════════════════
# Step 3: L2 行为规则
# ══════════════════════════════════════════════════════════

async def step_reduce_l2(config):
    """Phase 3: 从 RAG 检索结果归纳行为规则。"""
    from src.embedding.embed_model import EmbeddingModel
    from src.layers.l3_episodic import L3EpisodicMemory
    from src.preprocess.reduce_l2 import reduce_rules_from_rag

    embed_model = EmbeddingModel(config, use_train_device=True)
    embed_model.load()
    l3 = L3EpisodicMemory(config, embed_model)

    if l3.scene_count == 0:
        logger.error("向量库为空，请先运行 chunk")
        return False

    rules = await reduce_rules_from_rag(l3, config)
    logger.info(f"生成 {len(rules)} 条规则:")
    for r in rules:
        logger.info(f"  [{r.id}] {r.domain}: {r.condition}")

    output = config.character_data_dir / "l2_rules" / "rules.yaml"
    logger.info(f"保存到 {output}")
    return True


# ══════════════════════════════════════════════════════════
# Step 4: L1 核心模块
# ══════════════════════════════════════════════════════════

async def step_reduce_l1(config):
    """Phase 4: 从 L2 规则浓缩为 6 个核心身份模块。"""
    import yaml
    from src.models import Rule
    from src.preprocess.reduce_l1 import reduce_core_modules

    rules_path = config.character_data_dir / "l2_rules" / "rules.yaml"
    if not rules_path.exists():
        logger.error(f"L2 规则文件不存在: {rules_path}，请先运行 reduce_l2")
        return False

    with open(rules_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rules = [Rule(**r) for r in data.get("rules", [])]

    modules = await reduce_core_modules(rules, config)
    logger.info(f"生成 {len(modules)} 个核心模块:")
    for m in modules:
        logger.info(f"  [{m.id}] {m.domain}")

    # 修复 activate_on（从 taxonomy 读取正确的 tags）
    _fix_activate_on(config)
    return True


def _fix_activate_on(config):
    """确保 L1 模块的 activate_on 使用 taxonomy 中定义的实际 tags。"""
    import yaml

    with open(config.taxonomy_path, encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    domains = tax.get("domains", {})
    if not domains:
        return

    modules_dir = config.character_data_dir / "l1_modules"
    for f in sorted(modules_dir.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        domain = data["domain"]
        if domain in domains:
            data["activate_on"] = domains[domain].get("activate_on", [])
            with open(f, "w", encoding="utf-8") as fh:
                yaml.dump(data, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ══════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════

def rebuild_memory(config):
    """重建所有用户的潜意识向量库。"""
    from src.embedding.embed_model import EmbeddingModel
    from src.layers.subconscious import SubconsciousMemory
    embed_model = EmbeddingModel(config, use_train_device=True)
    embed_model.load()
    sc = SubconsciousMemory(config, embed_model)
    sc.rebuild_all()
    logger.info("潜意识向量库重建完成")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="人格树构建")
    parser.add_argument("--all", action="store_true", help="全部自动跑")
    parser.add_argument("--step", type=str, help="只跑某一步: chunk/annotate/reduce_l2/reduce_l1")
    parser.add_argument("--rebuild-memory", action="store_true", help="重建潜意识向量库")
    args = parser.parse_args()

    config = get_config()

    if args.rebuild_memory:
        rebuild_memory(config)
        return

    if args.step:
        _run_step(args.step, config)
        return

    # 交互式引导
    print("=" * 60)
    print("  人格树构建 — 从小说原文生成角色认知模型")
    print("=" * 60)
    print()
    print(f"小说目录: {config.novel_dir}")
    print(f"角色数据: {config.character_data_dir}")
    print(f"标注模型: {config.get_preprocess_provider('annotate_tags')}")
    print(f"规则模型: {config.get_preprocess_provider('reduce_global_rules')}")
    print()

    auto = args.all

    # Step 1
    if auto or confirm("Step 1: 分块 + embed + 入库？"):
        logger.info("=" * 40 + " Step 1: chunk " + "=" * 40)
        if not step_chunk(config):
            return

    # Step 2
    if auto or confirm("Step 2: tag 标注 + significance 标注？（需要调 LLM API，约 5-10 分钟）"):
        logger.info("=" * 40 + " Step 2: annotate " + "=" * 40)
        if not asyncio.run(step_annotate(config)):
            return

    # Step 3
    if auto or confirm("Step 3: 生成 L2 行为规则？（需要调 LLM API，约 10-20 分钟）"):
        logger.info("=" * 40 + " Step 3: reduce_l2 " + "=" * 40)
        if not asyncio.run(step_reduce_l2(config)):
            return

    # Step 4
    if auto or confirm("Step 4: 生成 L1 核心模块？（需要调 LLM API，约 2-5 分钟）"):
        logger.info("=" * 40 + " Step 4: reduce_l1 " + "=" * 40)
        if not asyncio.run(step_reduce_l1(config)):
            return

    print()
    print("=" * 60)
    print("  构建完成！可以启动服务器了：")
    print("  python -m src.server")
    print("=" * 60)


def _run_step(step: str, config):
    steps = {
        "chunk": lambda: step_chunk(config),
        "annotate": lambda: asyncio.run(step_annotate(config)),
        "reduce_l2": lambda: asyncio.run(step_reduce_l2(config)),
        "reduce_l1": lambda: asyncio.run(step_reduce_l1(config)),
    }
    if step not in steps:
        print(f"未知步骤: {step}")
        print(f"可选: {', '.join(steps.keys())}")
        sys.exit(1)
    steps[step]()


if __name__ == "__main__":
    main()
