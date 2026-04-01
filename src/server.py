"""FastAPI server: /retrieve (personality assembly), /query (direct), /health, /ingest."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException

from src.config import Config
from src.embedding.embed_model import EmbeddingModel
from src.embedding.reranker import Reranker
from src.layers.assembler import Assembler
from src.layers.l1_core import L1CoreIdentity
from src.layers.l2_behavior import L2BehaviorRules
from src.layers.l3_episodic import L3EpisodicMemory
from src.layers.l4_working import L4WorkingMemory
from src.layers.step_a import StepA
from src.models import (
    HealthResponse,
    IngestRequest,
    QueryRequest,
    QueryResponse,
    QuerySceneResult,
    RetrieveRequest,
    RetrieveResponse,
)

logger = logging.getLogger(__name__)

# Global component references (set during lifespan)
_config: Config | None = None
_embed_model: EmbeddingModel | None = None
_reranker: Reranker | None = None
_l3: L3EpisodicMemory | None = None
_l2: L2BehaviorRules | None = None
_l1: L1CoreIdentity | None = None
_step_a: StepA | None = None
_l4: L4WorkingMemory | None = None
_assembler: Assembler | None = None
_subconscious = None  # SubconsciousMemory | None
_kb = None  # KnowledgeBase | None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup, cleanup on shutdown."""
    global _config, _embed_model, _reranker, _l3, _l2, _l1, _step_a, _l4, _assembler, _start_time
    _start_time = time.time()

    _config = Config()

    # Load embedding model (required for core functionality)
    _embed_model = EmbeddingModel(_config)
    try:
        _embed_model.load()
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        _embed_model = None

    # Load reranker (optional, degrades gracefully)
    _reranker = Reranker(_config)
    try:
        _reranker.load()
    except Exception as e:
        logger.warning(f"Failed to load reranker (will skip reranking): {e}")
        _reranker = None

    # Initialize L3 (requires embedding model)
    if _embed_model and _embed_model.is_loaded:
        _l3 = L3EpisodicMemory(_config, _embed_model, _reranker)
    else:
        logger.warning("L3 unavailable — embedding model not loaded")

    # Load L2 rules (optional)
    _l2 = L2BehaviorRules(_config)
    try:
        _l2.load()
    except Exception as e:
        logger.warning(f"Failed to load L2 rules: {e}")

    # Load L1 modules (optional)
    _l1 = L1CoreIdentity(_config)
    try:
        _l1.load()
    except Exception as e:
        logger.warning(f"Failed to load L1 modules: {e}")

    # Load Step A
    _step_a = StepA(_config)
    try:
        _step_a.load()
    except Exception as e:
        logger.warning(f"Failed to load StepA taxonomy: {e}")

    # Initialize L4
    _l4 = L4WorkingMemory()

    # Initialize subconscious memory
    global _subconscious
    if _config.get("memory.enabled", False) and _embed_model and _embed_model.is_loaded:
        from src.layers.subconscious import SubconsciousMemory
        _subconscious = SubconsciousMemory(_config, _embed_model)
        logger.info("Subconscious memory system enabled")

    # Initialize knowledge base
    global _kb
    if _embed_model and _embed_model.is_loaded:
        from src.layers.knowledge_base import KnowledgeBase
        _kb = KnowledgeBase(_config, _embed_model)
        logger.info(f"Knowledge base loaded: {_kb.count} entries")

    # Build assembler if minimum components available
    if _l3 and _step_a and _step_a.is_loaded:
        _assembler = Assembler(_config, _step_a, _l1, _l2, _l3, _l4, subconscious=_subconscious, knowledge_base=_kb)
    else:
        logger.warning("Assembler unavailable — missing Step A or L3")

    logger.info("Server startup complete")
    yield
    logger.info("Server shutting down")


app = FastAPI(title="Eris RAG", version="0.1.0", lifespan=lifespan)


# ── Endpoints ────────────────────────────────────────────────


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest, background_tasks: BackgroundTasks):
    """Full personality assembly pipeline: Step A → L1 → L2 → L3 → L4 → assemble."""
    logger.info(f"[/retrieve] sender={req.sender_id} nick={req.sender_nickname} msg={req.user_message[:80]}")
    if _assembler is None:
        raise HTTPException(503, "Assembler not available — check /health for details")

    result = await _assembler.assemble_async(
        user_message=req.user_message,
        conversation_context=req.conversation_context,
        sender_id=req.sender_id,
        sender_nickname=req.sender_nickname,
    )

    # 异步提取潜意识记忆（不阻塞响应）
    if _subconscious and _subconscious.enabled and req.user_message:
        identity = _assembler._resolve_identity(req.sender_id, req.sender_nickname)

        async def _safe_extract():
            try:
                await _subconscious.extract_and_store(
                    sender_id=req.sender_id,
                    identity=identity,
                    user_message=req.user_message,
                    bot_reply="",
                    conversation_context=req.conversation_context,
                )
            except Exception as e:
                logger.error(f"[subconscious] background task failed: {e}")

        background_tasks.add_task(_safe_extract)

    # Debug: 打印完整 system_prompt
    meta = result.metadata
    logger.info(f"[/retrieve] tokens={meta.total_tokens} l1={meta.l1_modules_used} l2={meta.l2_rules_used} l3={meta.l3_scenes_used}")
    logger.debug(f"[/retrieve] system_prompt:\n{result.system_prompt}")

    return RetrieveResponse(
        enhanced_system_prompt=result.system_prompt,
        metadata=result.metadata,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Direct query mode: semantic L3 retrieval without personality layers."""
    if _l3 is None:
        raise HTTPException(503, "L3 scene retrieval not available")

    raw_results = _l3.retrieve_raw(query=req.query, top_k=req.top_k)

    if req.format == "structured":
        results = [QuerySceneResult(**r) for r in raw_results]
        return QueryResponse(results=results)
    else:
        # Raw text format
        text_parts = []
        for r in raw_results:
            text_parts.append(
                f"[第{r['volume']}卷 第{r['chapter']}章] (相关度: {r['score']:.2f})\n"
                f"{r['summary']}\n{r['text']}"
            )
        return QueryResponse(
            results=[QuerySceneResult(**r) for r in raw_results],
            raw_text="\n\n---\n\n".join(text_parts),
        )


@app.post("/reset")
async def reset(req: dict):
    """Reset L4 session state for a user."""
    sender_id = req.get("sender_id", "")
    identity = ""
    if sender_id and _assembler:
        identity = _assembler._resolve_identity(sender_id, req.get("sender_nickname", ""))
    l4_key = f"{sender_id}_{identity}" if identity else sender_id
    if l4_key and _l4:
        _l4.reset(l4_key)
        logger.info(f"[/reset] session reset for {l4_key}")
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with component status."""
    return HealthResponse(
        status="ok",
        embedding_loaded=_embed_model is not None and _embed_model.is_loaded,
        reranker_loaded=_reranker is not None and _reranker.is_loaded,
        scene_count=_l3.scene_count if _l3 else 0,
        l1_loaded=_l1 is not None and _l1.is_loaded,
        l2_loaded=_l2 is not None and _l2.is_loaded,
        uptime_seconds=time.time() - _start_time,
    )


@app.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger preprocessing stages in background."""
    if _config is None:
        raise HTTPException(503, "Server not initialized")

    background_tasks.add_task(_run_ingest, req.stage)
    return {"status": "started", "stage": req.stage}


# ── Background tasks ─────────────────────────────────────────


async def _run_ingest(stage: str):
    """运行预处理流水线。

    阶段:
      - "chunk": Phase 1 — 分块 + embed + 入库（纯文本，无 LLM）
      - "annotate": Phase 2 — RAG + LLM 元数据标注
      - "reduce_l2": Phase 3 — RAG + LLM 行为规则归纳
      - "reduce_l1": Phase 4 — LLM 核心模块浓缩
      - "all": 全部按顺序执行
    """
    try:
        if stage in ("chunk", "all"):
            # Phase 1: 分块 + embed + 入库
            from src.preprocess.chunker import parse_novel_to_chunks

            novel_files = sorted(_config.novel_dir.glob("*.txt"))
            if not novel_files:
                logger.error(f"No novel files found in {_config.novel_dir}")
                return

            all_chunks = []
            for novel_file in novel_files:
                chunks = parse_novel_to_chunks(
                    str(novel_file),
                    str(_config.taxonomy_path),
                    target_size=_config.get("chunking.target_size", 384),
                    max_size=_config.get("chunking.max_size", 480),
                    overlap=_config.get("chunking.overlap", 64),
                )
                all_chunks.extend(chunks)

            if all_chunks and _l3:
                count = _l3.ingest_chunks(all_chunks)
                logger.info(f"Phase 1 完成: {count} chunks 入库，总计 {_l3.scene_count}")

        if stage in ("annotate", "all"):
            # Phase 2: RAG + LLM 元数据标注
            if _l3 and _l3.scene_count > 0:
                from src.preprocess.annotator import annotate_significance, annotate_tags

                stats = await annotate_tags(_l3, _config)
                logger.info(f"Phase 2a: tag 标注完成 {stats}")

                sig_count = await annotate_significance(_l3, _config)
                logger.info(f"Phase 2b: significance 标注完成, {sig_count} chunks")
            else:
                logger.warning("Phase 2 跳过: 向量库为空，请先运行 chunk 阶段")

        if stage in ("reduce_l2", "all"):
            # Phase 3: RAG + LLM 行为规则归纳
            if _l3 and _l3.scene_count > 0:
                from src.preprocess.reduce_l2 import reduce_rules_from_rag

                rules = await reduce_rules_from_rag(_l3, _config)
                logger.info(f"Phase 3 完成: {len(rules)} 条规则")

                if _l2:
                    _l2.load()
            else:
                logger.warning("Phase 3 跳过: 向量库为空")

        if stage in ("reduce_l1", "all"):
            # Load rules and run REDUCE L1
            rules_path = _config.character_data_dir / "l2_rules" / "rules.yaml"
            if rules_path.exists():
                import yaml
                from src.models import Rule
                from src.preprocess.reduce_l1 import reduce_core_modules

                with open(rules_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                rules = [Rule(**r) for r in data.get("rules", [])]

                modules = await reduce_core_modules(rules, _config)
                logger.info(f"REDUCE L1 complete: {len(modules)} modules")

                # Reload L1
                if _l1:
                    _l1.load()

    except Exception as e:
        logger.error(f"Ingest stage '{stage}' failed: {e}", exc_info=True)


# ── Entry point ──────────────────────────────────────────────


def main():
    """Run the server via uvicorn."""
    import signal
    import sys
    import uvicorn

    config = Config()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # 第三方库日志压到 WARNING，只让自己的 DEBUG 输出
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    # 捕获退出信号，打印原因
    def _on_signal(sig, frame):
        logger.error(f"Received signal {sig}, shutting down")
        sys.exit(1)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _on_signal)

    import atexit
    atexit.register(lambda: logger.error("SERVER PROCESS EXITING (atexit)"))

    try:
        uvicorn.run(
            "src.server:app",
            host=config.server_host,
            port=config.server_port,
            reload=False,
            timeout_keep_alive=0,
        )
    except Exception as e:
        logger.error(f"uvicorn.run() exception: {e}", exc_info=True)
    finally:
        logger.error("main() function ending")


if __name__ == "__main__":
    main()
