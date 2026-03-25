## 1. Project Setup

- [x] 1.1 Create pyproject.toml with all dependencies (fastapi, uvicorn, chromadb, jieba, pyyaml, httpx, pydantic, sentence-transformers, FlagEmbedding, torch, tiktoken)
- [x] 1.2 Create src/ package structure: config.py, models.py, server.py, and subpackages preprocess/, layers/, embedding/, llm/
- [x] 1.3 Define tags_taxonomy.yaml with 6 domain tags and ~40 situation tags
- [x] 1.4 Create config.yaml with all settings: server, API providers (per-stage preprocess + runtime configs), embedding, reranker, vectordb, character, retrieval params
- [x] 1.5 Implement config.py: YAML loading with environment variable expansion for API keys
- [x] 1.6 Implement models.py: Pydantic models for Scene, Rule, Module, StepAResult, L1Result, L2Result, L3Result, L4State, AssemblyResult, API request/response schemas

## 2. LLM Client Abstraction

- [x] 2.1 Implement llm/client.py: unified LLMClient class with provider dispatch (gemini, claude, openai)
- [x] 2.2 Support per-call provider selection via provider name string from config
- [x] 2.3 Implement prompt template loading from Markdown files in preprocess/prompts/

## 3. Novel Ingestion (novel-ingestion spec)

- [x] 3.1 Implement preprocess/chunker.py: parse TXT, split by `^# 第(\d+)卷` regex, merge `(X/Y)` paginated parts
- [x] 3.2 Implement metadata cleaning: remove source attribution, publisher metadata, underscored titles, collapse blank lines
- [x] 3.3 Implement scene segmentation by `★ ★ ★` markers, generate scene_id as `v{vol}_c{chap}_s{seq}`
- [x] 3.4 Implement configurable volume-to-period mapping with default mapping for Mushoku Tensei
- [x] 3.5 Write MAP prompt template (preprocess/prompts/map_scene_extract.md) for structured scene extraction
- [x] 3.6 Implement preprocess/map_extract.py: call LLM API per chapter, parse structured Scene output, write to scenes.jsonl
- [x] 3.7 Add batch processing with configurable concurrency and delay, plus resume support (skip already-processed chapters)

## 4. Local Embedding & Reranker (scene-retrieval spec)

- [x] 4.1 Implement embedding/embed_model.py: load BGE-large-zh-v1.5 on GPU, expose encode() with batch support
- [x] 4.2 Implement embedding/reranker.py: load BGE-reranker-v2-m3 on GPU, expose rank(query, documents, top_k)

## 5. Scene Retrieval — L3 (scene-retrieval spec)

- [x] 5.1 Implement layers/l3_episodic.py: ChromaDB collection setup, scene ingestion with metadata (situation_tags, period, period_weight, etc.)
- [x] 5.2 Implement filtered semantic retrieval: metadata filter by situation_tags (any-of), combined with embedding cosine similarity
- [x] 5.3 Implement period_weight scoring: multiply similarity by period_weight, invert weights when topic_is_past=true
- [x] 5.4 Integrate reranker: coarse candidates → reranker → final top_k
- [x] 5.5 Implement period-aware output templates: 行为参考 (current period) vs 你的记忆 (earlier periods with memory_significance)
- [x] 5.6 Ensure standalone usability: retrieval works with no filter_tags (pure semantic search)

## 6. Behavior Rules — L2 (behavior-rules spec)

- [x] 6.1 Write REDUCE volume-level prompt template (preprocess/prompts/reduce_vol_patterns.md)
- [x] 6.2 Write REDUCE global rules prompt template (preprocess/prompts/reduce_global_rules.md)
- [x] 6.3 Implement preprocess/reduce_l2.py: volume-level pattern extraction → global rule synthesis, output rules.yaml
- [x] 6.4 Implement layers/l2_behavior.py: load rules.yaml, build in-memory tag→rule_id inverted index
- [x] 6.5 Implement runtime tag matching: input triggers + domain_filter, output prompt_text + tags_for_l3
- [x] 6.6 Ensure `always`-tagged rules (speech_style) are included in every query

## 7. Core Identity — L1 (core-identity spec)

- [x] 7.1 Write REDUCE core modules prompt template (preprocess/prompts/reduce_core_modules.md)
- [x] 7.2 Implement preprocess/reduce_l1.py: synthesize L2 rules into 6 domain modules, output l1_modules/*.yaml
- [x] 7.3 Implement layers/l1_core.py: load modules at startup, match triggers against activate_on tags
- [x] 7.4 Implement runtime module selection: always include core, select others by trigger overlap, output prompt_text + active_domains

## 8. Assembly Pipeline (personality-assembly spec)

- [x] 8.1 Implement layers/step_a.py: jieba + keyword dictionary → triggers, topic_is_past detection, emotion_hint
- [x] 8.2 Create keyword-to-tag dictionary for jieba-based trigger extraction
- [x] 8.3 Implement layers/l4_working.py: per-user session state (dict), rule-based emotion update with decay
- [x] 8.4 Implement layers/assembler.py: execute L1→L2→L3 pipeline, combine outputs with L4 state and meta-instruction
- [x] 8.5 Implement token budget enforcement: trim L3 first, then L2, never trim L1 core or L4
- [x] 8.6 Write meta-instruction template for roleplay guidance

## 9. Direct Query Mode (direct-query spec)

- [x] 9.1 Implement direct query path: semantic L3 retrieval without L1/L2/L4, returning raw or structured results
- [x] 9.2 Ensure zero dependency on personality layer modules (import only l3_episodic and embedding)

## 10. RAG Server (rag-server spec)

- [x] 10.1 Implement server.py: FastAPI app with startup event (load models, DB, L1, L2)
- [x] 10.2 Implement POST /retrieve endpoint: full personality assembly pipeline
- [x] 10.3 Implement POST /query endpoint: direct query mode
- [x] 10.4 Implement GET /health endpoint: model status, scene count, L1/L2 status
- [x] 10.5 Implement POST /ingest endpoint: trigger preprocessing stages in background
- [x] 10.6 Implement degraded startup: skip unavailable components, reflect in /health
- [x] 10.7 Configure binding to 0.0.0.0 with configurable host/port

## 11. AstrBot Plugin (astrbot-plugin spec)

- [x] 11.1 Create astrbot_plugin/main.py: Star class with @register decorator
- [x] 11.2 Implement @filter.on_llm_request() handler: extract message, call RAG /retrieve, inject system_prompt
- [x] 11.3 Implement /ask command detection: route to /query endpoint, append result to prompt context
- [x] 11.4 Implement graceful fallback: timeout/error → use original system_prompt, log error
- [x] 11.5 Create _conf_schema.json: rag_server_url, enabled, timeout_ms, query_command_prefix
- [x] 11.6 Create requirements.txt for plugin dependencies (httpx, aiohttp)

## 12. Testing

- [x] 12.1 Write test_step_a.py: verify trigger extraction for various input messages
- [x] 12.2 Write test_l1.py: verify module selection given different trigger sets
- [x] 12.3 Write test_l2.py: verify tag matching with domain filter and always-include rules
- [x] 12.4 Write test_l3.py: verify retrieval with synthetic scenes, period weighting, metadata filtering
- [x] 12.5 Write test_assembler.py: verify full pipeline assembly, token budget enforcement
- [x] 12.6 Write test_server.py: API endpoint integration tests for /retrieve, /query, /health
- [x] 12.7 Write test_chunker.py: verify chapter splitting, pagination merging, metadata cleaning on actual novel file

## 13. Preprocessing Execution

- [ ] 13.1 Run chunker on novel file, verify chapter count and scene segmentation
- [ ] 13.2 Run MAP extraction on 2-3 test chapters, review scene record quality (🔴 human review checkpoint)
- [ ] 13.3 Run full MAP extraction on all 275 chapters
- [ ] 13.4 Embed all scenes and ingest into ChromaDB
- [ ] 13.5 Run REDUCE L2: volume patterns → global rules (🔴 human review checkpoint)
- [ ] 13.6 Run REDUCE L1: rules → core modules (🔴 human review checkpoint)
- [ ] 13.7 Verify end-to-end: start server, send test queries, inspect system_prompt output
