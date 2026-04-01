"""Pydantic data models for all layers and API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── L3 Chunk (原文分块，embedding 的基本单位) ─────────────

class Chunk(BaseModel):
    """一个原文分块。父子块模式下，子块用于搜索，父块用于返回上下文。"""
    id: str                                     # 子块: "v{vol}_c{chap}_{seq}_s{sub}" 父块: "v{vol}_c{chap}_{seq}"
    raw_text: str                               # 原文片段
    volume: int
    chapter: int = 0
    chunk_index: int = 0                        # 全书中的序号
    char_offset: int = 0                        # 在源文件中的字符偏移
    source_file: str = ""                       # 来源文件名
    parent_id: str = ""                         # 子块指向父块的 ID（父块本身为空）
    is_child: bool = False                      # 是否为子块

    # Period（从卷号推断）
    period: str = ""
    period_weight: float = 1.0

    # 元数据（Phase 2 标注后填充，初始为空）
    situation_tags: list[str] = Field(default_factory=list)
    has_eris: bool | None = None                # None = 未标注
    emotion: str = ""


# ── L3 Scene (旧模型，保留兼容) ──────────────────────────

class Scene(BaseModel):
    """A single extracted scene from the novel (legacy, kept for compatibility)."""
    id: str
    volume: int
    chapter: int
    scene_index: int = 0
    summary: str = ""
    dialogue: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    inner_thought: str = ""
    context: str = ""
    emotion: str = ""
    situation_tags: list[str] = Field(default_factory=list)
    scene_type: str = "narration"
    speaker: str = ""
    characters_present: list[str] = Field(default_factory=list)
    period: str = ""
    period_weight: float = 1.0
    memory_significance: str = ""
    raw_text: str = ""
    embedding_text: str = ""


# ── L2 Behavioral Rule ───────────────────────────────────

class Rule(BaseModel):
    """A behavioral rule extracted from scene patterns."""
    id: str
    domain: str                                 # L1 domain tag
    situation_tags: list[str] = Field(default_factory=list)
    priority: int = 1                           # 0 = always include

    condition: str = ""
    behavior: str = ""
    motivation: str = ""
    growth_trace: str = ""
    language_examples: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    evidence_scenes: list[str] = Field(default_factory=list)


# ── L1 Identity Module ───────────────────────────────────

class Module(BaseModel):
    """An identity module representing one domain of the character."""
    id: str                                     # e.g. "core", "vulnerability"
    domain: str
    activate_on: list[str] = Field(default_factory=list)
    prompt_text: str = ""


# ── Step A Result ─────────────────────────────────────────

class StepAResult(BaseModel):
    """Output of intent analysis."""
    triggers: list[str] = Field(default_factory=list)
    topic_is_past: bool = False
    emotion_hint: str = ""
    search_queries: list[str] = Field(default_factory=list)  # LLM 生成的语义检索短语
    keywords: list[str] = Field(default_factory=list)         # LLM 提取的专有名词（用于精确匹配）


# ── Layer Results ─────────────────────────────────────────

class L1Result(BaseModel):
    """Output of L1 module selection."""
    prompt_text: str = ""
    active_domains: list[str] = Field(default_factory=list)
    modules_used: list[str] = Field(default_factory=list)


class L2Result(BaseModel):
    """Output of L2 rule matching."""
    prompt_text: str = ""
    tags_for_l3: list[str] = Field(default_factory=list)
    rules_used: list[str] = Field(default_factory=list)


class L3Result(BaseModel):
    """Output of L3 scene retrieval."""
    prompt_text: str = ""
    scenes_used: list[str] = Field(default_factory=list)


class L4State(BaseModel):
    """Per-user session state."""
    current_emotion: str = "平静"
    emotion_intensity: float = 0.3
    conversation_turns: int = 0
    active_topics: list[str] = Field(default_factory=list)
    summary: str = ""
    relationship_estimate: str = "未知"


class AssemblyResult(BaseModel):
    """Final assembled system prompt with metadata."""
    system_prompt: str = ""
    metadata: AssemblyMetadata = Field(default_factory=lambda: AssemblyMetadata())


class AssemblyMetadata(BaseModel):
    l1_modules_used: list[str] = Field(default_factory=list)
    l2_rules_used: list[str] = Field(default_factory=list)
    l3_scenes_used: list[str] = Field(default_factory=list)
    total_tokens: int = 0


# ── API Request/Response Schemas ──────────────────────────

class RetrieveRequest(BaseModel):
    """POST /retrieve request body."""
    user_message: str
    conversation_context: list[str] = Field(default_factory=list)
    sender_id: str = "default"          # QQ号或平台ID，用于匹配 user_prompts
    sender_nickname: str = ""           # 昵称，直接传给LLM


class RetrieveResponse(BaseModel):
    """POST /retrieve response body."""
    enhanced_system_prompt: str
    metadata: AssemblyMetadata


class QueryRequest(BaseModel):
    """POST /query request body."""
    query: str
    top_k: int = 3
    format: str = "raw"                         # "raw" | "structured"


class QueryResponse(BaseModel):
    """POST /query response body."""
    results: list[QuerySceneResult] = Field(default_factory=list)
    raw_text: str = ""


class QuerySceneResult(BaseModel):
    """A single scene in structured query results."""
    model_config = {"extra": "ignore"}
    scene_id: str
    volume: int
    chapter: int
    summary: str
    text: str
    score: float = 0.0


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str = "ok"
    embedding_loaded: bool = False
    reranker_loaded: bool = False
    scene_count: int = 0
    l1_loaded: bool = False
    l2_loaded: bool = False
    uptime_seconds: float = 0.0


class IngestRequest(BaseModel):
    """POST /ingest request body."""
    stage: str = "all"                          # "chunk" | "annotate" | "reduce_l2" | "reduce_l1" | "all"
