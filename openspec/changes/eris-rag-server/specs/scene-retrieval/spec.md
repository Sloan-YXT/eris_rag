## ADDED Requirements

### Requirement: Local embedding model
The system SHALL load an embedding model (default: BAAI/bge-large-zh-v1.5) on the local GPU for vector encoding. The model name and device SHALL be configurable. No API calls SHALL be made for embedding.

#### Scenario: Embed scene text on GPU
- **WHEN** a scene's embedding_text is submitted for encoding
- **THEN** a 1024-dimensional vector is returned, computed on the local GPU

### Requirement: Vector storage with metadata
The system SHALL store scene embeddings in ChromaDB with full metadata: scene_id, volume, chapter, period, period_weight, situation_tags, emotion, scene_type, speaker, characters_present. The persist directory SHALL be configurable.

#### Scenario: Store and persist scenes
- **WHEN** 1500 scene embeddings with metadata are ingested
- **THEN** they are persisted to disk and survive server restart

### Requirement: Filtered semantic retrieval
The system SHALL support combined semantic search with metadata filtering. Filters SHALL support: situation_tags (any-of match), period, scene_type, speaker. The number of candidates (pre-rerank) SHALL be configurable.

#### Scenario: Filter by situation tags
- **WHEN** a query is submitted with filter_tags=["受伤", "逞强"]
- **THEN** only scenes whose situation_tags overlap with the filter set are considered for semantic ranking

### Requirement: Period-weighted scoring
The system SHALL multiply the raw semantic similarity score by the scene's period_weight before ranking. When `topic_is_past` is true, the weight table SHALL be inverted (earlier periods get higher weight).

#### Scenario: Boost recent-period scenes by default
- **WHEN** a scene from 回归后 (weight=1.3) has similarity 0.70 and a scene from 少女期 (weight=0.85) has similarity 0.90
- **THEN** the weighted scores are 0.91 and 0.765, ranking 回归后 first

#### Scenario: Invert weights for past-topic queries
- **WHEN** `topic_is_past=true` and a 少女期 scene has similarity 0.80
- **THEN** the weight flips to 1.3, giving a weighted score of 1.04

### Requirement: Local reranker
The system SHALL load a reranker model (default: BAAI/bge-reranker-v2-m3) on the local GPU. After coarse retrieval, the top candidates SHALL be re-scored by the reranker and the final top_k returned. No API calls SHALL be made for reranking.

#### Scenario: Rerank candidates
- **WHEN** 6 coarse candidates are retrieved for query "你手怎么了"
- **THEN** the reranker re-scores all 6 against the query and returns the top 3

### Requirement: Period-aware output templates
The system SHALL format retrieved scenes differently based on their period relative to the target period. Scenes from the target period use a "behavioral reference" template. Scenes from earlier periods use a "memory" template that includes the memory_significance field.

#### Scenario: Format early-period scene as memory
- **WHEN** a retrieved scene has period="少女期" and the target period is "回归后"
- **THEN** it is formatted as "【你的记忆】{period}时，{summary}..."

#### Scenario: Format current-period scene as reference
- **WHEN** a retrieved scene has period="回归后"
- **THEN** it is formatted as "【行为参考】{summary}..."

### Requirement: Standalone usability
The scene-retrieval module SHALL be usable independently of L1, L2, and L4. When called without filter_tags or domain constraints, it SHALL perform pure semantic retrieval across all scenes.

#### Scenario: Use without personality layers
- **WHEN** scene retrieval is invoked with only a semantic query and no filter_tags
- **THEN** it returns the top_k most semantically similar scenes from the entire collection
