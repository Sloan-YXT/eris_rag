## ADDED Requirements

### Requirement: Personality enhancement endpoint
The server SHALL expose `POST /retrieve` accepting: user_message (string), conversation_context (list of recent messages, optional), sender_id (string), and returning: enhanced_system_prompt (string), metadata (object with l1_modules_used, l2_rules_used, l3_scenes_used, total_tokens).

#### Scenario: Full personality retrieval
- **WHEN** POST /retrieve with user_message="你手怎么了" and sender_id="user_123"
- **THEN** response contains enhanced_system_prompt (~2000 tokens) and metadata listing which L1/L2/L3 elements were used

### Requirement: Direct query endpoint
The server SHALL expose `POST /query` accepting: query (string), top_k (int, optional, default 3), format (string, "raw"|"structured", default "raw"), and returning scene results without personality framing.

#### Scenario: Direct novel query
- **WHEN** POST /query with query="转移事件发生在第几卷"
- **THEN** response contains matching scene texts in the requested format

### Requirement: Health check endpoint
The server SHALL expose `GET /health` returning server status including: model loading status (embedding, reranker), vector DB stats (total scenes), L1/L2 loading status, and uptime.

#### Scenario: Health check
- **WHEN** GET /health is called after successful startup
- **THEN** response shows all models loaded, scene count, and L1/L2 status

### Requirement: Ingestion trigger endpoint
The server SHALL expose `POST /ingest` to trigger the preprocessing pipeline. It SHALL accept: stage (string, "map"|"reduce_l2"|"reduce_l1"|"all"), and optional parameters. Long-running stages SHALL run in the background and report progress.

#### Scenario: Trigger MAP stage
- **WHEN** POST /ingest with stage="map"
- **THEN** the MAP extraction pipeline starts in the background, with progress queryable

### Requirement: Startup model loading
On startup, the server SHALL load: embedding model to GPU, reranker model to GPU, ChromaDB collection, L1 modules from YAML, L2 rules and tag index. If any component fails to load, the server SHALL start in degraded mode with the failed component disabled and /health reflecting the status.

#### Scenario: Degraded startup without L2
- **WHEN** L2 rules files are missing at startup
- **THEN** server starts, /health shows L2 as unavailable, /retrieve falls back to L1+L3 only, /query works normally

### Requirement: Network binding
The server SHALL bind to `0.0.0.0` by default to accept connections from remote AstrBot. Host and port SHALL be configurable.

#### Scenario: Remote access
- **WHEN** server is running on 0.0.0.0:8787
- **THEN** AstrBot on a remote machine can reach POST /retrieve
