## ADDED Requirements

### Requirement: Direct retrieval mode
The system SHALL support a direct query mode that bypasses L1, L2, and L4 entirely. Given a user query, it SHALL perform L3 semantic retrieval (optionally with reranking) and return matching scene texts as plain context, not formatted as personality prompts.

#### Scenario: Query novel details
- **WHEN** a direct query "艾莉丝什么时候学会了剑王级剑术" is submitted
- **THEN** the system returns the top_k most relevant original scene texts without personality framing

### Requirement: Configurable output format
Direct query results SHALL support two output formats: (1) raw — scene texts concatenated for direct LLM prompt injection, and (2) structured — JSON array of scenes with metadata (scene_id, volume, chapter, summary, text).

#### Scenario: Raw output for prompt injection
- **WHEN** format="raw" is specified
- **THEN** scene texts are concatenated with separators, suitable for appending to an LLM prompt

#### Scenario: Structured output for display
- **WHEN** format="structured" is specified
- **THEN** a JSON array is returned with scene objects including metadata

### Requirement: No personality layer dependency
The direct-query module SHALL NOT import or depend on L1, L2, L4, or assembler modules. It SHALL only depend on scene-retrieval (L3). This ensures the direct query path works even if personality layers are removed or broken.

#### Scenario: Works without personality layers
- **WHEN** L1 modules, L2 rules, and L4 state files are absent or empty
- **THEN** direct query still functions correctly using only L3 vector search
