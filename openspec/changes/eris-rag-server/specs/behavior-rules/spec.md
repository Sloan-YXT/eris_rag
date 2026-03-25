## ADDED Requirements

### Requirement: REDUCE volume-level pattern extraction
The system SHALL process all scene records grouped by volume and call a configurable LLM API to extract volume-level behavioral observations. The prompt template SHALL be a separate Markdown file. Each observation SHALL include: pattern description, evidence scene IDs, exceptions, and changes from previous volumes.

#### Scenario: Extract patterns from one volume
- **WHEN** volume 3 has 80 scene records involving the target character
- **THEN** the LLM produces 10-20 behavioral observations with evidence references

#### Scenario: Use configurable API provider
- **WHEN** config specifies `preprocess.reduce_vol_patterns: "claude"`
- **THEN** the volume pattern extraction uses the Claude API provider

### Requirement: REDUCE global rule synthesis
The system SHALL feed all volume-level observations to a configurable LLM API and synthesize a global behavioral rule set. Rules SHALL be written for the target character's mature/late-stage behavior. Each rule SHALL include: id, domain (L1 domain tag), situation_tags (from shared taxonomy), condition, behavior, motivation, growth_trace, language_examples, exclusions, and evidence references.

#### Scenario: Synthesize rules across all volumes
- **WHEN** 24 volumes of behavioral observations (totaling ~60K tokens) are submitted
- **THEN** 50-100 behavioral rules are produced, each tagged with domain and situation_tags

### Requirement: Growth trace in rules
Each behavioral rule SHALL include a `growth_trace` field documenting how this behavior evolved across character periods. Rules SHALL describe the mature behavior as the primary, with earlier behaviors as historical context in the trace.

#### Scenario: Rule with growth trace
- **WHEN** a rule about "reacting to injury" is synthesized
- **THEN** the growth_trace shows: 少女期→暴怒赶人, 魔大陆→仍逞强但允许默默帮助, 圣地期→独自承受, 回归后→允许亲近的人帮忙但口是心非

### Requirement: Tag-based runtime matching
The system SHALL load all rules at startup and build an in-memory inverted index (tag → rule_ids). At runtime, given a set of triggers and an optional domain_filter, it SHALL return the top_k rules ranked by tag overlap score. Rules with `always` tag SHALL always be included.

#### Scenario: Match rules by triggers
- **WHEN** triggers=["受伤", "关心", "身体"] and domain_filter=["vulnerability"]
- **THEN** rules with situation_tags overlapping the triggers, filtered to domain "vulnerability", are returned ranked by overlap count

#### Scenario: Always-include speech style
- **WHEN** any retrieval is performed
- **THEN** the rule tagged with `always` (speech_style) is included regardless of trigger match

### Requirement: Dual output format
L2 runtime output SHALL contain two parts: (1) prompt_text — formatted rule content for injection into system_prompt, and (2) tags_for_l3 — the union of all matched rules' situation_tags, passed as metadata filter to L3.

#### Scenario: Output passes tags downstream
- **WHEN** rules beh_037 (tags: [受伤, 被关心, 逞强, 隐瞒]) and beh_012 (tags: [剑术]) are matched
- **THEN** tags_for_l3 = [受伤, 被关心, 逞强, 隐瞒, 剑术]
