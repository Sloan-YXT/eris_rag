## ADDED Requirements

### Requirement: Chapter splitting
The system SHALL parse a single novel TXT file and split it into individual chapters. Chapter boundaries SHALL be detected by the regex `^# 第(\d+)卷`. Multi-page chapters marked with `(X/Y)` suffixes SHALL be merged into a single chapter unit.

#### Scenario: Parse standard chapter header
- **WHEN** the parser encounters a line matching `^# 第3卷 少年期 第五话「出发」`
- **THEN** it starts a new chapter with volume=3, title="少年期 第五话「出发」"

#### Scenario: Merge paginated chapters
- **WHEN** a chapter appears as `第5卷 少年期 第三话「父子吵架」(3/4)` followed by `(4/4)`
- **THEN** the content of both pages is merged into a single chapter unit

### Requirement: Metadata cleaning
The system SHALL remove non-story content including: underscored title lines (`^_第[^_]+_$`), source attribution (`铅笔小说`, `www.x23qb.com`), publisher metadata (`台版`, `扫图`, `录入`, `着:`, `译:`), and consecutive blank lines (collapse to one).

#### Scenario: Clean source attribution
- **WHEN** a line contains `铅笔小说` or `www.x23qb.com`
- **THEN** the line is removed from output

### Requirement: Scene segmentation
The system SHALL split each chapter into scenes using the `★ ★ ★` marker as scene boundaries. Each scene SHALL be the minimum indexable unit for L3.

#### Scenario: Split on scene break marker
- **WHEN** a chapter contains two `★ ★ ★` markers
- **THEN** the chapter is split into three scenes, each with a unique scene_id formatted as `v{vol}_c{chap}_s{seq}`

### Requirement: Volume-to-period mapping
The system SHALL map each volume number to a character period via a configurable mapping table. Default mapping: volumes 1-3 → 少女期, volumes 4-6 → 魔大陆流浪期, volumes 7-15 → 剑之圣地期, volumes 16-24 → 回归后.

#### Scenario: Assign period to scene
- **WHEN** a scene is extracted from volume 3
- **THEN** the scene's `period` field is set to "少女期" and `period_weight` is set according to the weight table in config

### Requirement: MAP scene extraction via LLM
The system SHALL call a configurable LLM API for each chapter to extract structured scene records for the target character. The LLM prompt template SHALL be stored as a separate Markdown file. Each scene record SHALL include: id, volume, chapter, summary, dialogue (character's original lines), actions, inner_thought, emotion, situation_tags (from predefined taxonomy), scene_type, context, and embedding_text.

#### Scenario: Extract scenes from a chapter
- **WHEN** MAP is invoked on a chapter containing 3 scenes involving the target character
- **THEN** 3 structured Scene objects are produced, each with all required fields populated

#### Scenario: Use configurable API provider
- **WHEN** config specifies `preprocess.map_extract: "gemini"`
- **THEN** the MAP stage uses the Gemini API provider for LLM calls

### Requirement: Batch processing with rate limiting
The system SHALL support configurable batch size and inter-request delay for MAP processing to avoid API rate limits. Processing SHALL be resumable — already-processed chapters are skipped on re-run.

#### Scenario: Resume interrupted processing
- **WHEN** MAP processing is interrupted after 50 of 275 chapters
- **THEN** re-running MAP skips the 50 completed chapters and continues from chapter 51
