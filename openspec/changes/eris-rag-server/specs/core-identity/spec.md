## ADDED Requirements

### Requirement: REDUCE core module generation
The system SHALL feed the complete L2 rule set to a configurable LLM API and generate a set of identity modules, one per domain. Default domains: core, vulnerability, combat, relationship, pride, growth. Each module SHALL include: id, domain, activate_on tags (from the shared taxonomy), and prompt_text (200-400 tokens). The prompt template SHALL be a separate Markdown file.

#### Scenario: Generate identity modules
- **WHEN** the L2 rule set (~80 rules) is submitted for core synthesis
- **THEN** 6 YAML module files are produced, one per domain

#### Scenario: Use configurable API provider
- **WHEN** config specifies `preprocess.reduce_core_modules: "claude"`
- **THEN** the core module generation uses the Claude API provider

### Requirement: Module content with causal chains
Each module's prompt_text SHALL express the character's traits as causal chains (because X happened → she now does Y) rather than flat labels. The core module SHALL be under 250 tokens. Other modules SHALL be 200-400 tokens each.

#### Scenario: Core module uses causal chains
- **WHEN** the core module is generated
- **THEN** it contains statements like "你用力量证明价值——因为学习总是失败，力量成了你唯一被认可的东西" rather than "你很暴躁"

### Requirement: Dynamic module selection at runtime
The system SHALL load all modules at startup. At runtime, given a set of triggers from Step A, it SHALL select the core module (always) plus any modules whose `activate_on` tags intersect with the triggers. Output SHALL include: prompt_text (concatenated selected modules) and active_domains (list of selected domain names, passed to L2 as filter).

#### Scenario: Select modules by triggers
- **WHEN** triggers=["受伤", "关心"] and vulnerability.activate_on contains "受伤" and relationship.activate_on contains "关心"
- **THEN** selected modules are: core + vulnerability + relationship; active_domains=["vulnerability", "relationship"]

#### Scenario: Core always selected
- **WHEN** triggers=["日常闲聊"] and no non-core module's activate_on matches
- **THEN** only the core module is selected; active_domains=[]
