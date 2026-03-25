## ADDED Requirements

### Requirement: Intent analysis (Step A)
The system SHALL analyze the user's message to extract triggers, topic_is_past, and emotion_hint. The default implementation SHALL use jieba segmentation + a keyword dictionary mapping words to situation_tags. The implementation SHALL be swappable to an LLM API call via config (`runtime.step_a`).

#### Scenario: Extract triggers via local dictionary
- **WHEN** user message is "你手怎么了" and config runtime.step_a="local"
- **THEN** jieba segments + dictionary lookup produces triggers=["受伤", "关心", "身体"]

#### Scenario: Detect past-topic
- **WHEN** user message contains "以前", "小时候", "还记得", or "那时候"
- **THEN** topic_is_past=true

### Requirement: Working memory (L4)
The system SHALL maintain per-user session state in memory, including: current_emotion, emotion_intensity (0.0-1.0), conversation_turns, active_topics, summary, and relationship_estimate. State SHALL be updated after each interaction. Default update logic SHALL be rule-based (no API). The implementation SHALL be swappable to LLM API via config (`runtime.l4_update`).

#### Scenario: Track emotion across turns
- **WHEN** the character is provoked in turn 3 and emotion shifts to "微怒" at intensity 0.7
- **THEN** in turn 4 (no further provocation), emotion_intensity decays but "微怒" persists

#### Scenario: Session isolation
- **WHEN** user_A and user_B are chatting simultaneously
- **THEN** their L4 states are completely independent

### Requirement: Four-layer assembly pipeline
The system SHALL execute layers in order: Step A → L1 → L2 → L3 → L4 read → assemble. Each layer's output feeds the next as defined: L1 outputs active_domains → L2 uses as filter; L2 outputs tags_for_l3 → L3 uses as metadata filter. The assembler SHALL concatenate all layer outputs plus a meta-instruction into the final system_prompt.

#### Scenario: Full pipeline execution
- **WHEN** a user message arrives
- **THEN** Step A produces triggers, L1 selects modules and outputs active_domains, L2 matches rules using triggers+active_domains and outputs tags_for_l3, L3 retrieves scenes using query+tags_for_l3, L4 provides current state, assembler combines all

### Requirement: Token budget enforcement
The assembler SHALL enforce a configurable total token target (default ~2000 tokens). If the combined output exceeds the budget, the assembler SHALL trim L3 scenes first (reduce count), then L2 rules, while never trimming L1 core or L4 state.

#### Scenario: Trim when over budget
- **WHEN** L1(450) + L2(600) + L3(900) + L4(150) + meta(100) = 2200 exceeds the 2000 target
- **THEN** L3 drops one scene to bring total under 2000

### Requirement: Meta-instruction
The assembler SHALL append a meta-instruction block that tells the LLM to roleplay using the provided context. The meta-instruction SHALL specify: respond in character, do not explain the roleplay, follow behavioral rules, use scenes as style reference, and match the character's typical response length.

#### Scenario: Meta-instruction appended
- **WHEN** the system_prompt is assembled
- **THEN** it ends with a meta-instruction block (~100 tokens)
