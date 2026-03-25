## ADDED Requirements

### Requirement: LLM request interception
The plugin SHALL register an `@filter.on_llm_request()` handler that intercepts every LLM request before it reaches the provider. It SHALL extract the user message and recent conversation context from the event.

#### Scenario: Intercept incoming message
- **WHEN** a user sends a message through AstrBot
- **THEN** the on_llm_request handler fires before the LLM provider processes the request

### Requirement: RAG server communication
The plugin SHALL call the RAG server's `/retrieve` endpoint via HTTP POST with the user message, conversation context, and sender ID. The RAG server URL SHALL be configurable via plugin settings. The request timeout SHALL be configurable (default: 5000ms).

#### Scenario: Call RAG server
- **WHEN** a message is intercepted
- **THEN** the plugin sends POST to `{rag_server_url}/retrieve` with the message content

### Requirement: System prompt injection
The plugin SHALL replace `req.system_prompt` with the enhanced_system_prompt returned by the RAG server.

#### Scenario: Inject enhanced prompt
- **WHEN** the RAG server returns an enhanced_system_prompt
- **THEN** req.system_prompt is set to the returned value before the LLM processes the request

### Requirement: Direct query command
The plugin SHALL recognize a configurable command prefix (default: `/ask`) to route messages to the RAG server's `/query` endpoint instead of `/retrieve`. The query result SHALL be appended to the LLM prompt as reference context rather than replacing the system prompt.

#### Scenario: Direct query via command
- **WHEN** user sends "/ask 转移事件发生在第几卷"
- **THEN** the plugin calls /query, appends the result to the prompt context, and lets the LLM answer based on the retrieved information

### Requirement: Graceful fallback
If the RAG server is unreachable or returns an error, the plugin SHALL fall back to the original system prompt (no modification) and log the error. The user's message SHALL still be processed normally by AstrBot.

#### Scenario: RAG server timeout
- **WHEN** the RAG server does not respond within the configured timeout
- **THEN** the original system prompt is used unchanged and an error is logged

### Requirement: Plugin configuration
The plugin SHALL expose configuration via `_conf_schema.json`: rag_server_url (string), enabled (boolean), timeout_ms (integer), query_command_prefix (string, default "/ask").

#### Scenario: Disable plugin
- **WHEN** enabled=false in configuration
- **THEN** the on_llm_request handler does nothing, passing through unchanged
