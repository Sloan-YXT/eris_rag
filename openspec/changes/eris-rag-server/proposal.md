## Why

使用 AstrBot + LLM API 模拟无职转生的艾莉丝·格雷拉特，手写人格 prompt 两个月效果很差——LLM 只能输出泛化的"傲娇模板"而非这个具体角色。根本原因是手写 prompt 是 zero-shot 标签描述，缺乏原文场景示范和行为因果链，LLM 无法从中还原角色的行为分寸和语感。

需要构建本地 RAG 人格增强服务器，从整本小说中提取结构化的角色认知模型（L1 核心身份 / L2 行为规则 / L3 情景记忆 / L4 工作记忆），运行时为每次对话动态组装精准的 ~2000 token 上下文，以 few-shot 示范 + 因果链驱动 LLM 角色扮演。

## What Changes

- **新增本地 RAG 服务器**：FastAPI 服务，接收用户消息，返回增强后的 system_prompt
- **新增预处理流水线**：从小说原文逐章提取场景（MAP）→ 归纳行为规则（REDUCE L2）→ 浓缩核心身份（REDUCE L1），生成顺序自底向上
- **新增四层运行时检索**：L1→L2→L3 逐级漏斗，上层输出作为下层过滤条件，调用顺序自顶向下
- **新增 AstrBot 插件**：Star 框架插件，在 `on_llm_request` 钩子中调用 RAG 服务器，注入增强 prompt
- **新增直接查询模式**：不经过人格增强，直接用 RAG 检索小说细节并返回，支持纯信息同步场景
- **RAG 全部本地运行**：embedding（BGE-large-zh）和 reranker（BGE-reranker）在本机 GPU 运行，不调 API
- **LLM API 按阶段独立配置**：预处理的每个阶段（MAP 提取、REDUCE L2、REDUCE L1）和运行时的每个阶段（Step A 意图分析、L4 状态更新等）各自独立配置使用哪个 LLM provider，生成时和调用时是两套配置
- **模块化可裁剪架构**：每层独立可用，即使人格增强模式被验证不可行，L3 场景检索仍然可以独立作为小说细节查询 RAG 使用

## Capabilities

### New Capabilities
- `novel-ingestion`: 小说文本导入、清洗、分章、场景提取（MAP 阶段）
- `scene-retrieval`: L3 场景向量检索 + reranker 精排，支持 metadata 过滤和 period 加权，独立可用
- `behavior-rules`: L2 行为规则生成（REDUCE）和运行时 tag 匹配
- `core-identity`: L1 核心身份模块生成（REDUCE）和运行时动态选择
- `personality-assembly`: 四层组装器，将 L1/L2/L3/L4 输出拼接为最终 system_prompt
- `direct-query`: 直接查询模式，绕过人格层，用 RAG 检索小说细节加入 prompt
- `rag-server`: FastAPI 服务器，暴露 /retrieve（人格增强）和 /query（直接查询）端点
- `astrbot-plugin`: AstrBot Star 框架插件，桥接 AstrBot 和 RAG 服务器

### Modified Capabilities
（无已有 spec 需修改）

## Impact

- **新增 Python 项目**：src/ 下全部为新代码，分 preprocess/layers/embedding/llm 四个子包
- **本地 GPU 资源**：embedding + reranker 约占 3-4 GB VRAM（RTX 5070 Ti 16GB 余量充足）
- **外部依赖**：FastAPI, ChromaDB, sentence-transformers, FlagEmbedding, jieba, httpx
- **网络**：RAG 服务器需对 AstrBot 所在远程服务器暴露 8787 端口
- **LLM API**：预处理阶段消耗 API 调用（一次性），运行时可选消耗（Step A / L4 如配置为 API 模式）
- **数据**：小说原文已就绪（Wu Zhi Zhuan Sheng/，单文件 8.3MB，24 卷 275 章）
