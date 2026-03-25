## Context

AstrBot 部署在远程服务器，通过 NapCat 等平台接收消息，调用 LLM API（Gemini Pro）生成回复。本地机器（Intel Ultra 7 265KF / 64GB RAM / RTX 5070 Ti 16GB / Win11）资源充足，专门运行 RAG 服务。

小说原文为单个 TXT 文件（8.3MB, 116K 行, 24 卷 275 章），使用 `# 第X卷...` 作为章节头，`★ ★ ★` 作为场景分隔，`「」` 作为对话标记，存在 `(X/Y)` 分页标记需合并，以及 `铅笔小说`、`www.x23qb.com` 等需清理的元数据。

系统需要支持两种运行模式：人格增强模式（四层组装）和直接查询模式（纯 RAG 检索）。代码需可裁剪，即使人格增强模式最终被验证不可行，RAG 层仍然可独立使用。

## Goals / Non-Goals

**Goals:**
- 构建四层认知模型（L1 核心身份 / L2 行为规则 / L3 情景记忆 / L4 工作记忆），生成和调用方向相反
- RAG 全部本地 GPU 运行（embedding + reranker），不调 API
- LLM API 调用按阶段独立配置，生成时配置和运行时配置分离
- 每层独立可用、可测试、可裁剪
- 支持直接查询模式，绕过人格层做小说细节检索
- 运行时单次请求延迟 < 200ms（不含 LLM API），system_prompt 目标 ~2000 tokens

**Non-Goals:**
- 不在本地跑 LLM 推理（角色扮演由远端 AstrBot 调 API 完成）
- 不做多角色支持（当前只服务艾莉丝，但架构上预留扩展路径）
- 不做对话生成（只做 prompt 增强，生成由 AstrBot 侧 LLM 负责）
- 不做前端 UI

## Decisions

### 1. 四层架构与反向流

**决策**：生成顺序 L3→L2→L1（归纳），运行时 L1→L2→L3（演绎）。

**原因**：角色行为不能从抽象描述出发（手写 prompt 已验证失败），必须从具体原文场景归纳。运行时则反过来，从抽象到具体逐级缩小检索范围，每层输出作为下层的过滤条件。

**备选**：全部用向量检索（所有层都存向量库）→ 否决，因为 L1/L2 是离散的结构化知识，向量检索反而不精确。

### 2. 每层不同的存储和检索策略

| 层 | 存储 | 检索方式 | 理由 |
|---|---|---|---|
| L1 核心身份 | YAML 文件（6 个 domain 模块） | triggers → activate_on 交集匹配 | 数量固定(6个)，按域选择即可 |
| L2 行为规则 | YAML + 内存反向索引 | situation_tags 精确匹配 | 规则是离散类别(~80条)，要精确命中不要"相似" |
| L3 情景记忆 | ChromaDB 向量库 + metadata | 语义检索 + metadata filter + reranker | 场景(~1500条)需要模糊语义匹配 |
| L4 工作记忆 | 内存 dict | 直接读 key | 只是当前会话状态 |

**备选**：全部存向量库 → 否决。L2 的"受伤"规则需要被"你手怎么了"精确命中，向量检索可能返回语义相似但方向错误的规则。

### 3. 三级标签体系

```
L1 domains (6 个): core, vulnerability, combat, relationship, pride, growth
    │ 1:N 映射
    ▼
L2 situation_tags (~40 个): 受伤, 逞强, 被关心, 战斗, 嫉妒, ...
    │ 共享词表
    ▼
L3 scene metadata: 每个场景标注 situation_tags (从同一词表选)
```

L1 输出 `active_domains` 过滤 L2 的搜索范围，L2 输出 `tags_for_l3` 作为 L3 的 metadata filter。标签词表定义在 `tags_taxonomy.yaml`，全系统共享。

**决策**：标签词表约 40 个情境标签 + 6 个域标签，预处理时由 LLM 从预定义列表中选择（不自由生成），保证一致性。

### 4. 时间线处理：加权而非过滤

**决策**：模拟成长后的艾莉丝，但不排除早期场景。L3 检索时用 `period_weight` 加权排序，晚期场景优先但早期高相关场景仍可入选。按 period 使用不同模板呈现（行为参考 vs 记忆）。

**权重方案**：
- 回归后: 1.3 / 剑之圣地期: 1.1 / 魔大陆流浪期: 1.0 / 少女期: 0.85
- 当 `topic_is_past=true` 时翻转（少女期 1.3，回归后 0.85）

L2 规则以成长后行为为基准，通过 `growth_trace` 字段记录行为演化脉络，`motivation` 字段引用过去经历作为因果链。

### 5. 双模式运行

```
POST /retrieve  →  人格增强模式: Step A → L1 → L2 → L3 → L4 → 组装
POST /query     →  直接查询模式: 语义检索 L3 → 返回匹配场景文本
```

直接查询模式跳过 L1/L2/L4，只用 L3 向量检索，返回原文场景。这保证即使人格增强效果不佳，系统仍有独立价值——作为小说细节的知识库。

AstrBot 插件根据消息前缀或命令区分模式（如 `/ask` 触发直接查询，普通消息走人格增强）。

### 6. 按阶段独立 API 配置

```yaml
api:
  providers:
    gemini:
      api_key: "${GEMINI_API_KEY}"
      model: "gemini-2.5-pro-latest"
    claude:
      api_key: "${CLAUDE_API_KEY}"
      model: "claude-sonnet-4-20250514"

  # 预处理阶段配置（一次性运行）
  preprocess:
    map_extract: "gemini"        # MAP: 逐章场景提取
    reduce_vol_patterns: "gemini" # REDUCE 1: 卷级行为归纳
    reduce_global_rules: "gemini" # REDUCE 2: 全局规则合成
    reduce_core_modules: "gemini" # REDUCE 3: 核心身份浓缩

  # 运行时阶段配置（每次请求）
  runtime:
    step_a: "local"              # 意图分析: 默认 jieba 本地, 可切 API
    l4_update: "local"           # 状态更新: 默认规则引擎, 可切 API
```

每个阶段独立指定 provider，`"local"` 表示不调 API 用本地逻辑。预处理和运行时是两套完全独立的配置。

### 7. 小说解析策略

基于实际文件格式分析：

**分章**：按 `^# 第(\d+)卷` 正则分割，合并同一章的多个分页 `(X/Y)`。

**清洗**：跳过 `^_第[^_]+_$`（下划线标题）、`^铅笔小说$`、`^www\.x23qb\.com$`、`^(台版|扫图|录入|着:|译:).*`（元数据）、连续空行。

**场景分割**：以 `★ ★ ★` 为场景边界，每个场景作为 L3 的最小索引单元。

**卷/时期映射**：
- 卷1-3: 少女期
- 卷4-6: 魔大陆流浪期
- 卷7-15: 包含剑之圣地期等
- 卷16-24: 回归后/完结

具体映射需根据实际内容在 chunker 配置中定义。

### 8. 模块化代码架构

```
src/
├── config.py              # 配置加载，环境变量展开
├── models.py              # 所有 Pydantic 数据模型
├── server.py              # FastAPI 入口（薄层，只做路由）
│
├── preprocess/            # 预处理（可独立运行，不依赖 server）
│   ├── chunker.py         # 只负责分章清洗
│   ├── map_extract.py     # 只负责 MAP 阶段
│   ├── reduce_l2.py       # 只负责 L2 归纳
│   ├── reduce_l1.py       # 只负责 L1 浓缩
│   └── prompts/           # LLM prompt 模板（Markdown 文件）
│
├── layers/                # 运行时层（每层独立可测试）
│   ├── step_a.py          # jieba + 词典 → triggers
│   ├── l1_core.py         # 模块选择 → prompt_text + active_domains
│   ├── l2_behavior.py     # tag 匹配 → prompt_text + tags_for_l3
│   ├── l3_episodic.py     # 向量检索 → prompt_text
│   ├── l4_working.py      # 会话状态 → state_text
│   └── assembler.py       # 拼接所有层输出为 system_prompt
│
├── embedding/             # 本地模型（独立于业务逻辑）
│   ├── embed_model.py     # embedding 加载和推理
│   └── reranker.py        # reranker 加载和推理
│
└── llm/                   # LLM API 抽象（独立于业务逻辑）
    └── client.py          # 统一接口，按 provider name 分发
```

**裁剪场景**：
- 只要 L3：删除 layers/l1、l2、l4，assembler 直接用 l3 输出
- 只要 RAG 检索：删除整个 layers/，server.py 只保留 /query 端点
- 换 embedding 模型：只改 embedding/embed_model.py
- 换向量库：只改 l3_episodic.py 的存储层
- 加新角色：在 data/characters/ 下新建目录，配置切换即可

### 9. 技术选型

| 组件 | 选择 | 备选 | 选择理由 |
|---|---|---|---|
| Web 框架 | FastAPI | Flask | 原生异步、自带 OpenAPI 文档、Pydantic 集成 |
| 向量数据库 | ChromaDB | Milvus Lite, FAISS | 嵌入式零运维、Python 原生、支持 metadata filter |
| Embedding | BAAI/bge-large-zh-v1.5 | m3e-large, text2vec | 中文 MTEB 排名领先、1024 维、~1.5GB VRAM |
| Reranker | BAAI/bge-reranker-v2-m3 | bce-reranker | 多语言支持、与 BGE embedding 生态一致 |
| 分词 | jieba | pkuseg | 成熟稳定、无额外依赖、自定义词典简单 |
| LLM 调用 | httpx + 手写抽象 | LangChain, LiteLLM | 架构高度定制化，抽象层只需极简接口 |
| 配置 | YAML + 环境变量 | TOML, .env | 层级清晰、注释友好、Pydantic 可直接加载 |

**不用 LangChain/LlamaIndex 的原因**：本系统的四层漏斗检索、tag 匹配、period 加权等逻辑高度定制，框架的抽象反而是约束。手搓的代码量不大但完全可控。

## Risks / Trade-offs

**[MAP 提取质量不稳定]** → 先用 2-3 章试跑验证 prompt 效果，确认后再全量。提取 prompt 存为独立 Markdown 模板，方便迭代。

**[L2 行为规则覆盖不全]** → 运行时当 L2 匹配结果不足时，降级为纯 L3 检索 + L1 核心。系统不强依赖 L2，缺少时只是精度下降。

**[标签词表设计不当]** → 词表存独立 YAML 文件，可随时修改。L3 场景的 tags 在预处理时打标，改词表后需重跑 MAP（成本可控）。

**[AstrBot 网络不通]** → 插件增加 fallback：RAG 不可用时退回原始 prompt，不影响基本功能。

**[整体效果不达预期]** → 分层实施，Phase 1（L3 场景检索）完成后即可做初步评估。最差情况下保留 L3 作为小说知识库独立使用。

**[单文件小说格式边界情况]** → chunker 支持配置化的正则模式和卷-时期映射表，遇到新格式只改配置不改代码。
