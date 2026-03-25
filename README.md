# Eris RAG — 小说角色人格增强服务器

从小说原文构建四层认知模型（L1 核心身份 → L2 行为规则 → L3 场景记忆 → L4 会话状态），为聊天机器人提供精准的角色扮演增强。

## 架构

```
用户消息 → Step A (意图分析)
              │
              ├── L1: 核心模块选择 → L2: 行为规则匹配
              │
              └── L3: 场景检索（语义 + 关键词混合）
              │
              └── L4: 会话状态 + 潜意识记忆
              │
              ▼
         组装 system_prompt → 返回给聊天平台
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要 CUDA GPU（embedding + reranker 模型在 GPU 上运行）。

### 2. 配置

复制并编辑配置文件：

```bash
cp config.yaml.example config.yaml
```

必须配置：
- `api.providers.deepseek.api_key`：DeepSeek API key（用于构建和运行时意图分析）
- `character.novel_dir`：小说 TXT 文件所在目录
- `character.data_dir`：角色数据输出目录

### 3. 准备小说

将小说 TXT 文件放入 `character.novel_dir` 指定的目录（默认 `./data/novel/mushoku_tensei/`）。

要求：
- UTF-8 编码的 TXT 文件
- 章节以 `# 第X卷` 格式标记（chunker 按此分卷）
- 单文件或多文件均可

### 4. 构建人格树

```bash
python train.py
```

交互式引导，逐步确认。也可以全自动：

```bash
python train.py --all
```

构建流程：

| 步骤 | 说明 | 耗时 | 费用 |
|------|------|------|------|
| Step 1: chunk | 分块 + embed + 入库 | ~1 分钟 | 免费（本地 GPU） |
| Step 2: annotate | tag 标注 + 核心记忆标注 | ~5-10 分钟 | ~0.5 元 |
| Step 3: reduce_l2 | 生成行为规则 | ~10-20 分钟 | ~2 元 |
| Step 4: reduce_l1 | 生成核心模块 | ~2-5 分钟 | ~1 元 |

### 5. 配置用户关系

编辑 `data/user_prompts.yaml`：

```yaml
custom_instructions: |
  【场景适配】
  你现在在现代社会的群聊中...

users:
  鲁迪乌斯:
    sender_ids: ["QQ号1", "QQ号2"]
    aliases: ["鲁迪"]
    prompt: |
      对方是鲁迪，你的丈夫...
```

此文件支持热加载，改完保存即生效，不用重启服务器。

### 6. 启动服务器

```bash
python -m src.server
```

服务器运行在 `config.yaml` 中配置的 host:port（默认 `0.0.0.0:8787`）。

### 7. 测试

终端对话测试：
```bash
python chat.py
```

API 测试：
```bash
curl -X POST http://localhost:8787/retrieve \
  -H "Content-Type: application/json" \
  -d '{"user_message": "你好", "sender_id": "12345", "sender_nickname": "鲁迪"}'
```

健康检查：
```bash
curl http://localhost:8787/health
```

## 换小说 / 换角色

### 换小说同角色

1. 替换 `character.novel_dir` 下的 TXT 文件
2. 删除 `vectordb/` 目录（清空旧向量库）
3. 重新运行 `python train.py`

### 换角色

1. 修改 `config.yaml`：
   - `character.name`：角色名
   - `character.data_dir`：新角色数据目录
   - `character.target_period`：目标时期

2. 修改 `data/tags_taxonomy.yaml`：
   - `domains`：6 个性格域及其激活标签
   - `situations`：情境标签列表
   - `tag_queries`：每个标签的检索查询词（用于 Phase 2 标注）
   - `keyword_dict`：jieba 分词关键词映射（用于本地意图分析回退）
   - `volume_periods`：卷号到时期的映射
   - `period_weights`：各时期检索权重

3. 修改 `data/user_prompts.yaml`：
   - 角色关系定义
   - 自定义指令

4. 修改 `src/preprocess/prompts/` 下的 prompt 模板：
   - `annotate_tags.md`：标注 prompt（替换角色名）
   - `reduce_global_rules.md`：规则生成 prompt
   - `reduce_core_modules.md`：核心模块生成 prompt

5. 删除旧数据 + 重新构建：
```bash
rm -rf vectordb/
rm -rf data/characters/旧角色/
python train.py
```

## 项目结构

```
├── config.yaml              # 主配置（API key、模型、参数）
├── train.py                 # 人格树构建脚本
├── chat.py                  # 终端对话测试
├── run_pipeline.py          # 底层管线工具（搜索、重建等）
├── instruction.txt          # AstrBot 插件开发指南
│
├── src/
│   ├── server.py            # FastAPI 服务器
│   ├── config.py            # 配置管理
│   ├── models.py            # 数据模型
│   ├── llm/client.py        # LLM API 客户端（多 provider）
│   ├── embedding/
│   │   ├── embed_model.py   # BGE-large-zh embedding
│   │   └── reranker.py      # BGE-reranker 精排
│   ├── layers/
│   │   ├── step_a.py        # 意图分析（LLM/jieba）
│   │   ├── l1_core.py       # L1 核心身份模块选择
│   │   ├── l2_behavior.py   # L2 行为规则匹配
│   │   ├── l3_episodic.py   # L3 场景检索（语义+关键词混合）
│   │   ├── l4_working.py    # L4 会话状态
│   │   ├── subconscious.py  # 潜意识记忆系统
│   │   └── assembler.py     # system_prompt 组装
│   └── preprocess/
│       ├── chunker.py       # 小说分卷
│       ├── text_chunker.py  # 固定窗口分块
│       ├── annotator.py     # RAG + LLM 标注
│       ├── reduce_l2.py     # L2 规则生成
│       ├── reduce_l1.py     # L1 模块生成
│       └── prompts/         # LLM prompt 模板
│
├── data/
│   ├── tags_taxonomy.yaml   # 标签体系（热加载）
│   ├── user_prompts.yaml    # 用户关系配置（热加载）
│   ├── novel/               # 小说原文
│   ├── characters/          # 生成的角色数据（L1/L2）
│   └── memories/            # 潜意识记忆（per-user JSON）
│
└── vectordb/                # ChromaDB 持久化目录
```

## API

### POST /retrieve — 人格增强

请求：
```json
{
  "user_message": "你练剑练了多久了？",
  "conversation_context": ["你好", "哼，你谁啊"],
  "sender_id": "649535675",
  "sender_nickname": "鲁迪"
}
```

响应：
```json
{
  "enhanced_system_prompt": "## 核心身份\n...",
  "metadata": {
    "l1_modules_used": ["core", "combat"],
    "l2_rules_used": ["speech_style", "beh_combat_001"],
    "l3_scenes_used": ["v13_ch02_4570"],
    "total_tokens": 2456
  }
}
```

### POST /query — 直接查询

### GET /health — 健康检查

## 热加载

以下文件修改后立即生效，无需重启服务器：
- `data/user_prompts.yaml`（用户关系、自定义指令）
- `data/tags_taxonomy.yaml`（标签词典、时期权重）

## 硬件要求

- GPU：4GB+ VRAM（embedding + reranker 约 3GB）
- RAM：8GB+
- 磁盘：2GB+（模型缓存 + 向量库）

## 技术栈

| 组件 | 选择 |
|------|------|
| Web 框架 | FastAPI |
| 向量数据库 | ChromaDB |
| Embedding | BAAI/bge-large-zh-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 分词 | jieba |
| LLM | DeepSeek V3/Reasoner（可配置） |
