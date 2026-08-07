# Company-RAG

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/fastapi-0.141-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/llm-DeepSeek--chat-purple.svg" alt="LLM">
  <img src="https://img.shields.io/badge/embedding-BGE--large--zh-orange.svg" alt="Embedding">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

<p align="center">
  <strong>企业级 RAG 知识库问答系统</strong><br>
  零 GPU · 全 API 调用 · 邮箱验证码登录 · 对话持久化 · SPA 前端
</p>

---

## RAG 技术架构设计

### 整体架构

本系统采用经典的 **检索增强生成（Retrieval-Augmented Generation）** 架构，分为**离线索引**和**在线推理**两条链路：

```
                          ┌── 离线索引链路 ──┐
                          │                  │
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐
│  .docx   │    │ 文档解析   │    │  Recursive   │    │ 硅基流动   │
│  .pdf    │───▶│ python-   │───▶│  Character   │───▶│ BGE-large │
│  .doc    │    │  docx     │    │  TextSplitter│    │ -zh-v1.5  │
│ (MIME)   │    │  PyPDF2   │    │  chunk=400   │    │ dim=1024  │
└──────────┘    └──────────┘    └──────────────┘    └─────┬─────┘
                                                          │
                                                  ┌───────▼───────┐
                                                  │   ChromaDB    │
                                                  │ HNSW 索引     │
                                                  │ cosine 距离   │
                                                  └───────────────┘

                          ┌── 在线推理链路 ──┐
                          │                  │
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌───────────┐
│ 用户问题   │───▶│ 向量化     │───▶│ Top-5 相似   │───▶│ Context   │───▶│ 流式返回   │
│          │    │ (同模型)  │    │ 检索         │    │ + Prompt  │    │ + 来源    │
└──────────┘    └──────────┘    └──────────────┘    └───────────┘    └───────────┘
                                                          │
                                                  ┌───────▼───────┐
                                                  │  DeepSeek     │
                                                  │  deepseek-chat│
                                                  │  temp=0.1     │
                                                  └───────────────┘
```

| 层级 | 职责 | 技术选型 | 设计决策 |
|------|------|----------|----------|
| **文档解析** | 提取原始文本 | python-docx、PyPDF2、email+BeautifulSoup | 保留表格结构（行拼接）、MIME 解析兼容旧版 .doc |
| **Chunk 切分** | 文本分段 | LangChain `RecursiveCharacterTextSplitter` | chunk_size=400、overlap=60、中文分隔符优先 |
| **向量化** | 文本 → 稠密向量 | 硅基流动 `BAAI/bge-large-zh-v1.5` | 1024 维、中文优化、API 调用零 GPU |
| **向量存储** | 向量索引 | ChromaDB (HNSW) | 本地持久化、cosine 相似度、内存高效 |
| **检索** | Top-K 召回 | 向量相似度检索 `k=5` | 无 Reranker、纯向量相似度排序 |
| **生成** | 答案合成 | DeepSeek `deepseek-chat` | temp=0.1 低随机性、max_tokens=2000、System Prompt 约束 |
| **评估** | 质量度量 | LLM-as-Judge | Hit Rate@5 + MRR@5 + Faithfulness + Relevancy |

### Chunk 设计

#### 分块策略

采用 `RecursiveCharacterTextSplitter`，按分隔符优先级递归切分，确保语义完整性：

```python
RecursiveCharacterTextSplitter(
    chunk_size=400,        # 每个 chunk 最大字符数
    chunk_overlap=60,      # 相邻 chunk 重叠 60 字符
    separators=[
        "\n\n",            # 优先级 1: 段落边界
        "\n",              # 优先级 2: 行边界
        "。",              # 优先级 3: 中文句号
        "；",              # 优先级 4: 中文分号
        "，",              # 优先级 5: 中文逗号
        " ",               # 优先级 6: 空格
        ""                 # 优先级 7: 字符级兜底
    ],
)
```

#### 参数设计理由

| 参数 | 值 | 设计理由 |
|------|-----|----------|
| `chunk_size` | 400 | 中文语义完整单元通常 200-500 字；BGE-large-zh 最佳上下文窗口 512 tokens；过大会稀释关键信息，过小会导致碎片化 |
| `chunk_overlap` | 60 | 15% 重叠率，防止关键句被切断在 chunk 边界，确保跨 chunk 的实体和指标名能被检索到 |
| 分隔符优先级 | 段落 → 句子 → 字符 | 优先在自然语义边界切分，保证每个 chunk 是一段完整、可独立理解的信息 |

#### chunk 元数据

每个 chunk 携带完整的溯源信息，用于前端展示和评估追踪：

```python
{
    "text": "会员品单价是在数据范围和时间范围内的会员类订单统计的品单价...",
    "metadata": {
        "source": "./wiki_docs/客单价.pdf",   # 源文件路径
        "title": "客单价",                      # 文档标题（用户可见）
        "chunk_index": 3,                       # 在文档中的位置编号
        "total_chunks": 24,                     # 该文档总 chunk 数
        "file_size": 256789,                    # 源文件大小
        "file_type": "pdf"                      # 文件类型
    }
}
```

### 检索设计

#### 检索策略

当前采用**单阶段向量检索**，检索流程如下：

```
用户问题 "客单价怎么计算？"
       │
       ▼
┌─────────────────────────┐
│ 1. Query Embedding      │  BGE-large-zh-v1.5 将问题转为 1024 维向量
│    ──────────────────   │
│ 2. ChromaDB.similarity_ │  HNSW 索引 + cosine 距离，召回 Top-5
│    search(query_vec, 5) │
│    ──────────────────   │
│ 3. 排序依据: cosine     │  相似度从高到低返回
│    相似度降序           │
└─────────────────────────┘
       │
       ▼
  Top-5 文档片段 → Prompt 拼接 → DeepSeek 生成
```

#### 为什么不加 Reranker？

| 方案 | 优势 | 劣势 | 当前选择 |
|------|------|------|----------|
| 纯向量检索 | 延迟低（~200ms）、无额外 LLM 调用成本 | 对同义词/缩写不敏感 | ✅ 当前方案 |
| + Cross-Encoder Reranker | 提升 Top-3 精度 5-15% | 需额外模型（GPU 或 API）、延迟 +300-800ms | 🔜 下一阶段 |

在知识库规模 < 100 篇、领域术语集中（零售数据分析）的场景下，BGE-large-zh 的向量相似度检索已能达到 Hit Rate@5 = 100%。当知识库扩展到 500+ 文档、跨多个业务领域时，建议引入 Reranker。

#### 检索数量选择

- **k=5**：经验值，平衡上下文长度和召回覆盖率
- 5 个 chunk × 400 字 ≈ 2000 字上下文 → DeepSeek 16K 上下文窗口占比仅 12.5%，留充足生成空间
- chunk_overlap 保证相邻 chunk 间冗余覆盖，减少检索遗漏

### 生成设计

#### System Prompt 约束

```python
RAG_SYSTEM = """你是一个公司内部知识助手。请根据以下检索到的文档片段，回答用户的问题。

## 规则
1. 只根据提供的文档片段回答，不要编造信息
2. 如果文档中没有相关内容，明确告知"根据现有文档无法回答该问题"
3. 回答简洁、准确、条理清晰
4. 涉及具体数字、日期、金额时，务必引用原文
5. 回答末尾标注信息来源"""
```

#### LLM 参数

| 参数 | 值 | 设计理由 |
|------|-----|----------|
| `model` | deepseek-chat | 中文能力强、16K 上下文、价格低（¥1/百万 token） |
| `temperature` | 0.1 | 知识问答场景需要确定性输出，避免幻觉和随机改写 |
| `max_tokens` | 2000 | 覆盖复杂多维度问题的详细回答，同时控制延迟 |

#### Context 拼接格式

检索到的文档片段按以下模板拼接后注入 Prompt：

```
## 检索到的文档片段
[片段1 | 来源：客单价]
会员品单价是在数据范围和时间范围内的会员类订单统计的品单价...

[片段2 | 来源：客单价]
非会员品单价是非会员类订单统计的品单价。会员同比差值 = 本期...

[片段3 | 来源：核心指标概览]
近效期占比环比差值>0或为空时，红色字体显示上升...

## 用户问题
客单价怎么计算？

## 回答
```

#### Fallback 策略

当知识库为空（ChromaDB collection 无数据）时，自动切换为通用知识模式：

```python
FALLBACK_SYSTEM = """⚠️ 当前知识库中没有已上传的文档，以下回答基于通用知识，仅供参考。"""
```

### Embedding 设计

| 维度 | 选择 | 说明 |
|------|------|------|
| **模型** | BAAI/bge-large-zh-v1.5 | C-MTEB 中文榜单 Top-3，专为中文检索优化 |
| **向量维度** | 1024 | 高维向量提供更强的语义区分能力 |
| **API 服务** | 硅基流动 (siliconflow.cn) | 国内低延迟（~50ms/chunk），免费额度充足 |
| **批处理** | chunk_size=20 | 一次请求 embedding 20 个 chunk，平衡速度和 API 配额 |
| **归一化** | 输出归一化向量 | 配合 cosine 相似度，下游可直接用内积计算距离 |
| **tiktoken** | 关闭 | BGE 模型不属于 OpenAI tiktoken 体系，关闭避免错误 |

### 文档解析设计

支持三种文件格式，各自采用针对性解析策略：

| 格式 | 解析器 | 特殊处理 |
|------|--------|----------|
| `.docx` | python-docx | 正文段落 + 表格逐行拼接（`cell1 \| cell2 \| cell3`），保留业务数据结构 |
| `.pdf` | PyPDF2 | 逐页提取、空页过滤、页间双换行拼接保持段落结构 |
| `.doc` (MIME) | email + BeautifulSoup | 旧版 Word 的 MIME HTML 封装，解析 `text/html` 部分，去除 script/style 标签后提取纯文本 |

### 评估体系

系统内置 RAG 评估框架，从**检索质量**和**生成质量**两个维度量化系统表现：

```
┌────────────────────────────────────────────────────────┐
│                   RAG 评估体系                          │
│                                                        │
│  ┌── 检索质量 ──┐      ┌── 生成质量 ──┐               │
│  │              │      │              │                │
│  │ Hit Rate@5  │      │ Faithfulness │               │
│  │ (检索命中率) │      │ (忠实度)     │  LLM-as-Judge │
│  │              │      │              │                │
│  │ MRR@5       │      │ Answer       │               │
│  │ (平均倒数排名)│      │ Relevancy   │  LLM-as-Judge │
│  │              │      │ (回答相关性)  │               │
│  └──────────────┘      └──────────────┘               │
│                                                        │
│  运行: python evaluate.py                              │
│  测试集: test_set.json (15 个业务场景问答对)            │
└────────────────────────────────────────────────────────┘
```

#### 指标详解

| 指标 | 计算方式 | 评估对象 | 当前得分 | 目标 |
|------|----------|----------|---------|------|
| **Hit Rate@5** | `hit_count / total`，正确答案文档是否出现在 Top-5 检索结果中 | 检索召回 | 100% | ≥90% |
| **MRR@5** | `Σ(1/rank) / total`，正确答案在检索结果中的平均倒数排名 | 检索排序 | 0.836 | ≥0.8 |
| **Faithfulness** | DeepSeek 作为 Judge 判定回答是否每条事实都能在检索上下文找到依据，0/1 二值评分 | 幻觉控制 | 0.87 | ≥0.85 |
| **Answer Relevancy** | DeepSeek 作为 Judge 判定回答是否直接完整地回答了问题，0/0.5/1 三档评分 | 回答质量 | 0.83 | ≥0.8 |

#### LLM-as-Judge 评估 Prompt

Faithfulness 判定逻辑：
```python
# 评分标准
# 1 分：回答中每条事实陈述都能在上下文片段中找到依据
# 0 分：回答中包含任何凭空编造、与上下文矛盾或无法验证的信息
# 边界情况：上下文不相关或为空，但回答声明"无法回答"，也得 1 分
```

### 数据流全链路

一条用户提问的完整处理链路：

```
1. 用户发送 "客单价怎么计算？"
         │
2. 认证中间件验证 session_token
         │
3. 问题写入 MySQL messages 表 (role=user)
         │
4. 问题向量化 → BGE-large-zh-v1.5 → 1024-dim vector
         │
5. ChromaDB HNSW 索引 → cosine 相似度 → Top-5 chunks
         │
6. Context 拼接 + System Prompt → DeepSeek deepseek-chat
         │
7. LLM 生成回答 + 来源溯源信息
         │
8. 回答 (含 HTML 来源折叠框) 写入 MySQL messages 表 (role=assistant)
         │
9. JSON 响应返回前端 → SPA 渲染消息气泡
```

---

## 功能特性

- **零 GPU 依赖** — 全 API 调用（硅基流动 Embedding + DeepSeek LLM），普通 CPU 服务器即可运行
- **中文深度优化** — BGE-large-zh-v1.5（1024 维）向量模型 + DeepSeek-chat 中文大模型
- **多格式文档** — `.docx`（含表格）、`.pdf`、旧版 `.doc`（MIME HTML）
- **邮箱验证码登录** — SMTP 真实邮件发送，24 小时 Session 免重复登录
- **对话历史管理** — 完整 CRUD，MySQL 持久化，Navicat 可直接查询
- **ChatGPT 风格 SPA** — 原生 HTML/CSS/JS 单页应用，响应式布局
- **来源可追溯** — 每轮回答附带 Top-5 检索片段，HTML 折叠展开
- **内置评估体系** — 4 项指标，LLM-as-Judge 自动评分
- **RESTful API** — FastAPI + Swagger 文档（`/docs`）

---

## 快速开始

### 环境要求

- Python 3.11+
- MySQL 8.0+
- [硅基流动 API Key](https://siliconflow.cn)
- [DeepSeek API Key](https://platform.deepseek.com)
- SMTP 邮箱（如 163，需开启 SMTP 授权码）

### 1. 安装依赖

```bash
conda create -n RAG python=3.11 -y
conda activate RAG
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

关键配置项：

| 变量 | 说明 | 示例 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | `sk-xxx` |
| `SILICONFLOW_API_KEY` | 硅基流动 API 密钥 | `sk-xxx` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-large-zh-v1.5` |
| `LLM_MODEL` | 生成模型 | `deepseek-chat` |
| `CHUNK_SIZE` | 分块大小 | `400` |
| `CHUNK_OVERLAP` | 分块重叠 | `60` |
| `DB_PASS` | MySQL 密码 | |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` | SMTP 配置 | `smtp.163.com:465` |

### 3. 初始化数据库

首次启动自动建表，也可手动创建：

```sql
CREATE DATABASE IF NOT EXISTS company_rag
  DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. 构建知识库

将文档放入 `wiki_docs/` 目录，执行离线索引入库：

```bash
python build_index.py
```

### 5. 启动服务

```bash
python app.py
```

| 地址 | 说明 |
|------|------|
| `http://localhost:7860` | 导航页 |
| `http://localhost:7860/login` | 邮箱验证码登录 |
| `http://localhost:7860/chat` | SPA 聊天界面 |
| `http://localhost:7860/docs` | Swagger API 文档 |

---

## API 接口

所有 `/api/*` 和 `/chat` 接口需要 `session_token` Cookie（登录后自动下发）。

### POST /login/send-code — 发送验证码

```bash
curl -X POST http://localhost:7860/login/send-code \
  -H "Content-Type: application/json" \
  -d '{"email": "user@company.com"}'
# {"ok": true}
```

### POST /login/verify — 验证码登录

```bash
curl -X POST http://localhost:7860/login/verify \
  -H "Content-Type: application/json" \
  -d '{"email": "user@company.com", "code": "123456"}'
# Set-Cookie: session_token=xxx; HttpOnly; Path=/
# {"ok": true}
```

### POST /api/chat — RAG 问答

```bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -b "session_token=YOUR_TOKEN" \
  -d '{"question": "客单价怎么计算？"}'
```

响应：

```json
{
  "conversation_id": 1,
  "answer": "客单价 = 销售金额 / 来客数。会员品单价是...",
  "sources": [
    {"title": "客单价", "content": "...", "chunk_index": 3}
  ],
  "has_kb": true,
  "latency_ms": 1500,
  "tokens_used": 1200
}
```

`conversation_id` 为 `null` 时自动以问题前 30 字为标题创建新对话。

### 对话管理

| 方法 | 接口 | 说明 |
|------|------|------|
| `GET` | `/api/me` | 当前用户信息 |
| `GET` | `/api/conversations` | 列出所有对话 |
| `POST` | `/api/conversations` | 创建新对话 |
| `DELETE` | `/api/conversations/{id}` | 删除对话 |
| `GET` | `/api/conversations/{id}/messages` | 获取对话消息 |

---

## 效果评估

```bash
python evaluate.py
```

| 指标 | 当前得分 | 说明 |
|------|---------|------|
| Hit Rate@5 | 100% | 正确文档进入 Top-5 的概率 |
| MRR@5 | 0.836 | 正确文档的倒数排名均值 |
| Faithfulness | 0.87 | LLM 判定：回答是否忠于检索上下文 |
| Answer Relevancy | 0.83 | LLM 判定：回答是否紧扣问题 |

编辑 `test_set.json` 扩充测试集可提升评估覆盖面。当前包含 15 个业务场景问答对，覆盖客单价、核心指标、毛利率、库存周转/积压/预警、分类等模块。

---

## 项目结构

```
company-rag/
├── app.py                  # FastAPI 主服务：认证中间件 + API 路由
├── auth.py                 # 邮箱验证码发送 + Session 生命周期管理
├── database.py             # MySQL 连接池 + 5 张表自动建表
├── build_index.py          # 离线索引构建入口（解析 → 分块 → 向量化 → 入库）
├── document_loader.py      # 文档解析：docx(含表格)/pdf/MIME HTML
├── chunker.py              # RecursiveCharacterTextSplitter 分块策略
├── embeddings.py           # 硅基流动 BGE-large-zh-v1.5 Embedding 配置
├── vector_store.py         # ChromaDB 写入 + 检索封装
├── rag_chain.py            # RAG 核心链路：检索 → Prompt 拼接 → 生成
├── evaluate.py             # 评估框架：检索指标 + LLM-as-Judge 生成指标
├── test_set.json           # 15 题评估测试集
├── requirements.txt
├── .env.example
├── .gitignore
├── static/
│   └── chat.html           # ChatGPT 风格 SPA 前端（原生 HTML/CSS/JS）
└── wiki_docs/              # 知识库源文档（gitignore）
    ├── 客单价.pdf
    ├── 核心指标概览.pdf
    ├── 实际毛利率.docx
    ├── 库存周转.docx
    ├── ...（14 篇零售数据分析文档）
```

---

## 数据库设计

MySQL `company_rag` 库，`utf8mb4` + `InnoDB`：

| 表名 | 核心字段 | 说明 |
|------|----------|------|
| `users` | id, email (UNIQUE), created_at | 注册用户 |
| `verification_codes` | email, code, expires_at, used | 限时 5 分钟、一次性 |
| `sessions` | token (UNIQUE), user_id, email, expires_at | 24 小时有效、httponly Cookie |
| `conversations` | id, user_id (FK), title, created_at | 用户对话分组 |
| `messages` | id, conversation_id (FK CASCADE), role, content, created_at | user/assistant 消息 |

级联删除：删除对话时自动清除关联消息。

---

## 开源协议

MIT
