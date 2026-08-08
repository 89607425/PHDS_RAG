# PHDS-RAG

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/fastapi-0.141-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/llm-DeepSeek--chat-purple.svg" alt="LLM">
  <img src="https://img.shields.io/badge/embedding-BGE--large--zh-orange.svg" alt="Embedding">
  <img src="https://img.shields.io/badge/reranker-BGE--reranker--v2--m3-yellow.svg" alt="Reranker">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

<p align="center">
  <strong>业务 Wiki RAG 智能问答系统</strong><br>
  零 GPU成本 · 混合检索 · 多轮记忆 · 自我检查 · 知识库管理
</p>

---

## RAG 技术架构设计

### 整体架构

本系统采用**混合检索 + 重排序（Hybrid Retrieval + Reranker）** 的增强型 RAG 架构，分为**离线索引**和**在线推理**两条链路：

![整体架构](P1.png)


| 层级 | 职责 | 技术选型 | 设计决策 |
|------|------|----------|----------|
| **文档解析** | 提取原始文本 | python-docx、PyMuPDF (fitz)、email+BeautifulSoup | PyMuPDF 替代 PyPDF2 提升中文 PDF 解析质量；保留表格结构 |
| **Chunk 切分** | 文本分段 | LangChain `RecursiveCharacterTextSplitter` | chunk_size=400、overlap=60、中文分隔符优先 |
| **向量化** | 文本 → 稠密向量 | 硅基流动 `BAAI/bge-large-zh-v1.5` | 1024 维、中文优化、API 调用零 GPU |
| **向量存储** | 向量索引 + BM25 | ChromaDB (HNSW) + rank-bm25 | 本地持久化、cosine 相似度、BM25 关键词互补 |
| **混合检索** | 双路召回 + RRF 融合 | BM25 + 向量检索 → Reciprocal Rank Fusion | 语义匹配与关键词匹配互补，提升专业术语召回 |
| **重排序** | Cross-Encoder 精排 | 硅基流动 `BAAI/bge-reranker-v2-m3` | 逐对计算 query-doc 语义相关性，Top-20 → Top-5 |
| **Query 改写** | 查询优化 | DeepSeek deepseek-chat | 补全缩写、口语化→正式术语，提升检索命中率 |
| **多轮记忆** | 对话上下文 + 摘要 | DeepSeek deepseek-chat | 摘要 + 最近 6 条消息联合压缩；超 8 条自动触发摘要合并 |
| **生成** | 答案合成 | DeepSeek `deepseek-chat` | temp=0.1、max_tokens=2000、流式输出 + SSE |
| **Self-Check** | 回答准入机制 | DeepSeek deepseek-chat (LLM-as-Judge) | 逐 claim 核验(supported/unsupported/contradicted/inference) → safe-rewrite → 拒答 |

### Chunk 设计

#### 分块策略

采用 `RecursiveCharacterTextSplitter`，按分隔符优先级递归切分：

```python
RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)
```

#### Chunk 元数据

```python
{
    "text": "会员品单价 = 会员类订单的销售金额 / 来客数...",
    "metadata": {
        "source": "./wiki_docs/客单价.pdf",
        "title": "客单价",
        "chunk_index": 3,
        "total_chunks": 24,
    }
}
```

### 混合检索设计

#### 检索流程

![检索流程](P2.png)

#### RRF 融合原理

```
RRF_score(d) = Σ 1 / (k + rank_r(d) + 1)  for each r ∈ R
```

- $R$：所有检索结果列表的集合
- $\text{rank}_r(d)$：文档 $d$ 在结果列表 $r$ 中的排名（从 0 开始）
- $k = 60$：平滑参数，防止极端排名差异

RRF 的优势：
- 无需归一化两路检索的分数（BM25 分数和 cosine 距离量纲完全不同）
- 多路共同命中自动获得更高权重
- 对任意路数均可扩展

### Reranker 设计

| 维度 | 选择 | 说明 |
|------|------|------|
| **模型** | BAAI/bge-reranker-v2-m3 | MTEB 榜单 Top 级 Cross-Encoder，多语言多粒度 |
| **API** | 硅基流动 `/v1/rerank` | 调用简单，延迟 ~200-400ms |
| **输入** | (query, document) 对 | 每对独立计算语义相关性 |
| **输出** | relevance_score (0-1) | 逐对打分，降序排列 |
| **Top-N** | 5 | 从 ~30 个候选文档中精选最优 5 个 |

为什么需要 Reranker？
- 向量检索和 BM25 都是**双塔模型**：query 和 doc 独立嵌入后比较，丢失了精细的交互信息
- Reranker 是 **Cross-Encoder**：将 query 和 doc 拼接后过 transformer，捕捉 token 级别的交互
- 初排（Top-20→RRF）保证召回率，精排（Reranker Top-5）保证准确率

### Query 改写设计

```
原始问题: "卡片区有哪些内容？"
     │
     ▼ (LLM)
改写结果: "卡片区包含哪些指标项及其定义？"
```

```python
QUERY_REWRITE_PROMPT = """你是一个查询优化助手。将用户的模糊问题改写为更精确的检索查询。

规则：
1. 补全缩写和专业术语（如"环比"→"环比率/环比差值"）
2. 将口语化表达转为正式术语
3. 如果问题本身已经很精确，直接返回原问题
4. 只输出改写后的问题，不要任何解释"""
```

改写规则：
- **补全缩写**：`"环比"` → `"环比率/环比差值"`
- **术语正规化**：`"客单价怎么算"` → `"客单价的计算公式是什么"`
- **消除歧义**：`"那个指标的排名"` → `"库存周转指标的门店排名规则"`
- 改写前后的相似性判断：若改写结果与原问题几乎相同，保留原问题以避免过度改写

### 多轮记忆设计

#### Condense（即时压缩）

```
历史对话:
  用户: 客单价有哪些类型？
  助手: 客单价包含会员品单价、非会员品单价、整体品单价...
  用户: 它们的环比怎么看？
  
   ↓ （摘要: "用户关注品单价类型及环比计算"）

独立问题: "会员品单价、非会员品单价、整体品单价的环比计算方法是什么？"
```

```python
CONDENSE_PROMPT = """根据对话摘要和历史，将用户的最新问题改写为独立问题。"""
```

- 取「摘要 + 最近 6 条未压缩消息」作为压缩上下文
- 将代词替换为具体实体，省略条件从历史中补全

#### 持久化摘要（长对话记忆）

```
超长对话 (>8 条未摘要消息)
  │
  ▼ (LLM)
【自动触发摘要合并】旧摘要 + 新消息 → 新摘要
  │
  ▼
存入 MySQL conversations 表（summary | summarized_until_message_id）
```

- 阈值：累计 8 条未摘要消息自动触发
- 摘要内容：业务对象、公式/数字、筛选条件、未解决问题
- 永不删除原始消息，摘要只用于 condense 上下文窗口控制

### Self-Check 回答准入机制

#### 设计理念

Self-Check 不是「评分」，而是**回答准入机制**：任何回答必须在逐条核验通过后才允许返回。流式接口采用「缓冲生成→核验→流式」策略，确保用户不会看到未经核验的内容。

#### 核验流程

```
生成回答（要求每条事实带 [来源：文档名-片段N] 标记）
  │
  ▼
LLM 逐 claim 核验（输出结构化 JSON）
  │
  ├─ 全部 supported/inference ──→ final_action: pass → 直接返回
  │
  ├─ 存在 unsupported/contradicted ──→ final_action: rewrite
  │     │                                    │
  │     ├─ 安全改写成功 ──→ 返回修正后回答
  │     └─ 改写失败 ──→ final_action: refuse → 拒答
  │
  └─ JSON 解析失败 / LLM 调用失败 ──→ final_action: refuse → 拒答
```

#### 逐条核验格式

```python
SELF_CHECK_PROMPT = """逐条核查 AI 回答中的每一项关键事实声明。

每条 claim 判定：
- supported：claim 在对应片段中有明确原文
- unsupported：claim 在检索上下文中找不到任何依据
- contradicted：claim 与上下文信息冲突
- inference：上下文信息的合理推断

输出严格 JSON。"""
```

```json
{
  "claims": [
    {"claim": "客单价 = 销售额 ÷ 来客数", "verdict": "supported", "source_fragment": "分类-片段1"},
    {"claim": "2020年客单价增长了15%", "verdict": "unsupported", "source_fragment": null}
  ],
  "overall_score": 0.5,
  "final_action": "rewrite"
}
```

#### 保守失败策略

- JSON 解析失败 → 直接拒答（不赋默认分）
- LLM API 调用失败 → 直接拒答
- 安全改写失败 → 拒答
- 安全改写返回内容过短（< 10 字符）→ 拒答

#### 流式安全策略

流式接口（`/api/chat/stream`）**不使用流式 LLM 生成**，而是：
1. 用非流式 LLM 生成完整回答
2. 运行 Self-Check 核验管线
3. 将**核验通过**的回答以 2 字符/块的粒度流式发送

这牺牲了首 token 延迟（TTFB），但确保用户看到的每一帧都是经过核验的安全内容。

### Embedding 设计

| 维度 | 选择 | 说明 |
|------|------|------|
| **模型** | BAAI/bge-large-zh-v1.5 | C-MTEB 中文榜单 Top-3 |
| **向量维度** | 1024 | 高维向量强语义区分能力 |
| **API** | 硅基流动 | 国内低延迟（~50ms/chunk） |
| **批处理** | chunk_size=20 | 20 个 chunk 一批 embedding |
| **归一化** | 已归一化 | 配合 cosine 相似度 |

### 文档解析设计

| 格式 | 解析器 | 特殊处理 |
|------|--------|----------|
| `.docx` | python-docx | 正文段落 + 表格逐行拼接（`cell1 \| cell2 \| cell3`），保留业务数据结构 |
| `.pdf` | **PyMuPDF (fitz)** | 逐页提取、空页过滤、页间双换行拼接。替换 PyPDF2 以提升中文 PDF 解析质量 |
| `.doc` (MIME) | email + BeautifulSoup | 旧版 Word MIME HTML 封装，解析 `text/html` 部分后提取纯文本 |


---

## 功能特性

- **零 GPU 依赖** — 全 API 调用（硅基流动 Embedding + Reranker + DeepSeek LLM），普通 CPU 服务器即可
- **混合检索** — BM25 + 向量双路召回 → RRF 融合 → Cross-Encoder 精排
- **Query 改写** — LLM 自动将模糊口语化问题转为精确检索查询
- **多轮 Condense** — 对话上下文压缩，支持多轮追问
- **多轮记忆** — 持久化会话摘要，超 8 条消息自动合并，长对话不丢上下文
- **Self-Check** — 逐 claim 核验（支持/不支持/矛盾/推断）→ 安全改写 → 拒答三级准入
- **流式输出** — SSE 协议，缓冲核验后逐块输出（2 字符/块），核验不通过自动拒答
- **知识库管理 UI** — 文档上传、删除、列表查看、一键重建索引，无需命令行
- **来源可追溯** — 每轮回答附带 Top-5 检索片段（含 rerank_score），HTML 折叠展开
- **RESTful API** — FastAPI + Swagger 文档（`/docs`）

---

## 快速开始

### 环境要求

- Python 3.11+
- MySQL 8.0+
- [硅基流动 API Key](https://siliconflow.cn)（Embedding + Reranker）
- [DeepSeek API Key](https://platform.deepseek.com)（LLM）
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

| 变量 | 说明 | 示例 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | `sk-xxx` |
| `SILICONFLOW_API_KEY` | 硅基流动 API 密钥（Embedding + Reranker） | `sk-xxx` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-large-zh-v1.5` |
| `LLM_MODEL` | 生成模型 | `deepseek-chat` |
| `CHUNK_SIZE` | 分块大小 | `400` |
| `CHUNK_OVERLAP` | 分块重叠 | `60` |
| `DB_PASS` | MySQL 密码 | |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` / `SMTP_FROM` | SMTP 配置 | `smtp.163.com:465` |

### 3. 初始化数据库

首次启动自动建表，也可手动：

```sql
CREATE DATABASE IF NOT EXISTS company_rag
  DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. 构建知识库

**方式一：Web UI 上传**（推荐）

启动服务后访问 `/kb`，通过 Web 界面上传 .pdf / .docx / .doc 文件并点击「重建索引」。

**方式二：命令行**

将文档放入 `wiki_docs/` 目录：

```bash
python build_index.py
```

### 5. 启动服务

```bash
python app.py
```
---

## 项目结构

```
company-rag/
├── app.py                  # FastAPI 主服务：Auth 中间件 + 全部 API 路由
├── auth.py                 # 邮箱验证码发送 + Session 生命周期管理
├── database.py             # MySQL 连接池 + 5 张表自动建表
├── build_index.py          # 命令行离线索引构建（解析→分块→向量化→入库）
├── kb_manager.py           # 知识库管理：文档 CRUD + 索引重建
├── document_loader.py      # 文档解析：docx(含表格)/pdf(PyMuPDF)/MIME HTML
├── chunker.py              # RecursiveCharacterTextSplitter 分块策略
├── embeddings.py           # 硅基流动 BGE-large-zh-v1.5 Embedding 配置
├── vector_store.py         # ChromaDB 向量库 + BM25Retriever + 索引持久化
├── rag_chain.py            # RAG 核心管线：Query改写→Condense(摘要+上下文)→混合检索→Reranker→生成(内联来源标注)→Self-Check(逐claim核验→safe-rewrite→refuse)→缓冲核验后流式输出
├── requirements.txt
├── .env.example
├── .gitignore
├── static/
│   ├── chat.html           # ChatGPT 风格 SPA 问答前端
│   └── kb.html             # 知识库管理界面（上传/删除/重建）
├── wiki_docs/              # 知识库源文档目录（gitignore）
├── chroma_db/              # ChromaDB 向量数据（gitignore）
└── bm25_index.pkl          # BM25 稀疏索引（gitignore）
```

---

## 开源协议

MIT
