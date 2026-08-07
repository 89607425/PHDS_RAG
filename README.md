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

## 系统架构

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│  .docx/.pdf   │───▶│  文本分块     │───▶│  硅基流动        │
│  .doc/MIME   │    │  (LangChain) │    │  BGE-large-zh   │
└──────────────┘    └──────────────┘    └────────┬────────┘
                                                 │
                    ┌────────────────────────────▼─────────────────────────────┐
                    │                      ChromaDB                             │
                    │                  （本地持久化向量库）                        │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
┌──────────────┐    ┌────────────────────────────▼─────────────────────────────┐
│  SPA 聊天界面  │◀──▶│  FastAPI 服务                                              │
│  (HTML/CSS/  │    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│   JS)        │    │  │  邮箱认证  │  │  RAG     │  │  对话    │  │  效果评估  │ │
│              │    │  │  SMTP    │  │  核心链路  │  │  增删改查  │  │  指标分析  │ │
└──────────────┘    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘ │
                    │       │             │             │               │        │
                    └───────┼─────────────┼─────────────┼───────────────┼────────┘
                            │             │             │               │
                    ┌───────▼─────────────▼─────────────▼───────────────▼────────┐
                    │                       MySQL                                 │
                    │   users │ verification_codes │ sessions │ conversations    │
                    │                          messages                           │
                    └────────────────────────────────────────────────────────────┘
```

| 环节 | 技术选型 |
|------|----------|
| 文档解析 | python-docx、PyMuPDF、MIME 邮件解析 |
| 文本分块 | LangChain `RecursiveCharacterTextSplitter` |
| 向量化 | 硅基流动 API · `BAAI/bge-large-zh-v1.5` |
| 向量数据库 | ChromaDB（本地持久化） |
| 大语言模型 | DeepSeek API · `deepseek-chat` |
| Web 框架 | FastAPI + Uvicorn |
| 前端界面 | 原生 SPA（HTML/CSS/JS）—— ChatGPT 风格 |
| 数据库 | MySQL 8.0 + PyMySQL |
| 用户认证 | 邮箱验证码 + Session Cookie |

---

## 功能特性

- **零 GPU 依赖** — 全 API 调用，普通 CPU 服务器即可运行
- **中文深度优化** — BGE-large-zh 向量模型 + DeepSeek 中文大模型
- **多格式文档支持** — `.docx`、`.pdf`、以及旧版 `.doc`（MIME HTML）文件
- **邮箱验证码登录** — SMTP 发送验证码，24 小时免重复登录
- **对话历史管理** — 完整增删改查，MySQL 持久化，Navicat 可直接查看
- **精美 SPA 前端** — ChatGPT 风格聊天界面，支持移动端
- **来源可追溯** — 每轮回答附带 Top-5 检索片段，可展开查看
- **内置评估体系** — Hit Rate@5、MRR、Faithfulness、Answer Relevancy（LLM-as-Judge）
- **RESTful API** — Swagger 文档地址 `/docs`

---

## 快速开始

### 环境要求

- Python 3.11+
- MySQL 8.0+（本地 `localhost:3306`）
- [硅基流动 API Key](https://siliconflow.cn)
- [DeepSeek API Key](https://platform.deepseek.com)
- SMTP 邮箱账号（如 163 邮箱，需开启 SMTP 授权码）

### 1. 安装依赖

```bash
conda create -n RAG python=3.11 -y
conda activate RAG
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API Key、MySQL 密码、SMTP 配置
```

关键配置项：

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `SILICONFLOW_API_KEY` | 硅基流动 API 密钥 |
| `DB_PASS` | MySQL 密码 |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` | SMTP 邮件服务配置 |

### 3. 初始化数据库

首次启动自动建表，也可手动创建：

```sql
CREATE DATABASE IF NOT EXISTS company_rag
  DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. 构建知识库

将文档放入 `wiki_docs/` 目录，执行：

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

### 用户认证

所有 `/api/*` 和 `/chat` 接口需要携带登录后下发的 `session_token` Cookie。

<details>
<summary><b>POST /login/send-code</b> —— 发送验证码</summary>

```bash
curl -X POST http://localhost:7860/login/send-code \
  -H "Content-Type: application/json" \
  -d '{"email": "user@company.com"}'

# {"ok": true}
```
</details>

<details>
<summary><b>POST /login/verify</b> —— 验证码登录</summary>

```bash
curl -X POST http://localhost:7860/login/verify \
  -H "Content-Type: application/json" \
  -d '{"email": "user@company.com", "code": "123456"}'

# 下发 session_token Cookie，返回 {"ok": true}
```
</details>

<details>
<summary><b>GET /logout</b> —— 退出登录</summary>

```bash
curl http://localhost:7860/logout
# 清除 session_token Cookie，跳转至 /login
```
</details>

### 知识问答

<details>
<summary><b>POST /api/chat</b> —— RAG 问答</summary>

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
  "answer": "客单价 = 销售金额 / 来客数...",
  "sources": [
    {"title": "客单价", "content": "...", "chunk_index": 3}
  ],
  "has_kb": true,
  "latency_ms": 1500,
  "tokens_used": 1200
}
```

若 `conversation_id` 为 `null`，系统自动以问题摘要为标题创建新对话。
</details>

<details>
<summary><b>GET /api/health</b> —— 健康检查</summary>

```bash
curl http://localhost:7860/api/health
# {"status": "ok", "has_kb": true}
```
</details>

### 对话管理

| 方法 | 接口 | 说明 |
|------|------|------|
| `GET` | `/api/me` | 当前用户信息 |
| `GET` | `/api/conversations` | 列出用户所有对话 |
| `POST` | `/api/conversations` | 创建新对话 |
| `DELETE` | `/api/conversations/{id}` | 删除对话 |
| `GET` | `/api/conversations/{id}/messages` | 获取对话消息列表 |

---

## 效果评估

项目内置 RAG 评估框架，可量化系统表现：

```bash
python evaluate.py
```

评估指标：

| 指标 | 说明 | 当前得分 |
|------|------|---------|
| Hit Rate@5 | 正确答案出现在 Top-5 检索结果中的比例 | 100% |
| MRR@5 | 正确答案在检索结果中的平均倒数排名 | 0.836 |
| Faithfulness | 回答是否忠于检索上下文（LLM 判定） | 0.87 |
| Answer Relevancy | 回答是否紧扣问题（LLM 判定） | 0.83 |

修改 `test_set.json` 补充更多问答对可提升基准覆盖面。

---

## 项目结构

```
company-rag/
├── app.py                  # FastAPI 主服务 + 认证中间件 + API 路由
├── auth.py                 # 邮箱验证码 + Session 管理
├── database.py             # MySQL 连接 + 自动建表
├── build_index.py          # 离线文档入库脚本
├── document_loader.py      # .docx / .pdf / .doc (MIME) 解析
├── chunker.py              # LangChain 文本分块
├── embeddings.py           # 硅基流动 Embedding 配置
├── vector_store.py         # ChromaDB 向量库操作
├── rag_chain.py            # RAG 核心链路（检索 → 生成）
├── evaluate.py             # RAG 效果评估
├── test_set.json           # 评估用测试集
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板
├── .gitignore
├── static/
│   └── chat.html           # ChatGPT 风格 SPA 前端
└── wiki_docs/              # 源文档目录（gitignore）
    ├── 客单价.pdf
    ├── 核心指标概览.pdf
    ├── 实际毛利率.docx
    ├── 库存周转.docx
    ├── ...（共 14 篇文档）
```

---

## 数据库结构

MySQL `company_rag` 库，共 5 张表：

| 表名 | 说明 |
|------|------|
| `users` | 注册用户（邮箱唯一） |
| `verification_codes` | 限时验证码（5 分钟有效） |
| `sessions` | 登录会话（24 小时有效） |
| `conversations` | 用户对话记录 |
| `messages` | 对话消息（级联删除） |

所有表采用 `utf8mb4` 字符集 + `InnoDB` 引擎。

---

## 开源协议

MIT
