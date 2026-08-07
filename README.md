# Company-RAG

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/fastapi-0.141-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/llm-DeepSeek--chat-purple.svg" alt="LLM">
  <img src="https://img.shields.io/badge/embedding-BGE--large--zh-orange.svg" alt="Embedding">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

<p align="center">
  <strong>Enterprise RAG Knowledge Q&A System</strong><br>
  Zero-GPU · Full API Stack · Email Auth · Conversation Persistence · SPA Frontend
</p>

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│  .docx/.pdf   │───▶│  Chunker     │───▶│  SiliconFlow    │
│  .doc/MIME   │    │  (LangChain) │    │  BGE-large-zh   │
└──────────────┘    └──────────────┘    └────────┬────────┘
                                                 │
                    ┌────────────────────────────▼─────────────────────────────┐
                    │                      ChromaDB                             │
                    │                 (Persistent Vector Store)                  │
                    └────────────────────────────┬─────────────────────────────┘
                                                 │
┌──────────────┐    ┌────────────────────────────▼─────────────────────────────┐
│   SPA Chat   │◀──▶│  FastAPI Server                                           │
│  (HTML/CSS/  │    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│   JS)        │    │  │  Auth    │  │  RAG     │  │  Conv    │  │  Eval     │ │
│              │    │  │  SMTP    │  │  Chain   │  │  CRUD    │  │  Metrics  │ │
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

| Layer | Technology |
|-------|-----------|
| Document Parsing | python-docx, PyMuPDF, email MIME parser |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embedding | SiliconFlow API · `BAAI/bge-large-zh-v1.5` |
| Vector Store | ChromaDB (local persistent) |
| LLM | DeepSeek API · `deepseek-chat` |
| Web Framework | FastAPI + Uvicorn |
| Frontend | Vanilla SPA (HTML/CSS/JS) — ChatGPT-style UI |
| Database | MySQL 8.0 + PyMySQL |
| Auth | Email verification code (SMTP) + session cookie |

---

## Features

- **Zero GPU** — Fully API-driven, runs on commodity CPU
- **Chinese Optimized** — BGE-large-zh embeddings + DeepSeek Chinese LLM
- **Multi-format Documents** — `.docx`, `.pdf`, and legacy `.doc` (MIME HTML) files
- **Email Login** — SMTP verification codes, 24h session persistence
- **Conversation History** — Full CRUD persisted to MySQL, visible in Navicat
- **Beautiful SPA** — ChatGPT-inspired responsive chat interface
- **Source Citations** — Every answer includes Top-5 retrieved document fragments
- **Built-in Evaluation** — Hit Rate@5, MRR, Faithfulness, Answer Relevancy via LLM-as-Judge
- **RESTful API** — Full Swagger docs at `/docs`

---

## Quick Start

### Prerequisites

- Python 3.11+
- MySQL 8.0+ running on `localhost:3306`
- [SiliconFlow API Key](https://siliconflow.cn)
- [DeepSeek API Key](https://platform.deepseek.com)
- SMTP credentials (e.g. 163 mailbox with auth code enabled)

### 1. Install Dependencies

```bash
conda create -n RAG python=3.11 -y
conda activate RAG
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys, MySQL credentials, and SMTP settings
```

Key environment variables:

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `SILICONFLOW_API_KEY` | SiliconFlow API key |
| `DB_PASS` | MySQL root password |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` | SMTP server config |

### 3. Initialize MySQL Schema

Tables are auto-created on first launch. You can also verify manually:

```sql
CREATE DATABASE IF NOT EXISTS company_rag
  DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 4. Build Knowledge Base

Place documents in `wiki_docs/`, then:

```bash
python build_index.py
```

### 5. Start Server

```bash
python app.py
```

| URL | Description |
|-----|-------------|
| `http://localhost:7860` | Landing page |
| `http://localhost:7860/login` | Email verification login |
| `http://localhost:7860/chat` | SPA chat interface |
| `http://localhost:7860/docs` | Swagger API docs |

---

## API Reference

### Authentication

All `/api/*` and `/chat` endpoints require a valid `session_token` cookie obtained via login.

<details>
<summary><b>POST /login/send-code</b> — Send verification code</summary>

```bash
curl -X POST http://localhost:7860/login/send-code \
  -H "Content-Type: application/json" \
  -d '{"email": "user@company.com"}'

# {"ok": true}
```
</details>

<details>
<summary><b>POST /login/verify</b> — Verify code & login</summary>

```bash
curl -X POST http://localhost:7860/login/verify \
  -H "Content-Type: application/json" \
  -d '{"email": "user@company.com", "code": "123456"}'

# Sets session_token cookie, returns {"ok": true}
```
</details>

<details>
<summary><b>GET /logout</b> — Logout</summary>

```bash
curl http://localhost:7860/logout
# Clears session_token cookie, redirects to /login
```
</details>

### Knowledge Q&A

<details>
<summary><b>POST /api/chat</b> — Ask a question (RAG)</summary>

```bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -b "session_token=YOUR_TOKEN" \
  -d '{"question": "客单价怎么计算？"}'
```

Response:

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

If `conversation_id` is `null`, a new conversation is auto-created from the question.
</details>

<details>
<summary><b>GET /api/health</b> — Health check</summary>

```bash
curl http://localhost:7860/api/health
# {"status": "ok", "has_kb": true}
```
</details>

### Conversation Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/me` | Current user info |
| `GET` | `/api/conversations` | List user's conversations |
| `POST` | `/api/conversations` | Create new conversation |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |
| `GET` | `/api/conversations/{id}/messages` | Get messages for a conversation |

---

## Evaluation

The project includes a self-contained RAG evaluation framework.

```bash
python evaluate.py
```

Metrics:

| Metric | Description | Score |
|--------|-------------|-------|
| Hit Rate@5 | Ground truth doc appears in Top-5 retrieval | 100% |
| MRR@5 | Mean Reciprocal Rank of correct doc | 0.836 |
| Faithfulness | Answer grounded in retrieved context (LLM-as-Judge) | 0.87 |
| Answer Relevancy | Answer directly addresses the question (LLM-as-Judge) | 0.83 |

Extend `test_set.json` with more question-answer pairs to improve benchmark coverage.

---

## Project Structure

```
company-rag/
├── app.py                  # FastAPI server + auth middleware + API routes
├── auth.py                 # Email verification + session management
├── database.py             # MySQL connection pool + schema init
├── build_index.py          # Offline document ingestion pipeline
├── document_loader.py      # .docx / .pdf / .doc (MIME) parser
├── chunker.py              # LangChain text splitter
├── embeddings.py           # SiliconFlow embedding config
├── vector_store.py         # ChromaDB vector store operations
├── rag_chain.py            # Core RAG pipeline (retrieve → generate)
├── evaluate.py             # RAG evaluation (Hit Rate, MRR, Faithfulness)
├── test_set.json           # Evaluation test questions + ground truth
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── static/
│   └── chat.html           # ChatGPT-style SPA frontend
└── wiki_docs/              # Source documents (gitignored)
    ├── 客单价.pdf
    ├── 核心指标概览.pdf
    ├── 实际毛利率.docx
    ├── 库存周转.docx
    ├── ... (14 documents total)
```

---

## Database Schema

5 tables in MySQL `company_rag` database:

| Table | Description |
|-------|-------------|
| `users` | Registered users (email unique) |
| `verification_codes` | Time-limited login codes (5 min expiry) |
| `sessions` | Auth session tokens (24h expiry) |
| `conversations` | Chat conversations per user |
| `messages` | Messages within conversations (CASCADE delete) |

All tables use `utf8mb4` charset with `InnoDB` engine.

---

## License

MIT
