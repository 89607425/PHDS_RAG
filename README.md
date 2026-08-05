# PHDS_RAG

> 企业级 RAG 知识库问答系统 · 零 GPU 部署 · 全 API 调用 · 分钟级上线

基于 **DeepSeek** 大语言模型 + **硅基流动** Embedding + **ChromaDB** 向量数据库，面向企业内部知识管理的轻量级 RAG 解决方案。支持 Word 文档一键入库，提供 Web 聊天界面和 RESTful API 两套访问方式。

---

## 技术架构

```
Word 文档 → python-docx 提取文本 → LangChain 分块 → 硅基流动 Embedding → ChromaDB 存储
                                                                              ↓
用户提问 → 硅基流动向量化 → ChromaDB 相似检索 Top-K → DeepSeek 生成答案 → 返回（含来源引用）
```

| 环节 | 技术选型 |
|------|----------|
| 文档解析 | python-docx |
| 文本分块 | LangChain RecursiveCharacterTextSplitter |
| 向量化 | 硅基流动 API · BAAI/bge-large-zh-v1.5 |
| 向量数据库 | ChromaDB（本地嵌入式） |
| 大语言模型 | DeepSeek API · deepseek-chat |
| 编排框架 | LangChain |
| Web 服务 | FastAPI + Uvicorn |
| 聊天界面 | Gradio |

---

## 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境
conda create -n RAG python=3.11 -y
conda activate RAG

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

在项目根目录创建 `.env` 文件（参考 `.env.example`）：

```env
SILICONFLOW_API_KEY=sk-your-siliconflow-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
LLM_MODEL=deepseek-chat
CHUNK_SIZE=400
CHUNK_OVERLAP=60
COLLECTION_NAME=company_wiki
DB_DIR=./chroma_db
```

> 🔑 申请地址：[硅基流动](https://siliconflow.cn) · [DeepSeek](https://platform.deepseek.com)

### 3. 导入知识库

将 Word 文档放入 `wiki_docs/` 目录，执行入库脚本：

```bash
python build_index.py
```

### 4. 启动服务

```bash
python app.py
```

服务启动后可访问：

| 地址 | 说明 |
|------|------|
| `http://localhost:7860` | 导航页 |
| `http://localhost:7860/chat` | 聊天界面 |
| `http://localhost:7860/docs` | API 文档 (Swagger) |

---

## API 接口

### POST /api/ask — RAG 知识问答

```bash
curl -X POST http://localhost:7860/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "年假有多少天？"}'
```

响应：

```json
{
  "answer": "根据《员工手册》规定...",
  "sources": [
    {"title": "员工手册", "source": "./wiki_docs/员工手册.docx"}
  ],
  "has_kb": true
}
```

### GET /api/health — 健康检查

```bash
curl http://localhost:7860/api/health
# {"status": "ok", "has_kb": true}
```

---

## 项目结构

```
PHDS_RAG/
├── .env.example            # 环境变量模板
├── .gitignore
├── requirements.txt        # Python 依赖
├── build_index.py          # 离线入库脚本
├── app.py                  # 主服务（FastAPI + Gradio）
├── document_loader.py      # Word 文档解析
├── chunker.py              # 文本分块
├── embeddings.py           # Embedding 模型配置
├── vector_store.py         # ChromaDB 向量库操作
├── rag_chain.py            # RAG 核心链路
├── wiki_docs/              # Word 文档目录（不入库）
└── chroma_db/              # 向量数据库（不纳入版本控制）
```

---

## 特性

- **零 GPU 依赖**：全 API 调用，普通 CPU 服务器即可运行
- **中文优化**：BGE-large-zh 向量模型 + DeepSeek 中文大模型
- **开箱即用**：pip install + 配置 API key + 放入文档 = 运行
- **双界面**：Gradio Web 聊天 + RESTful API（含 Swagger 文档）
- **来源可追溯**：答案附带文档来源引用
- **低成本**：DeepSeek ¥1/百万 Token，硅基流动免费额度充足

---

## 许可证

MIT License
