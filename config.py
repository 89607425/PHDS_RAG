"""RAG 系统统一配置文件

调参只需改这个文件，然后：
  python build_index.py  (改了 CHUNK_SIZE/CHUNK_OVERLAP 时需要)
  python evaluate.py     (评测结果自动记录)
"""

# ---- Doc Separation ----
CHUNK_SIZE = 500
CHUNK_OVERLAP = 40

# ---- LLM Generation ----
LLM_MODEL = "deepseek-chat"
TEMPERATURE = 0.1
MAX_TOKENS = 2000

# ---- Retrieval & Rank ----
RETRIEVAL_TOP_N = 40
RERANK_TOP_N = 5
RERANK_THRESHOLD = 0  # relevance_score 低于此值的 chunk 丢弃，0 表示禁用
RRF_K = 60

# ---- SELF-CHECK ----
SELF_CHECK_THRESHOLD = 0.8

# ---- Conversation Summary ----
SUMMARY_THRESHOLD = 8

# ---- Embedding ----
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_BATCH_SIZE = 20
EMBEDDING_MAX_TOKENS = 512  # bge-large-zh-v1.5 最大上下文窗口

# ---- reranker ----
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ---- visual model ----
VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
