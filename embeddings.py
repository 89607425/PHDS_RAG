from langchain_openai import OpenAIEmbeddings
from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_TOKENS
import os
import httpx

def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base="https://api.siliconflow.cn/v1",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        show_progress_bar=True,
        chunk_size=EMBEDDING_BATCH_SIZE,
        http_client=httpx.Client(proxy=None),
    )
