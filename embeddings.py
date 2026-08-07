from langchain_openai import OpenAIEmbeddings
import os
import httpx

def get_embedding_model():
    return OpenAIEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base="https://api.siliconflow.cn/v1",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        show_progress_bar=True,
        chunk_size=20,
        http_client=httpx.Client(proxy=None),
    )
