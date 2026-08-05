from langchain_openai import OpenAIEmbeddings
import os

def get_embedding_model():
    return OpenAIEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base="https://api.siliconflow.cn/v1",
    )
