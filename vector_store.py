import os
import chromadb
from langchain_community.vectorstores import Chroma
from embeddings import get_embedding_model

def get_vector_store():
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=os.getenv("COLLECTION_NAME", "company_wiki"),
        embedding_function=embeddings,
        persist_directory=os.getenv("DB_DIR", "./chroma_db"),
    )

def build_vector_store(chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vs = get_vector_store()
    vs.add_texts(texts=texts, metadatas=metadatas)
    vs.persist()
    print(f"已写入 {len(texts)} 个分块到向量库")
    return vs
