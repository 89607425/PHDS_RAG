import os
import json
import pickle
import chromadb
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from embeddings import get_embedding_model

_BM25_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_index.pkl")
_bm25_retriever = None


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
    _save_bm25_index(chunks)
    return vs


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        tokens = []
        for token in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\s]', text):
            token = token.strip()
            if token:
                tokens.append(token.lower())
        return tokens

    def search(self, query: str, top_n: int = 20) -> list[dict]:
        tokenized = self._tokenize(query)
        if not tokenized:
            return []
        scores = self.bm25.get_scores(tokenized)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        results = []
        for idx, score in indexed:
            chunk = self.chunks[idx]
            results.append({
                "content": chunk["text"],
                "metadata": chunk["metadata"],
                "bm25_score": float(score),
            })
        return results


def _save_bm25_index(chunks: list[dict]):
    with open(_BM25_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"BM25 索引已保存: {len(chunks)} 个分块")


def get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever
    if not os.path.exists(_BM25_FILE):
        raise FileNotFoundError(f"BM25 索引未找到: {_BM25_FILE}，请先运行 build_index.py")
    with open(_BM25_FILE, "rb") as f:
        chunks = pickle.load(f)
    _bm25_retriever = BM25Retriever(chunks)
    return _bm25_retriever


def rebuild_bm25_from_chromadb():
    vs = get_vector_store()
    count = vs._collection.count()
    if count == 0:
        print("向量库为空，跳过 BM25 重建")
        return
    results = vs._collection.get(include=["documents", "metadatas"])
    chunks = []
    for i, text in enumerate(results["documents"]):
        chunks.append({"text": text, "metadata": results["metadatas"][i] if results["metadatas"] else {}})
    _save_bm25_index(chunks)
    global _bm25_retriever
    _bm25_retriever = None
    print(f"BM25 索引已重新构建: {len(chunks)} 个分块")
