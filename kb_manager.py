import os
import shutil
from document_loader import load_all_docs
from chunker import split_documents
from vector_store import build_vector_store, rebuild_bm25_from_chromadb

WIKI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki_docs")


def _ensure_wiki_dir():
    os.makedirs(WIKI_DIR, exist_ok=True)


def list_documents() -> list[dict]:
    _ensure_wiki_dir()
    docs = []
    for fname in sorted(os.listdir(WIKI_DIR)):
        fpath = os.path.join(WIKI_DIR, fname)
        if fname.startswith("~$") or not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".pdf", ".docx", ".doc"):
            continue
        size = os.path.getsize(fpath)
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
        docs.append({
            "name": fname,
            "title": os.path.splitext(fname)[0],
            "type": ext.lstrip("."),
            "size_bytes": size,
            "size": size_str,
            "modified": os.path.getmtime(fpath),
        })
    return docs


def upload_document(file_bytes: bytes, filename: str) -> dict:
    _ensure_wiki_dir()
    safe_name = os.path.basename(filename)
    if not safe_name:
        raise ValueError("Invalid filename")
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise ValueError(f"Unsupported file type: {ext}")
    fpath = os.path.join(WIKI_DIR, safe_name)
    with open(fpath, "wb") as f:
        f.write(file_bytes)
    return {"name": safe_name, "path": fpath, "size_bytes": len(file_bytes)}


def delete_document(filename: str) -> bool:
    fpath = os.path.join(WIKI_DIR, os.path.basename(filename))
    if not os.path.exists(fpath):
        return False
    os.remove(fpath)
    return True


def rebuild_index() -> dict:
    _ensure_wiki_dir()
    print("📄 Loading documents...")
    raw_docs = load_all_docs(WIKI_DIR)
    print(f"   {len(raw_docs)} documents loaded")
    print("✂️ Chunking...")
    chunks = split_documents(raw_docs)
    print(f"   {len(chunks)} chunks created")
    print("📐 Vectorizing + BM25...")
    build_vector_store(chunks)
    return {"documents": len(raw_docs), "chunks": len(chunks)}
