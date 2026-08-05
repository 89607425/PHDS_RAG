from document_loader import load_all_docx
from chunker import split_documents
from vector_store import build_vector_store
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("📄 加载 Word 文档...")
    docs = load_all_docx("./wiki_docs")
    print(f"   共加载 {len(docs)} 篇文档")

    print("✂️ 分块处理...")
    chunks = split_documents(docs)
    print(f"   共生成 {len(chunks)} 个分块")

    print("📐 向量化并写入 ChromaDB...")
    build_vector_store(chunks)
    print("✅ 入库完成！")
