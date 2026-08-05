from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks = []
    for doc in docs:
        texts = splitter.split_text(doc["text"])
        for i, text in enumerate(texts):
            chunk = {
                "text": text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(texts),
                }
            }
            chunks.append(chunk)
    return chunks
