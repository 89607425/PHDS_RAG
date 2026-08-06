import os
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from vector_store import get_vector_store

logging.basicConfig(level=logging.WARNING)

def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=2000,
    )

RAG_SYSTEM = """你是一个公司内部知识助手。请根据以下检索到的文档片段，回答用户的问题。

## 规则
1. 只根据提供的文档片段回答，不要编造信息
2. 如果文档中没有相关内容，明确告知"根据现有文档无法回答该问题"
3. 回答简洁、准确、条理清晰
4. 涉及具体数字、日期、金额时，务必引用原文
5. 回答末尾标注信息来源"""

FALLBACK_SYSTEM = """你是一个公司内部知识助手。
⚠️ 当前知识库中没有已上传的文档，以下回答基于通用知识，仅供参考。"""

def _build_rag_prompt(context, question):
    return f"""## 检索到的文档片段
{context}

## 用户问题
{question}

## 回答"""

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "未知文档")
        formatted.append(f"[片段{i} | 来源：{title}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def _check_knowledge_base():
    try:
        vs = get_vector_store()
        count = vs._collection.count()
        return count > 0
    except Exception:
        return False

def _extract_tokens(response) -> int:
    try:
        usage = response.response_metadata.get("token_usage", {})
        return usage.get("total_tokens", 0)
    except Exception:
        return 0

def ask(question: str) -> dict:
    t0 = time.time()
    has_kb = _check_knowledge_base()
    llm = get_llm()
    sources = []

    if has_kb:
        try:
            vector_store = get_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            source_docs = retriever.invoke(question)
            context = format_docs(source_docs)
            user_content = _build_rag_prompt(context, question)
            messages = [SystemMessage(content=RAG_SYSTEM), HumanMessage(content=user_content)]
            response = llm.invoke(messages)
            answer = response.content
            tokens_used = _extract_tokens(response)
            sources = [
                {
                    "title": d.metadata.get("title", ""),
                    "source": d.metadata.get("source", ""),
                    "content": d.page_content[:300],
                    "chunk_index": d.metadata.get("chunk_index", 0),
                }
                for d in source_docs
            ]
            latency_ms = int((time.time() - t0) * 1000)
            return {"answer": answer, "sources": sources, "has_kb": True,
                    "latency_ms": latency_ms, "tokens_used": tokens_used}
        except Exception:
            has_kb = False

    messages = [SystemMessage(content=FALLBACK_SYSTEM), HumanMessage(content=question)]
    response = llm.invoke(messages)
    answer = response.content
    tokens_used = _extract_tokens(response)
    latency_ms = int((time.time() - t0) * 1000)
    return {"answer": answer, "sources": [], "has_kb": False,
            "latency_ms": latency_ms, "tokens_used": tokens_used}
