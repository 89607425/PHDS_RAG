import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

RAG_PROMPT = """你是一个公司内部知识助手。请根据以下检索到的文档片段，回答用户的问题。

## 规则
1. 只根据提供的文档片段回答，不要编造信息
2. 如果文档中没有相关内容，明确告知"根据现有文档无法回答该问题"
3. 回答简洁、准确、条理清晰
4. 涉及具体数字、日期、金额时，务必引用原文
5. 回答末尾标注信息来源

## 检索到的文档片段
{context}

## 用户问题
{question}

## 回答"""

FALLBACK_PROMPT = """你是一个公司内部知识助手。

⚠️ 当前知识库中没有已上传的文档，以下回答基于通用知识，仅供参考。

请回答用户的问题：
{question}"""

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

def ask(question: str) -> dict:
    has_kb = _check_knowledge_base()
    llm = get_llm()

    if has_kb:
        try:
            vector_store = get_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            source_docs = retriever.invoke(question)
            context = format_docs(source_docs)
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
            sources = [
                {"title": d.metadata.get("title", ""), "source": d.metadata.get("source", "")}
                for d in source_docs
            ]
            return {"answer": answer, "sources": sources, "has_kb": True}
        except Exception as e:
            has_kb = False

    prompt = ChatPromptTemplate.from_template(FALLBACK_PROMPT)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question})
    return {"answer": answer, "sources": [], "has_kb": False}
