import os
import re
import json
import time
import logging
from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from vector_store import get_vector_store, get_bm25_retriever
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_NO_PROXY_CLIENT = httpx.Client(proxy=None)


def get_llm(streaming: bool = False):
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=2000,
        streaming=streaming,
        http_client=_NO_PROXY_CLIENT,
    )


RAG_SYSTEM = """你是一个公司内部知识助手。请根据以下检索到的文档片段，回答用户的问题。

## 规则
1. 只根据提供的文档片段回答，不要编造信息
2. 如果文档中没有相关内容，明确告知"根据现有文档无法回答该问题"
3. 回答简洁、准确、条理清晰
4. 涉及具体数字、日期、金额时，务必引用原文

## 格式要求
- 使用 **粗体** 突出关键指标名称、数字和结论
- 涉及对比数据（如环比、同比）时使用表格呈现：| 指标 | 上期 | 本期 | 变化 |
- 涉及步骤或规则时使用编号列表：1. 2. 3.
- 涉及要点罗列时使用无序列表：- 要点一 - 要点二"""

FALLBACK_SYSTEM = """你是一个公司内部知识助手。
当前知识库中没有已上传的文档，以下回答基于通用知识，仅供参考。"""

LOW_CONFIDENCE_RESPONSE = "抱歉，资料不足，我无法确认该问题的准确答案。现有文档中没有找到足够的依据来回答您的问题，建议查阅原始文档或联系相关同事确认。"

VISION_PROMPT = """请详细描述这张截图/图片中的内容，包括：
1. 图片中的文字内容（完整提取）
2. 表格中的数据结构和数值
3. 图表中的趋势、指标和数据含义
4. UI 界面中的关键信息和布局

如果图片中有数字、日期、金额、百分比等关键数据，请精确提取。"""


def _check_knowledge_base() -> bool:
    try:
        vs = get_vector_store()
        return vs._collection.count() > 0
    except Exception:
        return False


def describe_image(image_base64: str) -> str | None:
    """使用硅基流动视觉模型将图片转为文字描述，供 RAG 管线使用。"""
    try:
        vision_model = os.getenv("VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        vision_llm = ChatOpenAI(
            model=vision_model,
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            temperature=0.1,
            max_tokens=800,
            http_client=_NO_PROXY_CLIENT,
        )
        msg = HumanMessage(content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ])
        resp = vision_llm.invoke([msg])
        description = resp.content.strip()
        if description:
            logger.info(f"Vision model described image: {description[:80]}...")
            return description
    except Exception as e:
        logger.warning(f"Vision model failed: {e}")
    return None


def _extract_tokens(response) -> int:
    try:
        usage = response.response_metadata.get("token_usage", {})
        return usage.get("total_tokens", 0)
    except Exception:
        return 0


def format_docs(docs: list[dict]) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.get("metadata", {}).get("title", "未知文档")
        content = doc.get("content", "")
        formatted.append(f"[片段{i} | 来源：{title}]\n{content}")
    return "\n\n".join(formatted)


# ===== 1. Query Rewriting =====

QUERY_REWRITE_PROMPT = """你是一个查询优化助手。将用户的模糊问题改写为更精确的检索查询。

规则：
1. 补全缩写和专业术语（如"环比"→"环比率/环比差值"）
2. 将口语化表达转为正式术语
3. 如果问题本身已经很精确，直接返回原问题
4. 只输出改写后的问题，不要任何解释

原始问题：{question}
改写后："""


def rewrite_query(question: str) -> str:
    try:
        llm = get_llm()
        prompt = QUERY_REWRITE_PROMPT.format(question=question)
        resp = llm.invoke(prompt)
        rewritten = resp.content.strip()
        if len(rewritten) > 3 and rewritten != question:
            logger.info(f"Query rewritten: {question} -> {rewritten}")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")
    return question


# ===== 2. Multi-turn Condense =====

CONDENSE_PROMPT = """你是一个对话压缩助手。根据提供的会话摘要和最近对话，将用户的最新问题改写为一个不依赖上下文也能理解的独立问题。

{session_context}

用户最新问题：{question}

请将最新问题改写为独立的检索查询（只输出改写后的问题）："""


def condense_question(question: str, history: list[dict], summary: str = None) -> str:
    if not history:
        return question
    try:
        recent = history[-6:]
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
            for m in recent
        )
        session_context = ""
        if summary:
            session_context = f"## 会话摘要（包含早期对话中的业务对象、指标、筛选条件等上下文）\n{summary}\n\n## 最近对话\n{history_text}"
        else:
            session_context = f"## 对话历史\n{history_text}"
        llm = get_llm()
        prompt = CONDENSE_PROMPT.format(session_context=session_context, question=question)
        resp = llm.invoke(prompt)
        condensed = resp.content.strip()
        if len(condensed) > 3:
            logger.info(f"Question condensed: {question[:30]}... -> {condensed[:30]}...")
            return condensed
    except Exception as e:
        logger.warning(f"Condense failed: {e}")
    return question


# ===== 3. Reranker =====

def rerank(query: str, documents: list[dict]) -> list[dict]:
    if not documents:
        return []
    try:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        docs_text = [d["content"][:500] for d in documents]
        resp = httpx.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": docs_text,
                "top_n": min(5, len(docs_text)),
            },
            timeout=30,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            ranked = []
            for r in sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True):
                idx = r.get("index", 0)
                if idx < len(documents):
                    doc = documents[idx].copy()
                    doc["rerank_score"] = r.get("relevance_score", 0)
                    ranked.append(doc)
            logger.info(f"Reranker: {len(documents)} -> {len(ranked)}")
            return ranked[:5]
        else:
            logger.warning(f"Reranker API error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Reranker failed: {e}")
    return documents[:5]


# ===== 4. Reciprocal Rank Fusion =====

def reciprocal_rank_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    scores = {}
    doc_map = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = f"{doc.get('metadata', {}).get('source', '')}:{doc.get('metadata', {}).get('chunk_index', 0)}"
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    merged = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["fusion_score"] = scores[doc_id]
        merged.append(doc)
    return merged


# ===== 5. Hybrid Retrieval =====

def hybrid_retrieve(query: str, top_n: int = 20) -> list[dict]:
    vector_docs = []
    bm25_results = []

    vs = get_vector_store()
    try:
        raw_docs = vs.similarity_search(query, k=top_n)
        vector_docs = [{
            "content": d.page_content,
            "metadata": d.metadata,
        } for d in raw_docs]
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")

    try:
        bm25 = get_bm25_retriever()
        bm25_results = bm25.search(query, top_n)
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")

    if vector_docs and bm25_results:
        merged = reciprocal_rank_fusion([vector_docs, bm25_results])
    elif vector_docs:
        merged = vector_docs
    elif bm25_results:
        merged = bm25_results
    else:
        return []

    final = rerank(query, merged)
    return final


# ===== 6. Self-Check =====

SELF_CHECK_PROMPT = """你是一个事实核查员。判断以下 AI 回答中的关键数据、公式、规则是否都能在检索上下文中找到原文支持。

## 检索上下文
{context}

## 用户问题
{question}

## AI 回答
{answer}

## 评分标准
- 1.0: 所有关键事实在上下文中有明确原文
- 0.8-0.9: 大部分事实有依据，少量合理推断
- 0.4-0.7: 部分事实缺乏依据
- 0.0-0.3: 大量编造或与上下文冲突

只输出一个 0.0-1.0 之间的数字，不要任何其他内容。"""


def self_check(question: str, answer: str, context: str) -> float:
    try:
        llm = get_llm()
        prompt = SELF_CHECK_PROMPT.format(
            question=question, answer=answer, context=context[:4000]
        )
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            score = float(match.group(1))
            score = max(0.0, min(1.0, score))
            logger.info(f"Self-check score: {score}")
            return score
        logger.warning(f"Self-check: could not parse score from: {text[:100]}")
        return 0.8
    except Exception as e:
        logger.warning(f"Self-check failed: {e}")
        return 0.8


# ===== Main API =====

def _build_sources(docs: list[dict]) -> list[dict]:
    return [
        {
            "title": d.get("metadata", {}).get("title", ""),
            "source": d.get("metadata", {}).get("source", ""),
            "content": d.get("content", "")[:300],
            "chunk_index": d.get("metadata", {}).get("chunk_index", 0),
            "rerank_score": d.get("rerank_score"),
            "bm25_score": d.get("bm25_score"),
        }
        for d in docs
    ]


def ask(question: str, conversation_history: list[dict] = None,
        conv_id: int = None, summary: str = None, summarized_until: int = 0,
        image_base64: str = None) -> dict:
    t0 = time.time()
    original_question = question

    image_description = None
    if image_base64:
        image_description = describe_image(image_base64)
        if image_description:
            question = f"【图片内容描述】\n{image_description}\n\n【用户问题】\n{question}"

    is_pure_image = image_base64 and len(original_question.strip()) <= 10

    if conversation_history and not image_base64:
        question = condense_question(question, conversation_history, summary)

    if is_pure_image:
        rewritten = question
    else:
        rewritten = rewrite_query(question)

    has_kb = _check_knowledge_base()
    sources = []

    if has_kb:
        try:
            retrieved_docs = hybrid_retrieve(rewritten)
            if retrieved_docs:
                context = format_docs(retrieved_docs)
                user_content = f"""## 检索到的文档片段
{context}

## 用户问题
{question}

## 回答"""

                llm = get_llm()
                messages = [SystemMessage(content=RAG_SYSTEM), HumanMessage(content=user_content)]
                response = llm.invoke(messages)
                answer = response.content
                tokens_used = _extract_tokens(response)

                self_check_score = self_check(question, answer, context)
                if self_check_score < 0.7:
                    answer = LOW_CONFIDENCE_RESPONSE

                sources = _build_sources(retrieved_docs)
                latency_ms = int((time.time() - t0) * 1000)

                return {
                    "answer": answer,
                    "sources": sources,
                    "has_kb": True,
                    "latency_ms": latency_ms,
                    "tokens_used": tokens_used,
                    "rewritten_query": rewritten if rewritten != original_question else None,
                    "self_check_score": self_check_score,
                }
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            has_kb = False

    llm = get_llm()
    messages = [SystemMessage(content=FALLBACK_SYSTEM), HumanMessage(content=question)]
    response = llm.invoke(messages)
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "answer": response.content, "sources": [], "has_kb": False,
        "latency_ms": latency_ms, "tokens_used": _extract_tokens(response),
    }


def ask_stream(question: str, conversation_history: list[dict] = None,
               conv_id: int = None, summary: str = None, summarized_until: int = 0,
               image_base64: str = None) -> Generator[str, None, None]:
    original_question = question

    if image_base64:
        yield json.dumps({"type": "status", "data": "正在识别图片内容..."}, ensure_ascii=False) + "\n"
        image_description = describe_image(image_base64)
        if image_description:
            question = f"【图片内容描述】\n{image_description[:2000]}\n\n【用户问题】\n{question}"

    is_pure_image = image_base64 and len(original_question.strip()) <= 10

    if conversation_history and not image_base64:
        question = condense_question(question, conversation_history, summary)

    if is_pure_image:
        rewritten = question
    else:
        rewritten = rewrite_query(question)

    has_kb = _check_knowledge_base()

    if has_kb:
        try:
            retrieved_docs = hybrid_retrieve(rewritten)
            if retrieved_docs:
                context = format_docs(retrieved_docs)
                user_content = f"""## 检索到的文档片段
{context}

## 用户问题
{question}

## 回答"""

                yield json.dumps({"type": "sources", "data": [
                    {
                        "title": d.get("metadata", {}).get("title", ""),
                        "content": d.get("content", "")[:300],
                        "chunk_index": d.get("metadata", {}).get("chunk_index", 0),
                        "rerank_score": d.get("rerank_score"),
                    }
                    for d in retrieved_docs
                ]}, ensure_ascii=False) + "\n"

                yield json.dumps({"type": "meta", "data": {
                    "rewritten_query": rewritten if rewritten != question else None,
                }}, ensure_ascii=False) + "\n"

                llm = get_llm(streaming=True)
                messages = [SystemMessage(content=RAG_SYSTEM), HumanMessage(content=user_content)]
                for chunk in llm.stream(messages):
                    if chunk.content:
                        yield json.dumps({"type": "token", "data": chunk.content}, ensure_ascii=False) + "\n"

                return
        except Exception as e:
            logger.error(f"Stream RAG failed: {e}")

    llm = get_llm(streaming=True)
    messages = [SystemMessage(content=FALLBACK_SYSTEM), HumanMessage(content=question)]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield json.dumps({"type": "token", "data": chunk.content}, ensure_ascii=False) + "\n"


# ===== 7. Persistent Conversation Summary =====

SUMMARIZE_PROMPT = """你是一个会话摘要助手。根据已有的摘要和新的对话内容，生成更新后的会话摘要。

## 已有摘要
{old_summary}

## 新对话
{new_conversation}

## 要求
1. 将新对话中的关键信息合并到已有摘要中，新摘要覆盖旧摘要
2. 必须保留以下信息（如有）：
   - 用户关注的具体业务对象（如会员品单价、客单价、库存周转等指标）
   - 数字、公式和计算规则
   - 区域/门店/时间范围等筛选条件
   - 用户尚未得到满意回答的问题
3. 只基于对话内容，不得编造信息
4. 用简洁的中文分条列出（3-8 条）
5. 直接输出摘要，不要任何前缀解释"""


def maybe_update_summary(conv_id: int):
    from auth import get_conversation_summary, get_conversation_messages_from, update_conversation_summary

    summary, summarized_until = get_conversation_summary(conv_id)
    new_msgs = get_conversation_messages_from(conv_id, summarized_until)

    if len(new_msgs) < 8:
        logger.info(f"[Summary] conv={conv_id}: {len(new_msgs)} unsummarized msgs (threshold: 8), skip")
        return

    logger.info(f"[Summary] conv={conv_id}: {len(new_msgs)} unsummarized msgs, triggering LLM summarization...")

    old_summary_text = summary or "（新对话，暂无历史摘要）"
    new_conversation = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:300]}"
        for m in new_msgs
    )

    try:
        llm = get_llm()
        prompt = SUMMARIZE_PROMPT.format(old_summary=old_summary_text, new_conversation=new_conversation)
        resp = llm.invoke(prompt)
        new_summary = resp.content.strip()
        if len(new_summary) < 10:
            logger.warning(f"[Summary] conv={conv_id}: generated summary too short, skip")
            return
    except Exception as e:
        logger.warning(f"[Summary] conv={conv_id}: LLM summarization failed: {e}")
        return

    last_msg_id = new_msgs[-1]["id"]
    update_conversation_summary(conv_id, new_summary, last_msg_id)
    logger.info(f"[Summary] conv={conv_id}: summary updated → covers up to message #{last_msg_id}, "
                f"length={len(new_summary)} chars")
