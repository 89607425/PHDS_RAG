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
5. **每条关键事实后必须标注来源**，格式：`[来源：文档名-片段N]`
   例如：会员品单价 = 销售额 / 来客数 [来源：客单价-片段2]
6. 不要在文末重复罗列来源，直接内联标注"""

FALLBACK_SYSTEM = """你是一个公司内部知识助手。
当前知识库中没有已上传的文档，以下回答基于通用知识，仅供参考。"""

REFUSE_RESPONSE = "抱歉，根据现有文档无法回答该问题。建议查阅原始文档或联系相关同事确认。"
VERIFY_FAILED_RESPONSE = "核验失败，暂无法确认该回答的准确性。请稍后重试或查阅原始文档。"


def _check_knowledge_base() -> bool:
    try:
        vs = get_vector_store()
        return vs._collection.count() > 0
    except Exception:
        return False


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


# ===== 6. Self-Check (per-claim verification) =====

SELF_CHECK_PROMPT = """你是一个严格的事实核查员。逐条核查 AI 回答中的每一项关键事实声明是否能在检索上下文中找到原文支持。

## 检索上下文（每个片段有唯一 ID）
{context}

## 用户问题
{question}

## AI 回答（含来源标记）
{answer}

## 核查要求
1. 将回答拆解为独立的事实声明（claim）
2. 对每条 claim，在上下文中找到最相关的片段 ID
3. 判断 verdict：
   - supported：claim 在对应片段中有明确原文
   - unsupported：claim 在检索上下文中找不到任何依据
   - contradicted：claim 与上下文中的信息冲突
   - inference：claim 是上下文信息的合理推断（非原文但逻辑一致）
4. 判定 overall_score（0.0-1.0）和 final_action

## 输出严格 JSON（不要 markdown 代码块，直接输出纯 JSON）
{{"claims":[{{"claim":"...","verdict":"supported|unsupported|contradicted|inference","source_fragment":"文档名-片段N 或 null"}}],"overall_score":0.0,"final_action":"pass|rewrite|refuse"}}

## final_action 规则
- pass：所有 verdict 为 supported 或 inference，无 unsupported/contradicted
- rewrite：存在 unsupported 或 contradicted 的 claim，但核心问题仍有 supported claim 可回答
- refuse：retrieved docs 中没有与问题相关的任何内容，或全部 claim 均为 unsupported/contradicted"""


SAFE_REWRITE_PROMPT = """你是一个回答安全改写员。以下回答经核查有部分事实无法在检索上下文中找到依据，请仅保留有证据支持的内容，删除或改写不被支持的部分。

## 检索上下文
{context}

## 用户问题
{question}

## 原始回答
{original_answer}

## 核查结果（只关注 unsupported/contradicted 的 claim）
{failed_claims}

## 改写规则
1. 仅保留检索上下文中有证据支持的信息
2. 可以简化、合并，但不能添加任何新信息
3. 改写后的每条事实仍需带来源标记 [来源：文档名-片段N]
4. 如果改写后完全没有可回答的内容，直接输出：REFUSE
5. 直接输出改写后的回答，不要任何解释或前缀"""


def self_check(question: str, answer: str, context: str) -> dict:
    try:
        llm = get_llm()
        prompt = SELF_CHECK_PROMPT.format(
            question=question, answer=answer, context=context[:4000]
        )
        resp = llm.invoke(prompt)
        text = resp.content.strip()

        # Robust JSON extraction
        import re as _re
        # Remove markdown code fences
        text = _re.sub(r'```(?:json)?\s*', '', text)
        text = _re.sub(r'```', '', text)
        # Find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or start >= end:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        json_text = text[start:end+1]

        result = json.loads(json_text)
        logger.info(f"Self-check: score={result.get('overall_score', '?')}, "
                    f"action={result.get('final_action', '?')}, claims={len(result.get('claims', []))}")
        for c in result.get("claims", []):
            v = c.get("verdict", "?")
            if v in ("unsupported", "contradicted"):
                logger.warning(f"  ⚠ {v}: {c.get('claim', '?')[:80]}")
        return result
    except (json.JSONDecodeError, KeyError) as e:
        raw_preview = resp.content[:200] if 'resp' in locals() else 'N/A'
        logger.error(f"Self-check JSON parse failed: {e}. Raw: {raw_preview}")
        return {"claims": [], "overall_score": 0.0, "final_action": "refuse",
                "_error": "parse_failed"}
    except Exception as e:
        logger.error(f"Self-check LLM call failed: {e}")
        return {"claims": [], "overall_score": 0.0, "final_action": "refuse",
                "_error": "llm_failed"}


def safe_rewrite(question: str, original_answer: str, context: str, check_result: dict) -> str | None:
    failed = [c for c in check_result.get("claims", [])
              if c.get("verdict") in ("unsupported", "contradicted")]
    if not failed and check_result.get("final_action") != "rewrite":
        return None

    try:
        llm = get_llm()
        failed_text = "\n".join(f"- [{c.get('verdict')}] {c.get('claim', '')}" for c in failed[:10])
        prompt = SAFE_REWRITE_PROMPT.format(
            question=question,
            original_answer=original_answer,
            context=context[:3000],
            failed_claims=failed_text or "无具体失败项",
        )
        resp = llm.invoke(prompt)
        rewritten = resp.content.strip()
        if rewritten and rewritten.strip() != "REFUSE" and len(rewritten) > 10:
            logger.info(f"Safe rewrite succeeded: {len(rewritten)} chars")
            return rewritten
        logger.info("Safe rewrite returned REFUSE or too short")
    except Exception as e:
        logger.warning(f"Safe rewrite failed: {e}")
    return None


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


def _verify_and_finalize(question: str, raw_answer: str, context: str) -> dict:
    """Run self-check and safe-rewrite pipeline. Returns {final_answer, check_result}."""
    check_result = self_check(question, raw_answer, context)
    action = check_result.get("final_action", "refuse")

    if action == "pass":
        return {"final_answer": raw_answer, "check_result": check_result}

    if action == "rewrite":
        rewritten = safe_rewrite(question, raw_answer, context, check_result)
        if rewritten:
            return {"final_answer": rewritten, "check_result": check_result,
                    "rewritten": True}

    if check_result.get("_error"):
        logger.warning(f"Verification failed due to error: {check_result.get('_error')}")
        return {"final_answer": VERIFY_FAILED_RESPONSE, "check_result": check_result,
                "_refuse_reason": check_result.get("_error")}

    return {"final_answer": REFUSE_RESPONSE, "check_result": check_result}


def ask(question: str, conversation_history: list[dict] = None,
        conv_id: int = None, summary: str = None, summarized_until: int = 0) -> dict:
    t0 = time.time()
    original_question = question

    if conversation_history:
        question = condense_question(question, conversation_history, summary)

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
                raw_answer = response.content
                tokens_used = _extract_tokens(response)

                verified = _verify_and_finalize(question, raw_answer, context)
                final_answer = verified["final_answer"]
                check_result = verified["check_result"]

                sources = _build_sources(retrieved_docs)
                latency_ms = int((time.time() - t0) * 1000)

                return {
                    "answer": final_answer,
                    "sources": sources,
                    "has_kb": True,
                    "latency_ms": latency_ms,
                    "tokens_used": tokens_used,
                    "rewritten_query": rewritten if rewritten != original_question else None,
                    "self_check": {
                        "score": check_result.get("overall_score"),
                        "action": check_result.get("final_action"),
                        "claims": check_result.get("claims", []),
                        "rewritten": verified.get("rewritten", False),
                        "_refuse_reason": verified.get("_refuse_reason"),
                    },
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
               conv_id: int = None, summary: str = None, summarized_until: int = 0) -> Generator[str, None, None]:
    original_question = question

    if conversation_history:
        question = condense_question(question, conversation_history, summary)

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

                # Generate full answer first (non-streaming) for safety verification
                yield json.dumps({"type": "status", "data": "生成中，请稍候..."}, ensure_ascii=False) + "\n"

                llm_sync = get_llm(streaming=False)
                messages = [SystemMessage(content=RAG_SYSTEM), HumanMessage(content=user_content)]
                resp = llm_sync.invoke(messages)
                raw_answer = resp.content

                verified = _verify_and_finalize(question, raw_answer, context)
                final_answer = verified["final_answer"]
                check_result = verified["check_result"]

                yield json.dumps({"type": "self_check", "data": {
                    "score": check_result.get("overall_score"),
                    "action": check_result.get("final_action"),
                    "claims": [
                        {"claim": c.get("claim", ""), "verdict": c.get("verdict", ""),
                         "source_fragment": c.get("source_fragment")}
                        for c in check_result.get("claims", [])
                    ],
                    "rewritten": verified.get("rewritten", False),
                }}, ensure_ascii=False) + "\n"

                # Stream the verified final answer token by token
                for i in range(0, len(final_answer), 2):
                    chunk = final_answer[i:i+2]
                    yield json.dumps({"type": "token", "data": chunk}, ensure_ascii=False) + "\n"

                yield json.dumps({"type": "done", "data": {}}, ensure_ascii=False) + "\n"
                return
        except Exception as e:
            logger.error(f"Stream RAG failed: {e}")

    llm = get_llm(streaming=True)
    messages = [SystemMessage(content=FALLBACK_SYSTEM), HumanMessage(content=question)]
    for chunk in llm.stream(messages):
        if chunk.content:
            yield json.dumps({"type": "token", "data": chunk.content}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "done", "data": {}}, ensure_ascii=False) + "\n"


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
