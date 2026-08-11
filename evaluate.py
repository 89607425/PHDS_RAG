"""RAG 评估脚本

使用方法:
    python evaluate.py

指标:
  - Hit Rate@k  : 预期文档是否在 Top-k 检索结果中出现（k = RERANK_TOP_N）
  - MRR@k       : 正确答案在检索结果中的平均倒数排名
  - Recall@k    : 检索到的相关文档占比（召回率）
  - NDCG@k      : 归一化折损累计增益（排位敏感指标）
  - Faithfulness: 生成回答是否忠于检索到的上下文（LLM-as-Judge）
  - Answer Relevancy: 回答是否紧扣问题（LLM-as-Judge）

下一步:
  1. 编辑 test_set.json，添加更多问题和标准答案
  2. 调整 chunk_size / prompt 后重新运行，对比指标变化
"""

import json
import math
import re
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from rag_chain import ask, format_docs, hybrid_retrieve
from config import (
    RERANK_TOP_N, RETRIEVAL_TOP_N, RRF_K,
    CHUNK_SIZE, CHUNK_OVERLAP,
    LLM_MODEL, TEMPERATURE, MAX_TOKENS,
    SELF_CHECK_THRESHOLD, SUMMARY_THRESHOLD,
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_TOKENS,
    RERANKER_MODEL, VISION_MODEL,
)

JUDGE_PROMPT_FAITHFULNESS = """你是一个 RAG 系统评估专家。请判断以下 AI 回答是否完全基于提供的上下文片段，没有编造任何信息。

## 评估标准
- 如果回答中每条事实陈述都能在上下文片段中找到依据，得 1 分
- 如果回答中包含任何凭空编造、与上下文矛盾或无法验证的信息，得 0 分
- 如果上下文不相关或为空，但回答声明"无法回答"，也得 1 分

## 上下文
{context}

## 问题
{question}

## AI 回答
{answer}

请输出 JSON 格式，只输出分数和简短理由:
{{"score": 0 或 1, "reason": "一句中文理由"}}"""

JUDGE_PROMPT_RELEVANCY = """你是一个 RAG 系统评估专家。请判断以下 AI 回答是否直接、完整地回答了用户问题。

## 评估标准
- 1 分：回答直接针对问题，内容完整，没有跑题
- 0.5 分：部分相关，但遗漏了关键信息或者包含了无关内容
- 0 分：答非所问，完全没有回答用户的问题

## 问题
{question}

## AI 回答
{answer}

请输出 JSON 格式，只输出分数和简短理由:
{{"score": 0 / 0.5 / 1, "reason": "一句中文理由"}}"""


def get_judge_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        max_tokens=200,
    )


def compute_retrieval_metrics(test_set: list[dict], k: int = 5) -> tuple[float, float, float, float]:
    """用 rag_chain.hybrid_retrieve() 走完整检索链路（向量+BM25+RRF+reranker）"""
    hit_count = 0
    mrr_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    n = 0

    for item in test_set:
        expected_doc = item.get("expected_doc")
        if not expected_doc:
            continue
        n += 1
        docs = hybrid_retrieve(item["question"])
        docs = docs[:k]
        hit = False
        found_rank = None
        for rank, doc in enumerate(docs, 1):
            title = doc.get("metadata", {}).get("title", "")
            if expected_doc in title or title in expected_doc:
                mrr_sum += 1.0 / rank
                hit = True
                found_rank = rank
                break
        if hit:
            hit_count += 1
            recall_sum += 1.0
            ndcg_sum += 1.0 / math.log2(found_rank + 1)

    hit_rate = hit_count / n if n else 0
    mrr = mrr_sum / n if n else 0
    recall = recall_sum / n if n else 0
    ndcg = ndcg_sum / n if n else 0
    return hit_rate, mrr, recall, ndcg


def judge(judge_llm, prompt_template: str, **kwargs) -> dict:
    prompt = prompt_template.format(**kwargs)
    resp = judge_llm.invoke(prompt)
    try:
        result = json.loads(resp.content)
    except json.JSONDecodeError:
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
            result = json.loads(text)
        else:
            result = {"score": 0, "reason": f"parse error: {text[:80]}"}
    return result


def main():
    print("=" * 60)
    print("  RAG 评估报告")
    print("=" * 60)

    with open("test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"\n📋 测试集: {len(test_set)} 个问题")

    judge_llm = get_judge_llm()
    top_k = RERANK_TOP_N

    # ---- 1. 检索指标 ----
    print(f"\n" + "-" * 40)
    print(f"🔍 检索指标 (RERANK_TOP_N = {top_k})")
    print("-" * 40)

    hr, mrr, recall, ndcg = compute_retrieval_metrics(test_set, k=top_k)
    n_retrieval = sum(1 for t in test_set if t.get("expected_doc"))
    print(f"  Hit Rate@{top_k} : {hr:.2%}  ({n_retrieval} 题)")
    print(f"  MRR@{top_k}      : {mrr:.3f}")
    print(f"  Recall@{top_k}   : {recall:.2%}")
    print(f"  NDCG@{top_k}     : {ndcg:.3f}")

    # ---- 2. 生成指标 (逐个运行) ----
    print("\n" + "-" * 40)
    print("🤖 生成指标 (Faithfulness + Relevancy)")
    print("-" * 40)

    faithfulness_scores = []
    relevancy_scores = []
    total_latency = 0
    details = []

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        expected_doc = item.get("expected_doc")

        t0 = time.time()
        result = ask(question)
        latency = int((time.time() - t0) * 1000)
        total_latency += latency

        answer = result["answer"]
        sources = result.get("sources", [])

        source_titles = [s.get("title", "") for s in sources]
        rank_str = "-"
        if expected_doc:
            for rank, t in enumerate(source_titles, 1):
                if expected_doc in t or t in expected_doc:
                    rank_str = str(rank)
                    break

        context_text = "\n---\n".join(
            s.get("content", "") for s in sources
        ) if sources else "(无上下文 - 回退模式)"
        faith_result = judge(judge_llm, JUDGE_PROMPT_FAITHFULNESS,
                             context=context_text, question=question, answer=answer)
        faithfulness_scores.append(faith_result["score"])

        rel_result = judge(judge_llm, JUDGE_PROMPT_RELEVANCY,
                           question=question, answer=answer)
        relevancy_scores.append(rel_result["score"])

        details.append({
            "rank": rank_str,
            "faith": faith_result["score"],
            "rel": rel_result["score"],
            "latency": latency,
        })

        faith_icon = "✅" if faith_result["score"] == 1 else "❌"
        print(f"\n  [{i}] {question[:45]}...")
        print(f"      预期文档: {expected_doc or '无'}  |  检索排名: #{rank_str}")
        print(f"      {faith_icon} Faithfulness: {faith_result['score']} — {faith_result['reason'][:60]}")
        print(f"      📌 Relevancy: {rel_result['score']} — {rel_result['reason'][:60]}")
        print(f"      ⏱ {latency}ms")

    # ---- 3. 汇总 ----
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_rel = sum(relevancy_scores) / len(relevancy_scores)
    avg_latency = total_latency / len(test_set)

    print("\n" + "=" * 60)
    print("  📊 汇总")
    print("=" * 60)
    print(f"  Hit Rate@{top_k}      : {hr:.2%}")
    print(f"  MRR@{top_k}           : {mrr:.3f}")
    print(f"  Recall@{top_k}        : {recall:.2%}")
    print(f"  NDCG@{top_k}          : {ndcg:.3f}")
    print(f"  Faithfulness    : {avg_faith:.2f} / 1.0")
    print(f"  Answer Relevancy: {avg_rel:.2f} / 1.0")
    print(f"  Avg Latency     : {avg_latency:.0f}ms")
    print(f"  Test Questions  : {len(test_set)}")
    print("=" * 60)

    print(f"""
  💡 如何解读:
  - Hit Rate@{top_k} 高 → 检索能在前{top_k}个片段中找到正确答案
  - Faithfulness 高 → 模型没有编造信息，忠于文档
  - Relevancy 高 → 回答紧扣问题，没有跑题
  - 如果 Faithfulness 低但 Relevancy 高 → prompt 可能太宽松
  - 如果 Hit Rate 低 → chunk_size 或 embedding 可能需要调优
""")

    _append_to_markdown(test_set, hr, mrr, recall, ndcg, top_k, n_retrieval,
                        faithfulness_scores, relevancy_scores,
                        total_latency, details)


RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.md")



def _get_params_table() -> str:
    return f"""| CHUNK_SIZE | {CHUNK_SIZE} |
| CHUNK_OVERLAP | {CHUNK_OVERLAP} |
| TEMPERATURE | {TEMPERATURE} |
| MAX_TOKENS | {MAX_TOKENS} |
| RETRIEVAL_TOP_N | {RETRIEVAL_TOP_N} |
| RERANK_TOP_N | {RERANK_TOP_N} |
| RRF_K | {RRF_K} |
| SELF_CHECK_THRESHOLD | {SELF_CHECK_THRESHOLD} |
| SUMMARY_THRESHOLD | {SUMMARY_THRESHOLD} |
| EMBEDDING_MODEL | {EMBEDDING_MODEL} |
| EMBEDDING_BATCH_SIZE | {EMBEDDING_BATCH_SIZE} |
| EMBEDDING_MAX_TOKENS | {EMBEDDING_MAX_TOKENS} |
| LLM_MODEL | {LLM_MODEL} |
| RERANKER_MODEL | {RERANKER_MODEL} |
| VISION_MODEL | {VISION_MODEL} |"""


def _append_to_markdown(test_set, hr, mrr, recall, ndcg, top_k, n_retrieval,
                        faith_scores, rel_scores, total_latency, details):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_total = len(test_set)
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0
    avg_latency = total_latency / n_total if n_total else 0

    run_num = 1
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            run_num = len(re.findall(r'^### Run \d+', f.read(), re.MULTILINE)) + 1

    rows = ""
    for i, item in enumerate(test_set, 1):
        d = details[i - 1]
        q = item["question"][:30]
        doc = item.get("expected_doc") or "无"
        rows += f"| {i} | {q}... | {doc} | #{d['rank']} | {d['faith']} | {d['rel']} | {d['latency']}ms |\n"

    params_table = _get_params_table()
    markdown = f"""
### Run {run_num} — {now}

**参数:**

| Param | Value |
|-------|-------|
{params_table}

| 指标 | 数值 |
|------|------|
| 测试用例数 | {n_total} |
| Hit Rate@{top_k} | **{hr:.2%}** ({sum(1 for t in test_set if t.get("expected_doc"))} 题) |
| MRR@{top_k} | {mrr:.3f} |
| Recall@{top_k} | {recall:.2%} |
| NDCG@{top_k} | {ndcg:.3f} |
| Faithfulness | **{avg_faith:.2f}** / 1.0 |
| Answer Relevancy | {avg_rel:.2f} / 1.0 |
| Avg Latency | {avg_latency:.0f}ms |

| # | 问题 | 预期文档 | 排名 | Faith | Rel | 延迟 |
|---|------|----------|------|-------|-----|------|
{rows}
"""

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\n📝 结果已追加到 {RESULTS_FILE} (Run {run_num})")


if __name__ == "__main__":
    main()
