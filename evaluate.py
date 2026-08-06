"""RAG 评估脚本

使用方法:
    python evaluate.py

指标:
  - Hit Rate@5  : ground_truth 是否在 Top-5 检索结果中出现
  - MRR@5       : 正确答案在检索结果中的平均倒数排名
  - Faithfulness: 生成回答是否忠于检索到的上下文（LLM-as-Judge）
  - Answer Relevancy: 回答是否紧扣问题（LLM-as-Judge）

下一步:
  1. 编辑 test_set.json，添加更多问题和标准答案
  2. 调整 chunk_size / prompt 后重新运行，对比指标变化
"""

import json
import time
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from vector_store import get_vector_store
from rag_chain import ask, format_docs

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


def compute_retrieval_metrics(test_set: list[dict]) -> tuple[float, float]:
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    hit_count = 0
    mrr_sum = 0
    n = 0

    for item in test_set:
        expected_doc = item.get("expected_doc")
        if not expected_doc:
            continue
        n += 1
        docs = retriever.invoke(item["question"])
        hit = False
        for rank, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "")
            if expected_doc in title or title in expected_doc:
                mrr_sum += 1.0 / rank
                hit = True
                break
        if hit:
            hit_count += 1

    hit_rate = hit_count / n if n else 0
    mrr = mrr_sum / n if n else 0
    return hit_rate, mrr


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

    # ---- 1. 检索指标 ----
    print("\n" + "-" * 40)
    print("🔍 检索指标 (Hit Rate + MRR)")
    print("-" * 40)

    hr, mrr = compute_retrieval_metrics(test_set)
    n_retrieval = sum(1 for t in test_set if t.get("expected_doc"))
    print(f"  Hit Rate@5 : {hr:.2%}  ({n_retrieval} 题)")
    print(f"  MRR@5      : {mrr:.3f}")

    # ---- 2. 生成指标 (逐个运行) ----
    print("\n" + "-" * 40)
    print("🤖 生成指标 (Faithfulness + Relevancy)")
    print("-" * 40)

    faithfulness_scores = []
    relevancy_scores = []
    total_latency = 0

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

        # 标注预期文档在来源中的排名
        source_titles = [s.get("title", "") for s in sources]
        rank_str = "-"
        if expected_doc:
            for rank, t in enumerate(source_titles, 1):
                if expected_doc in t or t in expected_doc:
                    rank_str = str(rank)
                    break

        # Judge: faithfulness
        context_text = "\n---\n".join(
            s.get("content", "") for s in sources
        ) if sources else "(无上下文 - 回退模式)"
        faith_result = judge(judge_llm, JUDGE_PROMPT_FAITHFULNESS,
                             context=context_text, question=question, answer=answer)
        faithfulness_scores.append(faith_result["score"])

        # Judge: relevancy
        rel_result = judge(judge_llm, JUDGE_PROMPT_RELEVANCY,
                           question=question, answer=answer)
        relevancy_scores.append(rel_result["score"])

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
    print(f"  Hit Rate@5      : {hr:.2%}")
    print(f"  MRR@5           : {mrr:.3f}")
    print(f"  Faithfulness    : {avg_faith:.2f} / 1.0")
    print(f"  Answer Relevancy: {avg_rel:.2f} / 1.0")
    print(f"  Avg Latency     : {avg_latency:.0f}ms")
    print(f"  Test Questions  : {len(test_set)}")
    print("=" * 60)

    print("""
  💡 如何解读:
  - Hit Rate@5 高 → 检索能在前5个片段中找到正确答案
  - Faithfulness 高 → 模型没有编造信息，忠于文档
  - Relevancy 高 → 回答紧扣问题，没有跑题
  - 如果 Faithfulness 低但 Relevancy 高 → prompt 可能太宽松
  - 如果 Hit Rate 低 → chunk_size 或 embedding 可能需要调优

  🔧 下一步:
  1. 补充更多测试问题到 test_set.json
  2. 调整 chunker.py 的 chunk_size/overlap 对比指标
  3. 优化 rag_chain.py 的 prompt 模板
  4. 尝试 reranker 提升检索精度
""")


if __name__ == "__main__":
    main()
