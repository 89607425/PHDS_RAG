# RAG eval results

## Metrics

| Metric | Meaning |
|--------|---------|
| Hit Rate@k | Whether expected doc appears in Top-k retrieval results |
| MRR@k | Mean Reciprocal Rank, 1/rank averaged |
| Recall@k | Proportion of relevant docs retrieved |
| NDCG@k | Normalized Discounted Cumulative Gain, rank-sensitive |
| Faithfulness | Answer fidelity to retrieved context (LLM-as-Judge) |
| Answer Relevancy | Whether answer directly addresses the question (LLM-as-Judge) |

> 每次 Run 自动记录当时的 config.py 参数。@k = RERANK_TOP_N。

## Tuning guide

```bash
# Edit params -> rebuild index -> evaluate
vim config.py        # change any parameter
python build_index.py  # only needed if CHUNK_SIZE/CHUNK_OVERLAP changed
python evaluate.py     # results auto-appended to this file
```
## Runs

### Run 1 -- 2026-08-11 10:10
| Param | Value |
|-------|-------|
| CHUNK_SIZE | 400 |
| CHUNK_OVERLAP | 60 |
| TEMPERATURE | 0.1 |
| MAX_TOKENS | 2000 |
| RETRIEVAL_TOP_N | 20 |
| RERANK_TOP_N | 5 |
| RRF_K | 60 |
| SELF_CHECK_THRESHOLD | 0.7 |
| SUMMARY_THRESHOLD | 8 |
| EMBEDDING_MODEL | BAAI/bge-large-zh-v1.5 |
| LLM_MODEL | deepseek-chat |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 |
| VISION_MODEL | Qwen/Qwen3-VL-8B-Instruct |

| Metric | Value |
|--------|-------|
| Test cases | 27 |
| Hit Rate@5 | 80.00% (25 cases) |
| MRR@5 | 0.635 |
| Recall@5 | 80.00% |
| NDCG@5 | 0.734 |
| Faithfulness | 0.93 / 1.0 |
| Answer Relevancy | 0.67 / 1.0 |
| Avg Latency | 4233ms |

| # | Question | Expected Doc | Rank | Faith | Rel | Latency |
|---|----------|-------------|------|-------|-----|---------|

### Run 2 — 2026-08-11 11:25
| Param | Value |
|-------|-------|
| CHUNK_SIZE | 400 |
| CHUNK_OVERLAP | 60 |
| TEMPERATURE | 0.1 |
| MAX_TOKENS | 2000 |
| RETRIEVAL_TOP_N | 40 |
| RERANK_TOP_N | 8 |
| RRF_K | 60 |
| SELF_CHECK_THRESHOLD | 0.6 |
| SUMMARY_THRESHOLD | 8 |
| EMBEDDING_MODEL | BAAI/bge-large-zh-v1.5 |
| LLM_MODEL | deepseek-chat |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 |
| VISION_MODEL | Qwen/Qwen3-VL-8B-Instruct |


| 指标 | 数值 |
|------|------|
| 测试用例数 | 47 |
| Hit Rate@8 | **75.56%** (45 题) |
| MRR@8 | 0.649 |
| Recall@8 | 75.56% |
| NDCG@8 | 0.675 |
| Faithfulness | **0.83** / 1.0 |
| Answer Relevancy | 0.77 / 1.0 |
| Avg Latency | 6571ms |


### Run 3 — 2026-08-11 12:03

| 指标 | 数值 |
|------|------|
| 测试用例数 | 47 |
| Hit Rate@5 | **75.56%** (45 题) |
| MRR@5 | 0.649 |
| Recall@5 | 75.56% |
| NDCG@5 | 0.675 |
| Faithfulness | **0.89** / 1.0 |
| Answer Relevancy | 0.68 / 1.0 |
| Avg Latency | 5009ms |


### Run 4 — 2026-08-11 13:44

**参数:**

| Param | Value |
|-------|-------|
| CHUNK_SIZE | 500 |
| CHUNK_OVERLAP | 70 |
| TEMPERATURE | 0.1 |
| MAX_TOKENS | 2000 |
| RETRIEVAL_TOP_N | 30 |
| RERANK_TOP_N | 5 |
| RRF_K | 60 |
| SELF_CHECK_THRESHOLD | 0.7 |
| SUMMARY_THRESHOLD | 8 |
| EMBEDDING_MODEL | BAAI/bge-large-zh-v1.5 |
| LLM_MODEL | deepseek-chat |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 |
| VISION_MODEL | Qwen/Qwen3-VL-8B-Instruct |

| 指标 | 数值 |
|------|------|
| 测试用例数 | 47 |
| Hit Rate@5 | **82.22%** (45 题) |
| MRR@5 | 0.639 |
| Recall@5 | 82.22% |
| NDCG@5 | 0.685 |
| Faithfulness | **0.77** / 1.0 |
| Answer Relevancy | 0.76 / 1.0 |
| Avg Latency | 82459ms |


### Run 5 — 2026-08-11 14:13

**参数:**

| Param | Value |
|-------|-------|
| CHUNK_SIZE | 500 |
| CHUNK_OVERLAP | 40 |
| TEMPERATURE | 0.1 |
| MAX_TOKENS | 2000 |
| RETRIEVAL_TOP_N | 30 |
| RERANK_TOP_N | 5 |
| RRF_K | 60 |
| SELF_CHECK_THRESHOLD | 0.7 |
| SUMMARY_THRESHOLD | 8 |
| EMBEDDING_MODEL | BAAI/bge-large-zh-v1.5 |
| LLM_MODEL | deepseek-chat |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 |
| VISION_MODEL | Qwen/Qwen3-VL-8B-Instruct |

| 指标 | 数值 |
|------|------|
| 测试用例数 | 47 |
| Hit Rate@5 | **82.22%** (45 题) |
| MRR@5 | 0.641 |
| Recall@5 | 82.22% |
| NDCG@5 | 0.686 |
| Faithfulness | **0.77** / 1.0 |
| Answer Relevancy | 0.76 / 1.0 |
| Avg Latency | 5545ms |


### Run 6 — 2026-08-11 14:38

**参数:**

| Param | Value |
|-------|-------|
| CHUNK_SIZE | 450 |
| CHUNK_OVERLAP | 40 |
| TEMPERATURE | 0.1 |
| MAX_TOKENS | 2000 |
| RETRIEVAL_TOP_N | 30 |
| RERANK_TOP_N | 5 |
| RRF_K | 60 |
| SELF_CHECK_THRESHOLD | 0.7 |
| SUMMARY_THRESHOLD | 8 |
| EMBEDDING_MODEL | BAAI/bge-large-zh-v1.5 |
| LLM_MODEL | deepseek-chat |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 |
| VISION_MODEL | Qwen/Qwen3-VL-8B-Instruct |

| 指标 | 数值 |
|------|------|
| 测试用例数 | 47 |
| Hit Rate@5 | **71.11%** (45 题) |
| MRR@5 | 0.601 |
| Recall@5 | 71.11% |
| NDCG@5 | 0.628 |
| Faithfulness | **0.81** / 1.0 |
| Answer Relevancy | 0.70 / 1.0 |
| Avg Latency | 4653ms |



### Run 7 — 2026-08-11 14:57

**参数:**

| Param | Value |
|-------|-------|
| CHUNK_SIZE | 500 |
| CHUNK_OVERLAP | 40 |
| TEMPERATURE | 0.1 |
| MAX_TOKENS | 2000 |
| RETRIEVAL_TOP_N | 30 |
| RERANK_TOP_N | 5 |
| RRF_K | 60 |
| SELF_CHECK_THRESHOLD | 0.8 |
| SUMMARY_THRESHOLD | 8 |
| EMBEDDING_MODEL | BAAI/bge-large-zh-v1.5 |
| LLM_MODEL | deepseek-chat |
| RERANKER_MODEL | BAAI/bge-reranker-v2-m3 |
| VISION_MODEL | Qwen/Qwen3-VL-8B-Instruct |

| 指标 | 数值 |
|------|------|
| 测试用例数 | 47 |
| Hit Rate@5 | **73.33%** (45 题) |
| MRR@5 | 0.641 |
| Recall@5 | 73.33% |
| NDCG@5 | 0.663 |
| Faithfulness | **0.81** / 1.0 |
| Answer Relevancy | 0.66 / 1.0 |
| Avg Latency | 5051ms |

| # | 问题 | 预期文档 | 排名 | Faith | Rel | 延迟 |
|---|------|----------|------|-------|-----|------|
| 1 | 客单价模块中，会员品单价和非会员品单价分别怎么计算？... | 客单价 | #1 | 1 | 0.5 | 4335ms |
| 2 | 核心指标概览中，近效期占比环比差值的涨跌颜色规则是什么？... | 核心指标概览 | #1 | 1 | 1 | 3855ms |
| 3 | 实际毛利率模块中，分析指标包含哪些？... | 实际毛利率 | #1 | 0 | 1 | 4220ms |
| 4 | 分类模块中，达成贡献和同比贡献的提示信息分别是什么？... | 分类 | #1 | 1 | 0.5 | 4456ms |
| 5 | 库存积压模块的不动销商品趋势图展示哪些内容？... | 库存积压 | #1 | 0 | 1 | 5383ms |
| 6 | 库存周转模块中，门店周转排行支持的排序规则是什么？... | 库存周转 | #1 | 1 | 1 | 4885ms |
| 7 | 库存预警模块的SKU详情列表包含哪些近效期商品的指标字段？... | 库存预警 | #1 | 1 | 0.5 | 6908ms |
| 8 | 客单价模块列表区的同比数据标红规则是什么？... | 客单价 | #1 | 0 | 1 | 5394ms |
| 9 | 核心指标概览模块的卡片区包含哪些内容？... | 核心指标概览 | #1 | 0 | 1 | 4153ms |
| 10 | 库存周转模块筛选区支持哪些筛选维度？... | 库存周转 | #1 | 1 | 1 | 4284ms |
| 11 | 实际毛利率的变更记录中，S22迭代新增了哪些功能？... | 实际毛利率 | #1 | 1 | 1 | 3533ms |
| 12 | 库存预警模块包含哪些主要的分析专题？... | 库存预警 | #5 | 1 | 0 | 4700ms |
| 13 | 分类模块中，营销日的分析指标有哪些？... | 分类 | #1 | 1 | 0 | 3936ms |
| 14 | 库存积压模块包含哪些主要的功能区域？... | 库存积压 | #1 | 1 | 1 | 4702ms |
| 15 | 什么是公司的核心价值观？... | 无 | #- | 1 | 0 | 4478ms |
| 16 | 概览模块包含哪些核心功能区域？... | 概览 | #- | 1 | 0 | 5408ms |
| 17 | 管理单元模块中，如何查看不同层级的组织架构数据？... | 管理单元 | #1 | 0 | 1 | 5125ms |
| 18 | 库存缺断货模块的分析逻辑是什么？如何定义缺货和断货？... | 库存缺断货 | #- | 1 | 1 | 7053ms |
| 19 | 商品采购状态模块支持哪些采购状态类型的筛选？... | 商品采购状态 | #1 | 1 | 1 | 8576ms |
| 20 | 商品排行模块中，排行支持的指标维度有哪些？... | 商品排行 | #1 | 0 | 1 | 4815ms |
| 21 | 销售明细模块的数据粒度是什么？支持哪些维度的下钻？... | 销售明细 | #- | 1 | 0.5 | 8080ms |
| 22 | 自主核心商品模块中，核心商品的定义标准和筛选规则是什么？... | 自主核心商品 | #1 | 1 | 0.5 | 5792ms |
| 23 | 客单价与库存周转之间有什么业务关联关系？... | 客单价 | #- | 1 | 0 | 5097ms |
| 24 | 核心指标概览的筛选区的时间范围默认显示多长时间？... | 核心指标概览 | #2 | 1 | 0 | 5975ms |
| 25 | 库存积压和库存预警这两个模块有什么区别和联系？... | 库存积压 | #3 | 1 | 0.5 | 6642ms |
| 26 | 实际毛利率模块中，9级毛利区间的各级阈值是如何划分的？... | 实际毛利率 | #1 | 1 | 0 | 5902ms |
| 27 | 不存在的文档中有没有提到今年的销售目标？... | 无 | #- | 1 | 1 | 5460ms |
| 28 | 门店分类模块中，店龄维度的默认排序规则是什么？... | 分类 | #- | 1 | 0 | 4339ms |
| 29 | 概览模块中，销售金额卡片的达成图在目标值缺失时应如何展示？... | 概览 | #4 | 1 | 1 | 4904ms |
| 30 | 管理单元模块中，门店维度列表的行表头展示内容是什么？... | 管理单元 | #- | 1 | 0 | 3629ms |
| 31 | 客单价模块中，列表区的标红数据规则是什么？... | 客单价 | #1 | 0 | 1 | 4621ms |
| 32 | 商品采购状态模块中，新品卡片区的经营SKU数如何计算？... | 商品采购状态 | #1 | 1 | 1 | 4731ms |
| 33 | 商品排行模块中，SPU行业排行在分析维度为企业时的默认展示是... | 商品排行 | #1 | 1 | 1 | 2888ms |
| 34 | 实际毛利率模块中，2025年1月新增的商品类目选择功能如何影... | 实际毛利率 | #1 | 1 | 1 | 5716ms |
| 35 | 销售明细模块中，2026年1月新增的保质期剩余天数字段如何计... | 销售明细 | #1 | 0 | 1 | 5157ms |
| 36 | 自主核心商品模块中，SKU维度的列表展示哪些核心商品？... | 自主核心商品 | #1 | 1 | 0 | 4023ms |
| 37 | 门店分类模块中，营销日维度切换按钮的可用性受什么因素影响？... | 分类 | #1 | 1 | 0 | 4670ms |
| 38 | 概览模块中，毛利率卡片点击后的交互行为是什么？... | 概览 | #1 | 1 | 1 | 4713ms |
| 39 | 管理单元模块中，区域维度的异常提示在什么条件下触发？... | 管理单元 | #2 | 1 | 1 | 4751ms |
| 40 | 商品采购状态模块中，停采商品的界定标准是什么？... | 商品采购状态 | #1 | 0 | 1 | 4699ms |
| 41 | 商品排行模块中，SPU同质化分析在无销售数据时如何展示？... | 商品排行 | #1 | 1 | 1 | 4984ms |
| 42 | 实际毛利率模块中，全部行选中时图表展示的内容是什么？... | 实际毛利率 | #- | 1 | 0 | 4888ms |
| 43 | 销售明细模块中，筛选区的门店名称下拉框在什么情况下不显示？... | 销售明细 | #- | 1 | 0 | 4698ms |
| 44 | 自主核心商品模块中，概览区的销售金额达成率如何计算？... | 自主核心商品 | #1 | 1 | 1 | 5700ms |
| 45 | 门店分类模块中，达成差值的计算逻辑在2025年8月做了哪些优... | 分类 | #2 | 1 | 1 | 6299ms |
| 46 | 商品排行模块中，SPU商品明细列表的排序规则是什么？... | 商品排行 | #1 | 1 | 1 | 4379ms |
| 47 | 销售明细模块中，导出功能对商品类目字段的处理方式是什么？... | 销售明细 | #1 | 1 | 1 | 4441ms |

