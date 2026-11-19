# Retrieval Comparison (Dense vs Hybrid vs Hybrid+Rerank)

## Configuration (selected baseline)
- Chunking: fixed-character 800/160 (61 chunks), collection `satcom_ngp_production_fixed_char_800_160`
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Dense top-k: 5 final; dense candidates 20, BM25 candidates 20 for fusion
- Fusion: Reciprocal Rank Fusion (RRF, k=60)
- Reranker: Cohere `rerank-english-v3.0` (top_n=5 from fused candidates), integrated in `src/rag_core/retrieval.py`
- Evaluation set: CI golden dataset (15 questions: 12 answerable, 3 unanswerable), extractive grounded pipeline, `evaluation/ci_eval.py`

## Results (measured, 800/160)

### Dense vs Hybrid (measured)
Source: `evaluation/results/ci_compare_800_160/report.json`

| Mode | recall@5 | precision@5 | mrr@5 | evidence_coverage | supported_answer_rate | citation_support_rate | abstention_accuracy | invalid citations | p50 ms | p95 ms | failure_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.9236 | 0.3000 | 0.8611 | 0.9236 | 0.5833 | 0.4828 | 1.0 | 0 | 24.88 | 3010.97 | 0.0 |
| hybrid (dense+BM25 RRF) | 0.9236 | 0.3333 | 0.8472 | 0.9236 | 0.8333 | 0.5484 | 1.0 | 0 | 40.31 | 43.54 | 0.0 |

**Interpretation:** Hybrid maintained recall/evidence coverage and improved precision@5 and downstream grounded answer support substantially (supported answers 10/12 vs 7/12). MRR decreased slightly in aggregate, but hybrid produced better evidence sets for extractive selection on Q05,Q07,Q12,Q14 in this measured run. All unanswerable questions correctly abstained.

### Dense vs Hybrid vs Hybrid+Rerank (Cohere)
- **Mocked integration tests:** passed (`tests/test_cohere_reranker.py`) verifying rerank API integration and ordering behavior.
- **Live Cohere validation:** not completed because `COHERE_API_KEY`/`CO_API_KEY` was not configured in the environment (empty). Evaluation runs in `hybrid_rerank` mode fell back to hybrid candidates with recorded error `Cohere API key not configured`; aggregate metrics matched hybrid fallback and must **not** be interpreted as reranker quality evidence.
- **Required to complete live comparison:** configure valid Cohere API key and rerun `python -m evaluation.ci_eval --retrieval-modes dense,hybrid,hybrid_rerank ...`; record model name, rerank scores, latency, errors, billed units/usage/cost if available.

## Q05 and lexical/terminology cases
- With prior 1000/200 chunking, Q05 (`2.5.6`, "licenses to offer services in a territory") had dense recall@5 0.0 on CI set.
- Chunking experiment showed Q05 retrievable (recall 1.0) with 500/100, 800/160, 1200/240; selected 800/160 resolved this retrieval miss for dense and hybrid in measured runs.
- Hybrid vs dense comparison on 800/160: Q05 recall 1.0 in both; hybrid improved supported statement count (2/3 vs 1/3) for Q05. Q08 remains partially retrieved (1/3 evidence units) in both modes; Q10 evidence retrieved but extractive selection remains unsupported (known selector limitation).

## Decision
- **Hybrid retrieval (dense+BM25 with RRF)** adopted as default application retrieval mode based on measured improvement in grounded answer/citation support on the fixed CI set, with no loss in recall/evidence coverage and acceptable latency.
- **Cohere reranking** integrated and testable, but **not promoted as evidenced improvement** pending live credentialed evaluation.
