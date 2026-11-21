# Chunking Experiment (Controlled)

## Objective
Identify suitable fixed-size chunking with overlap baseline for `data/satcom-ngp.pdf` using measured retrieval performance and latency (not intuition).

## Setup (controlled)
- PDF: `data/satcom-ngp.pdf` (same source)
- Chunking family: fixed-character sliding window, per-page stripping/windowing (canonical `FixedCharacterChunker`)
- Configurations tested: 500/100, 800/160, 1000/200, 1200/240 (step = size-overlap)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (same)
- Retrieval: dense similarity search, top-k=5 (same)
- Evaluation questions: CI golden answerable subset (12 verified questions with evidence spans) from `evaluation/ci_golden_dataset.json`
- Metrics: recall@5, precision@5, MRR@5 (evidence-span overlap), mean retrieval latency; audit/integrity verified per config.

## Results (measured)
Source: `experiments/results/chunking_experiment/results.json`, `summary.md`

| chunk_size | overlap | step | chunks | recall@5 | precision@5 | mrr@5 | mean retrieval ms | audit |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | 100 | 400 | 93 | 0.8194 | 0.3 | 0.8194 | 20.651 | ok (100% coverage) |
| 800 | 160 | 640 | 61 | **0.9236** | **0.3** | **0.8611** | 19.317 | ok (100% coverage) |
| 1000 | 200 | 800 | 49 | 0.8611 | 0.2667 | 0.743 | 22.53 | ok (100% coverage) |
| 1200 | 240 | 960 | 43 | 0.8403 | 0.25 | 0.8333 | 19.359 | ok (100% coverage) |

Per-question highlights: Q05 (`2.5.6`, lexical/terminology-sensitive) retrieved with recall 1.0 at 500/100, 800/160, 1200/240 but **0.0 at 1000/200**; Q08 remained partial (0.3333) across tested configs.

## Selection
- **Selected:** fixed-character `chunk_size=800`, `chunk_overlap=160` (step 640), 61 chunks.
- **Rationale:** highest recall@5 and MRR@5 on the fixed evaluation subset with identical embedding/retrieval settings; precision@5 tied-best; audit passed with complete non-whitespace coverage.
- **Consistency:** selected config is the application default (`src/rag_app/config.py`), used by ingestion/index rebuild (`scripts/ingest.py`, `POST /ingest`) and query-time retrieval against collection `satcom_ngp_production_fixed_char_800_160`.

## Impact / reassessment
- Historical accepted baseline EXP-001/PROD-001 used 1000/200 (49 chunks). Chunking experiment provided controlled evidence that 800/160 improves retrieval on the CI set (including Q05 retrievability). Final configuration updated accordingly and documented; historical artifacts/results preserved unchanged.
- CI dense baseline re-measured on 800/160: recall@5 0.9236, mrr@5 0.8611, evidence coverage 0.9236 (see `docs/CI_EVALUATION.md`).
