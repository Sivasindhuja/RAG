# SATCOM NGP RAG

Grounded Retrieval-Augmented Generation (RAG) application for the Indian Satellite Communication (SATCOM) New Generation Policy (NGP) PDF (`data/satcom-ngp.pdf`). The system answers policy questions with citation-backed, evidence-grounded responses and explicit abstention when evidence is insufficient.

## What the application does

- Ingests and chunks the SATCOM NGP PDF into a ChromaDB vector index.
- Retrieves evidence using hybrid dense + BM25 retrieval with Reciprocal Rank Fusion (RRF).
- Generates grounded Gemini answers by default when Gemini is enabled/configured; extractive mode remains available. Gemini receives retrieved evidence only, citations remain app-resolved, and insufficient-evidence abstention/fallback behavior is preserved.
- Returns citations with source metadata/offsets, abstains on insufficient evidence, and exposes health/metrics/dashboard endpoints.
- Supports protected index rebuild via `POST /ingest` (disabled by default; token-protected when enabled).

## Architecture and request flow

```text
data/satcom-ngp.pdf
  -> extraction/metadata (src/rag_core/document.py)
  -> chunking (src/rag_core/chunker.py; production default: fixed-character 800/160)
  -> embeddings (sentence-transformers/all-MiniLM-L6-v2)
  -> ChromaDB index (src/rag_core/index.py, collection satcom_ngp_production_fixed_char_800_160)
  -> POST /query (backend/main.py -> src/rag_app/service.py)
  -> retrieval (src/rag_core/retrieval.py: dense or hybrid dense+BM25 RRF; optional Cohere rerank when configured)
  -> evidence selection/sufficiency/abstention (src/rag_core/pipeline.py)
  -> optional Gemini generation with validation/fallback (src/rag_core/generation.py)
  -> grounded response with citations, trace/timing metadata
```

**Key modules**
- `backend/main.py` — FastAPI API endpoints.
- `src/rag_app/` — app config, service composition, schemas, prompts, metrics, Langfuse tracing.
- `src/rag_core/` — canonical document/chunking/index/retrieval/pipeline/generation/evaluation core.
- `frontend/` — Vite/React UI.
- `evaluation/` — deterministic CI evaluation harness, golden dataset, baseline metrics.
- `docs/` — engineering log, project brief, deployment/evaluation/chunking/retrieval docs.

## Repository structure

```text
.
├── backend/                  # FastAPI app
├── src/
│   ├── rag_app/              # Active application layer
│   └── rag_core/             # Active canonical RAG core
├── frontend/                 # Vite/React frontend
├── prompts/                  # Versioned prompt templates/manifest
├── data/                     # Source PDF
├── evaluation/               # CI eval harness, dataset, baselines
├── scripts/                  # Utility scripts (ingest, evaluate)
├── tests/                    # Pytest suite
├── docs/                     # Project and engineering documentation
├── .github/workflows/ci.yml  # Consolidated CI workflow
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
├── requirements.runtime.txt
└── .env.example
```

## Local setup

### Prerequisites
- Python 3.11+
- Node.js 20+ (for frontend)
- A local virtual environment is recommended

### Backend

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt -r requirements.runtime.txt
cp .env.example .env  # fill required values
uvicorn backend.main:app --reload --port 8000
```

Build/verify index if needed:

```bash
python scripts/ingest.py
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

Set `VITE_API_BASE_URL` (default `http://localhost:8000`) for non-local backend targets.

## Environment variables

See `.env.example` for local defaults/placeholders. Key variables:

| Variable | Purpose | Default / note |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini generation (optional) | Unset disables Gemini use |
| `CO_API_KEY` / `COHERE_API_KEY` | Cohere rerank (optional) | Unset falls back to non-rerank path |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `LANGFUSE_ENABLED` | Langfuse tracing (optional) | No-op when not configured |
| `RAG_INGEST_TOKEN` | Protects `POST /ingest` via `X-Ingest-Token` | Unset => `/ingest` returns 403 fail-closed |
| `RAG_DEFINITION_INTENT_BOOST` | Opt-in definition-intent boost | `false` (default) |
| `RAG_CORS_ORIGINS` | Comma-separated allowed origins | Local dev defaults when unset; production must set actual frontend origin |
| `RAG_PERSIST_DIR`, `RAG_COLLECTION_NAME` | Chroma persistence/collection | Defaults match production config |
| `RAG_RETRIEVAL_MODE`, `RAG_GENERATION_MODE`, `RAG_TOP_K` | Runtime RAG behavior | `hybrid`, `gemini`, `5` |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM` | Conservative runtime threading | `1`, `1`, `false` recommended on constrained hosts |

## Running tests and CI evaluation

```bash
# Deterministic test suite (same CI exclusion)
PYTHONPATH=. python -m pytest tests -q -k "not real_index" --maxfail=5

# CI evaluation regression gate
PYTHONPATH=. python -m evaluation.ci_eval \
  --dataset evaluation/ci_golden_dataset.json \
  --baseline evaluation/baseline_metrics.json \
  --check-regression
```

## Render backend deployment

- Canonical backend deploy target: **Render** (see `render.yaml`).
- Health check: `GET /healthz` (liveness). `GET /health` is lightweight config/status and does not trigger model/index initialization.
- Build/start are defined in `render.yaml` (Python runtime, CPU torch install, `uvicorn backend.main:app`).
- Set required env vars/secrets in Render dashboard (do not commit secrets). Set `RAG_CORS_ORIGINS` to the actual Vercel frontend origin before public frontend integration.

## Vercel frontend deployment

- Canonical frontend deploy target: **Vercel** with project root `frontend/`.
- Frontend config: `frontend/vercel.json`.
- Set `VITE_API_BASE_URL` in Vercel to the deployed Render backend URL.
- After Vercel URL is known, set Render `RAG_CORS_ORIGINS` to that exact origin.

## Known deployment limitations

- **Render Free memory constraints:** Render Free (512MB) OOM has been confirmed for this workload. Lightweight health checks and conservative thread limits reduce avoidable overhead, but SentenceTransformer + Chroma runtime memory may still exceed 512MB. Treat Render Free as unproven/at-risk until validated with deployment evidence.
- **Cold starts / ephemeral filesystem:** Render free instances may spin down; Chroma persistence may be lost on ephemeral FS unless index is packaged/recovered appropriately. Index recovery path exists, but recovery can add startup cost.
- **Optional external services:** Gemini, Cohere rerank, and Langfuse are optional and depend on credentials/configuration; missing credentials should degrade gracefully.
- **RAG quality is evidence-based, not absolute:** deterministic CI eval/regression gates are used; see `docs/CI_EVALUATION.md`, `docs/RETRIEVAL_COMPARISON.md`, and `docs/ENGINEERING_LOG.md` for measured results and limitations.



## Isolated Weaviate Cloud experiment (Task 24, opt-in)

An isolated, opt-in Weaviate Cloud retrieval adapter exists for evaluation only. Default backend remains `chroma`; production Chroma/BM25/RRF path is unchanged. See `docs/RUN_LOCAL.md` and `docs/DEPLOYMENT.md` for env vars (names only), ingestion/eval commands, free-tier limitations, and disable instructions. No live Weaviate credentials were used and no deployment was performed as part of this task.
## Documentation

- `docs/PROJECT_BRIEF.md` — project brief and implementation status.
- `docs/ENGINEERING_LOG.md` — chronological engineering decisions/investigations/fixes.
- `docs/DEPLOYMENT.md` — deployment details and constraints.
- `docs/CI_EVALUATION.md` — CI evaluation and regression gate.
- `docs/CHUNKING_EXPERIMENT.md` — chunking selection evidence.
- `docs/RETRIEVAL_COMPARISON.md` — retrieval comparison evidence.
