# Run Local (WP1–WP3, extractive-first)

Validated SATCOM NGP RAG application over `data/satcom-ngp.pdf` using:
- corrected production fixed-character index: `chroma_db_production/` (`satcom_ngp_production_fixed_char`, 49 chunks)
- `src/rag_core` + `src/rag_app` (dense retrieval, EXP-003 selection default, statement-governing citations, explicit abstention, conservative selected-evidence answerability guard)
- FastAPI backend (`backend/main.py`, `/health`, `/query`)
- React/Vite frontend (`frontend/`, extractive default, inspectable citations/abstention/errors)
- optional Gemini generation via `src/rag_core/generation.py` (`GeminiGenerator`, m1-v1 contract) with app-side citation validation and safe fallback (**M1 not accepted for quality**)

## Prerequisites
- **Backend:** Python 3.11+ with project dependencies installed (`requirements.txt`), local embedding/model artifacts for `sentence-transformers/all-MiniLM-L6-v2`, production index present at `chroma_db_production/` (corrected fixed-char).
- **Frontend:** Node.js/npm; `frontend/package-lock.json` present (use npm). Vite 8 build verified with current lockfile.
- Optional for Gemini mode: `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env`/environment (not required for extractive default).

## Environment variables (optional)
- `RAG_PERSIST_DIR` (default `chroma_db_production`)
- `RAG_COLLECTION_NAME` (default `satcom_ngp_production_fixed_char`)
- `RAG_TOP_K` (default `5`)
- `RAG_GENERATION_MODE` (`extractive` default, or `gemini`)
- `RAG_GEMINI_MODEL` (default `gemini-flash-lite-latest`)
- `RAG_GEMINI_TEMPERATURE` (default `0.0`)
- `RAG_GEMINI_MAX_OUTPUT_TOKENS` (default `1024`)
- `RAG_ALLOW_GEMINI` (`true`/`false`)
- `RAG_CORS_ORIGINS` (comma-separated, default `*`)

Defaults keep EXP-003 selection and **no** experimental opt-ins (EXP-006/007/008 not promoted/combined).

## Start backend API
From repo root (ensure `PYTHONPATH` includes repo root if needed):
```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Verify: `GET http://127.0.0.1:8000/health` returns `status=ok`, `index_available=true`.

## Start frontend
`frontend/package-lock.json` is present; use reproducible install:
```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```
Production build:
```bash
cd frontend
npm ci
npm run build
npm run preview
```
Optional: set `VITE_API_BASE_URL` (default `http://localhost:8000`) before dev/build if backend runs elsewhere. CORS is enabled by backend (`RAG_CORS_ORIGINS`, default `*`).

## Endpoints
- `GET /` service info
- `GET /health` index/config/gemini-config status
- `POST /query` (preferred), `POST /ask` (compat alias)

Request:
```json
{ "question": "...", "mode": "extractive" }
```
`mode`: `extractive` (default, recommended for grounded traceability) or `gemini` (controlled; may fall back to extractive/abstain; requires API key and sufficient evidence).

Response (key fields): `request_id`, `status` (`answered|abstained|error`), `mode`, `answer`, `abstained`, `abstention_reason`, `citations[]` (app-resolved, inspectable: page/section/offset/chunk/source/quote), `statements[]` (verbatim selected evidence with offsets), `retrieved[]`, `selected_evidence_ids`, `citation_validation`, `generation` (attempt/fallback/errors/model/prompt metadata), `timings_ms`, `errors`, `debug`.

## Smoke checks (API)
```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question":"What satellite communication services are classified as domestic services?","mode":"extractive"}'
curl -s -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question":"Who is responsible for obtaining licenses to offer services in a territory?","mode":"extractive"}'
```

## Tests / validation
```bash
# Backend/API
python -m pytest tests/test_api.py -q
python -m pytest tests -q

# Frontend
cd frontend
npm ci
npm run build
```
`tests/test_api.py` includes fast contract tests (mocked) and a representative real-index smoke test (skips if production index missing). Real-index runs load the embedding model and may take longer.

## WP2/WP3 verification record (2026-11-15)
- **Frontend install/build:** `npm ci` (170 packages) succeeded; `npm run build` succeeded (Vite 8.1.3, `dist/assets/index-*.js` 256.36 kB, 223–275ms).
- **Running-app HTTP/CORS checks** (uvicorn + vite dev, browser-equivalent `fetch`/CORS from `http://127.0.0.1:5173`): `/health` ok/index available; CORS preflight allowed POST/content-type; **Q12** answered with inspectable citations/quotes/section-page-offset metadata (e.g. secs `4.1`, `4.3.4`, p13–14, chunk IDs/offsets present); **Q05** abstained with clear reason `insufficient_selected_evidence:selected_lexical_support=0.3333<threshold=0.5`, preserved retrieved evidence/statements/citations/guard debug (BL-001 retrieval miss remains); **normal supported question** answered with citations (e.g. secs `2.6.2`,`2.6.4`); **repeated requests** returned consistent answers with different request IDs; **backend unavailable** produced connection failure (curl exit 7 / fetch would error), matching frontend error-state handling.
- **Not visually automated:** in-browser DOM rendering, loading-spinner/disabled states, and manual click flows were not exercised via headless browser automation in this environment; frontend UI code implements these states and was served by vite dev.
- **Full validation:** `python -m pytest tests -q` → **108 passed, 33 warnings** (19.99s); `npm run build` → **success**.

## Known limitations (current defaults)
- M1 Gemini path **not accepted for quality**; extractive mode is the grounded baseline.
- **Answerability guard (application-level):** extractive and Gemini-precheck responses abstain when selected evidence is insufficient (`no selected statements` or `selected_lexical_support < RAG_GEMINI_SUFFICIENCY_THRESHOLD`, default 0.5). This is a conservative heuristic reusing existing sufficiency signals; it improves safety (e.g. Q05 abstains with reason despite retrieval miss) but does **not** guarantee semantic correctness/completeness and may still allow weak-but-above-threshold answers. Selected evidence/citations are preserved in abstained responses for inspection.
- BL-001 (Q05): required 2.5.6 evidence still **not retrieved** (retrieval blocker remains); service now abstains via guard instead of returning unsupported answer.
- BL-002 (Q08): incomplete retrieval/evidence.
- BL-003/004/005: opt-in candidates exist separately (EXP-006/008) but **not default/promoted/combined**.
- 15-question evaluation set only; not a general production-quality claim.
- Local Chroma + local model loading not validated for Vercel Hobby serverless (M3). No new experiments/retrieval/chunking/reranking added.

## Local observability (M5, 2026-11-15)
- Request/trace IDs and stage timings are returned in API responses for debugging (see `request_id`, `timings_ms`, `errors`/`debug.trace` in `/query` responses).
- **`GET /metrics`** returns a lightweight in-memory aggregated snapshot (totals, abstention/failure rates, answered citation-coverage proxy, p50/p95 latency for total/retrieval/selection/generation, recent records) — exportable JSON, no paid dependency. Metrics are per-process and reset on restart; cold-start spikes may affect early latency aggregates.
- See `docs/M5_OBSERVABILITY.md` for captured fields and limitations. External dashboards/CI gates remain follow-up work (BL-013/BL-012).

## Evaluation (M4 foundation, 2026-11-15)
- Reproducible offline baseline harness (extractive, no production behavior changes):
  ```bash
  PYTHONPATH=. python -m experiments.m4_eval_harness --out experiments/results/m4_baseline --mode extractive --k 5
  ```
- Outputs: `answerable_results.json`, `unanswerable_results.json`, `summary.json`, `summary.md`.
- See `docs/M4_EVALUATION_PLAN.md` and `docs/M4_PROGRESS.md`. Thresholds must be derived from measured (expanded) baselines, not set arbitrarily.
