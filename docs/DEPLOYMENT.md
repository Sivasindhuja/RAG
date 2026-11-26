# Deployment

## Target and constraints (Vercel Hobby)
- **Target:** publicly accessible application compatible with Vercel Hobby plan (per project constraints).
- **Observed constraints:** serverless execution limits, cold starts, function bundle size limits, no persistent local filesystem across invocations, request/response and timeout limits, Python runtime constraints, and secret management requirements.
- **Repository reality:** backend depends on `sentence-transformers`, `torch` (transitively), `chromadb`, `pypdf`, `rank_bm25`, etc., and loads a local embedding model (`all-MiniLM-L6-v2`) plus a persistent Chroma index. This is a poor fit for Vercel Hobby serverless functions (large dependencies/model artifacts, cold-start model load, non-persistent local storage).

## Implemented deployment artifacts (best-effort Vercel compatibility)
- Frontend: Vite app (`frontend/`) configurable via `VITE_API_BASE_URL`.
- Canonical deployment (Task 38): Render Free backend (`render.yaml`, Weaviate production runtime) + Vercel frontend (`frontend/vercel.json`, Vercel project root = `frontend`). Legacy root Vercel serverless artifacts were removed during Task 16 cleanup.
- Backend CORS configurable via `RAG_CORS_ORIGINS` (comma-separated explicit origins; local-dev defaults when unset; explicit `*` only when intentionally set; production must set actual Vercel origin).

These artifacts enable deployment attempts/local Vercel-compatible structure but **do not by themselves guarantee a working Hobby deployment** given dependency/model/storage constraints.

## Persistence strategy (required for serverless)
Local Chroma persistence (`chroma_db_production`) and local model loading cannot be assumed in serverless. Options and trade-offs:
1. **Bundled read-only index artifact + model download/cache:** include prebuilt Chroma SQLite/index files in deployment artifact and allow model download on cold start. Trade-offs: large bundle/cold start, ephemeral cache, possible size limit violations on Hobby. Cost: free-tier attempt possible but likely infeasible.
2. **External persistent vector store / hosted retrieval service (free tier where possible):** e.g. hosted Chroma-compatible or alternative free-tier vector DB/API; keep embeddings/index there. Trade-offs: network latency, auth/secrets, service limits, operational complexity. No paid infra added without approval.
3. **Selected production path (Task 38, Render Free 512 MB constraint):** FastAPI backend on Render Free with Weaviate Cloud for vector storage + hosted embeddings; React/Vite frontend on Vercel. This avoids local torch/sentence-transformers/Chroma runtime in the production image but requires actual Render memory/runtime verification. Trade-offs: cold starts/spin-down on Render Free, external Weaviate dependency/quota, retrieval-quality trade-off vs historical Chroma baseline. If Render Free OOMs, record evidence before choosing another zero-cost host.

**Selected documented strategy:** for Hobby-compatible public deployment, prefer **Vercel frontend + separate free-tier backend host** if a verified free-tier host is available; otherwise document blocker and keep local/public-source deployment reproducible. No paid infrastructure incurred.

## Configuration and secrets
- Backend env (Weaviate production): see Render checklist below. Key vars: `RAG_RETRIEVER_BACKEND=weaviate`, `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION=SatcomNgpExp800160`, `WEAVIATE_EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0`, `WEAVIATE_INIT_TIMEOUT_S=10`, `WEAVIATE_QUERY_TIMEOUT_S=30`, `RAG_GENERATION_MODE`, `RAG_GEMINI_MODEL`, `GEMINI_API_KEY`/`GOOGLE_API_KEY` (optional), `RAG_CORS_ORIGINS` (actual Vercel origin), `RAG_INGEST_TOKEN` (keep unset; fail-closed). Chroma/local path retains prior env vars when explicitly selected.
- **CORS (`RAG_CORS_ORIGINS`):** comma-separated explicit origins. Default when unset/empty is local dev origins (`http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000`), **not** wildcard. Explicit `*` is supported only when intentionally configured. For Render+Vercel public deployment, set the **actual Vercel frontend origin** (e.g. `https://<your-app>.vercel.app`) before frontend integration; do not leave wildcard/placeholder in production.
- Frontend env: `VITE_API_BASE_URL`.
- Never commit or expose secrets.

## Ingestion/index management
- Reproducible build: `python scripts/ingest.py` (local/CLI) or protected `POST /ingest` (HTTP, disabled by default).
- **Security (2026-11-21):** `POST /ingest` is **disabled by default** and fails closed. To enable authorized HTTP ingestion, set `RAG_INGEST_TOKEN` and send header `X-Ingest-Token: <token>`. Unauthorized/missing-token requests return 403 and do not trigger ingestion. Internal automatic index recovery/re-ingestion in `src/rag_app/service.py` remains internal behavior and is not exposed as an unauthenticated public endpoint.
- Current selected index config (measured): fixed-character 800/160, 61 chunks, collection `satcom_ngp_production_fixed_char_800_160`, embedding `sentence-transformers/all-MiniLM-L6-v2`, source `data/satcom-ngp.pdf`.

## Public deployment status
- **Public URL:** not completed in this environment (no verified free-tier backend hosting/account provisioning and no Vercel deployment credentials/access executed here). This is an external-access blocker.
- **Precise remaining action:** (1) choose and verify a genuinely free-tier backend host with sufficient RAM/disk for model+index and outbound API access; (2) provision and configure secrets/storage; (3) deploy backend (`uvicorn backend.main:app`) with persistent index artifact and env vars; (4) deploy frontend to Vercel with `VITE_API_BASE_URL` pointing to backend; (5) run public smoke tests (`/health`, `/query`, dashboard) and record URL/results.
- All application code, ingestion, evaluation, CI, tracing integration, docs, and local validation are completed to the extent possible without external account/provisioning.

## Smoke tests (local/public)
- `GET /` service info
- `GET /health` lightweight config/gemini/langfuse status; does **not** trigger SentenceTransformer load, Chroma init, index recovery, or embedding generation. `index_status` is `unknown`/`available`/`unavailable`; `index_available` is true only when index already initialized and available, false otherwise; `index_initialized` reports whether deep index state was already initialized in-process.
- `GET /metrics` live operational metrics
- `GET /dashboard` project dashboard (live + offline artifacts)
- `POST /query` extractive grounded answer with citations/abstention
- `POST /ingest` protected index rebuild (controlled environment; requires `RAG_INGEST_TOKEN` + `X-Ingest-Token`; expect 403 when disabled/unauthorized)

---

## Production migration: Weaviate selected for Render Free 512 MB (Task 38, prepared; not deployed)

### Architecture decision (recorded)

> Chroma produced better results in some offline quality metrics. Weaviate was selected for production because the available Render Free backend has a 512 MB memory limit, and moving vector storage and hosted embedding outside the application is necessary to pursue deployment within that constraint. Retrieval quality is an acknowledged trade-off to be improved through targeted post-deployment experiments.

**Production target:**
- FastAPI backend on Render Free (512 MB) using `requirements.weaviate.runtime.txt` (no torch/sentence-transformers/chromadb/rank_bm25).
- Weaviate Cloud for vector storage/retrieval with existing collection `SatcomNgpExp800160` and hosted embedding `Snowflake/snowflake-arctic-embed-l-v2.0`.
- Gemini for generation (existing config preserved).
- React/Vite frontend on Vercel; `VITE_API_BASE_URL` must be set at build time (no localhost fallback in production build).

**Chroma status:** existing Chroma implementation, local `chroma_db_production/`, and historical evaluation results are preserved for offline comparison. Chroma is not the selected production retrieval backend for this Render deployment path. Do not delete/overwrite Chroma artifacts.

### Render Free constraints (official docs, https://render.com/free, fetched 2026-11-19; recheck before deploy)
- 750 free instance hours/month; service spins down after 15 minutes idle; spin-up on request.
- Ephemeral filesystem; free web services cannot attach persistent disks.
- 512 MB RAM / limited CPU on free instances; OOM is a deployment risk and must be measured by actual deployment.

### Reliability adaptations (Task 38)
- **Weaviate production runtime split:** `render.yaml` uses `requirements.weaviate.runtime.txt`; Dockerfile installs lightweight Weaviate runtime deps only. Heavy local model/index deps excluded from this image.
- **Lightweight health checks:** `/healthz` liveness; `/health` remains lightweight and does not trigger SentenceTransformer/Chroma/index recovery/embedding generation. Weaviate backend reports local index as not initialized/unknown.
- **No silent fallback:** when `RAG_RETRIEVER_BACKEND=weaviate` is selected and Weaviate is unavailable/misconfigured, service fails clearly; it does not fall back to Chroma.
- **Frontend production guard:** `frontend/src/App.jsx` uses localhost only in dev; production build requires `VITE_API_BASE_URL`.

### Render environment-variable checklist (set in Render dashboard; do not commit secrets)

| Variable | Required | Value / note |
|---|---:|---|
| `RAG_RETRIEVER_BACKEND` | yes | `weaviate` |
| `WEAVIATE_URL` | yes (secret) | Existing Weaviate Cloud endpoint (env var only) |
| `WEAVIATE_API_KEY` | yes (secret) | Existing Weaviate Cloud API key (env var only) |
| `WEAVIATE_COLLECTION` | yes | `SatcomNgpExp800160` (existing, 61 objects; no re-ingest) |
| `WEAVIATE_EMBEDDING_MODEL` | yes | `Snowflake/snowflake-arctic-embed-l-v2.0` |
| `WEAVIATE_INIT_TIMEOUT_S` | yes | `10` |
| `WEAVIATE_QUERY_TIMEOUT_S` | yes | `30` |
| `RAG_RETRIEVAL_MODE` | yes | `hybrid` (app-level mode label; Weaviate retriever uses configured Weaviate search mode/alpha defaults) |
| `RAG_GENERATION_MODE` | yes | `extractive` (default; Gemini optional) |
| `RAG_ALLOW_GEMINI` | optional | `true` if Gemini enabled |
| `RAG_GEMINI_MODEL` | optional | `gemini-flash-lite-latest` |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | optional secret | Required only for Gemini generation path |
| `RAG_CORS_ORIGINS` | yes before frontend integration | Actual Vercel frontend origin, e.g. `https://<app>.vercel.app` (no wildcard/placeholder in production) |
| `LANGFUSE_*`, `LANGFUSE_ENABLED` | optional | Observability; keep disabled if not needed |
| `RAG_INGEST_TOKEN` | optional secret | Keep unset/disabled; `/ingest` is Chroma-oriented and fail-closed by default |
| `RAG_DEFINITION_INTENT_BOOST` | optional | `false` (default) |

### Preflight (Task 38, read-only, no secrets exposed)
- Weaviate Cloud connection: ready.
- Collection `SatcomNgpExp800160`: exists.
- Object count: 61.
- Read-only smoke retrieve: 1 hit with text.
- No collection creation/deletion/reset/re-ingestion performed.

### Exact deployment steps (do not execute without explicit approval)
1. Confirm Render/Vercel account access and GitHub repo/branch access. Preserve all uncommitted local work; do not use `git add .`.
2. Push reviewed deployment-ready code to a branch/repo accessible by Render/Vercel.
3. Create/connect Render service from `render.yaml` (or equivalent dashboard config). Confirm build uses `requirements.weaviate.runtime.txt`.
4. Set Render env vars/secrets from checklist above. Set `RAG_CORS_ORIGINS` to actual Vercel origin only after known; keep strict.
5. Deploy Render backend and capture backend URL. Verify `/healthz` returns 200, `/health` returns 200 without loading local embedding models, `/query` returns grounded/abstained response, `/metrics` and `/dashboard` respond, `/ingest` remains 403 fail-closed when token unset.
6. Deploy Vercel frontend (project root `frontend`) with `VITE_API_BASE_URL=<render-backend-url>`. Verify no localhost fallback in production build.
7. Run public smoke tests: cold start, query/citations/abstention/error flow, CORS, restart/redeploy behavior. Capture Render logs/metrics for OOM or memory pressure.
8. If Render Free 512 MB OOMs, record evidence and stop; do not claim fit. Consider separate zero-cost host only after evidence.

### Weaviate experiment/config notes (historical + current)
- Experimental Weaviate adapter/eval scripts remain in-repo. Historical results preserved (Tasks 24-37).
- Task 36 experimental abstention policy remains disabled by default (`weaviate_abstention_experimental=false`).
- No Weaviate score threshold calibrated; do not enable experimental abstention threshold in production without separate approval and evidence.


---

## Task 39 — Final Read-Only Deployment Readiness Checklist (2026-11-26, docs-only)

**Status:** Read-only audit completed; no application code modified; no deploy performed. Distinguish confirmed facts, measured results, inferences, unverified assumptions, and pending deployment checks.

### Confirmed facts (code/config inspection)
- `RAG_RETRIEVER_BACKEND=weaviate` selects Weaviate path; missing Weaviate config fails clearly; no silent fallback to Chroma.
- `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION`, `WEAVIATE_EMBEDDING_MODEL`, `WEAVIATE_INIT_TIMEOUT_S`, `WEAVIATE_QUERY_TIMEOUT_S` are wired through AppConfig/service/retriever.
- Render/Docker production path uses `requirements.weaviate.runtime.txt` (no torch/sentence-transformers/transformers/chromadb/rank_bm25 package entries).
- `/healthz` lightweight; `/health` lightweight (no SentenceTransformer/Chroma init/recovery/embedding generation).
- CORS defaults to local dev origins when unset; explicit `*` only when configured; `RAG_CORS_ORIGINS` must be set to actual Vercel origin for production.
- `/ingest` fail-closed when `RAG_INGEST_TOKEN` unset; keep disabled in Weaviate production.
- Frontend production build requires `VITE_API_BASE_URL`; localhost fallback only in dev.

### Measured results (this audit)
- Weaviate preflight (Task 38, read-only): ready, collection exists, 61 objects, smoke retrieve returned 1 hit with text.
- Tests: `pytest tests/test_weaviate_retriever.py tests/test_api.py -q` => 63 passed, 1 failed, 4 warnings. Failure is environment-sensitive (`.env` present) for `test_missing_weaviate_credentials_fail_clearly`; passes with Weaviate env vars cleared. Warnings are existing Pydantic/Chroma deprecations.

### Gemini generation-mode (Task 43, approved behavior implemented)
- `RAG_GENERATION_MODE=gemini` = default response mode is Gemini for normal user requests.
- Gemini receives only the user question and selected retrieved evidence (grounded prompt contract); citations are app-resolved against retrieved chunks.
- Insufficient evidence results in abstention (existing guard/pipeline behavior preserved).
- `RAG_ALLOW_GEMINI=false` or Gemini unavailable/error: system fails safely with explicit fallback to grounded extractive (`generation.fallback_to_extractive=true`, explicit generation status/errors). No ungrounded/fabricated answer is presented.
- Explicit `mode=extractive` requests remain supported.

### Unverified assumptions / pending deployment checks
- Render Free 512 MB RSS/cold-start peak not measured.
- Weaviate gRPC/REST behavior under Render networking not measured.
- Actual Vercel origin/CORS integration not tested.
- Gemini live availability/quota/model behavior not tested in this audit.
- docker-compose.yml is legacy/dev-oriented; not canonical for Render.

### Before publishing
- [ ] Review all dirty/untracked files; identify exact deployment-intended files. Do not use `git add .`.
- [ ] Preserve unrelated uncommitted work and local data (`.agents/`, `evaluation/results/`, `chroma_db_*`, `.env`, etc.).
- [ ] Confirm env var names/values in Render dashboard match checklist; never commit secrets.
- [x] Owner decision (Task 43): Gemini is default generation mode for normal requests; extractive remains available on explicit request; /ingest disabled.

### Render deployment
- [ ] Configure secrets: `WEAVIATE_URL`, `WEAVIATE_API_KEY`, optional `GEMINI_API_KEY`/`GOOGLE_API_KEY`, optional Langfuse, keep `RAG_INGEST_TOKEN` unset.
- [ ] Configure non-secrets: `RAG_RETRIEVER_BACKEND=weaviate`, `WEAVIATE_COLLECTION=SatcomNgpExp800160`, `WEAVIATE_EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0`, `WEAVIATE_INIT_TIMEOUT_S=10`, `WEAVIATE_QUERY_TIMEOUT_S=30`, `RAG_GENERATION_MODE=gemini`, `RAG_ALLOW_GEMINI=true`, `RAG_GEMINI_MODEL=gemini-flash-lite-latest`, `RAG_DEFINITION_INTENT_BOOST=false`, `LANGFUSE_ENABLED=false` (unless needed).
- [ ] Set `RAG_CORS_ORIGINS` to actual Vercel origin before frontend integration.
- [ ] Build/deploy from `render.yaml` / Dockerfile. Expected: build completes with `requirements.weaviate.runtime.txt`.
- [ ] Verify `/healthz` returns 200. Failure: check startup logs/port binding.
- [ ] Verify `/health` returns 200 lightweight status. Failure: check Weaviate config/import errors; do not interpret as index-ready in Weaviate mode.
- [ ] Test real `/query` (default Gemini). Expected: grounded Gemini answer with app-resolved citations or explicit abstention. If Gemini disabled/unavailable, expect explicit grounded-extractive fallback (`generation.fallback_to_extractive=true`) or safe abstention/error per config; never ungrounded answer. Also verify explicit `mode=extractive` still works.
- [ ] Measure memory/CPU in Render metrics/logs. If OOM >512 MB, record evidence and stop; do not claim fit.

### Vercel deployment
- [ ] Set `VITE_API_BASE_URL` to Render backend URL at build time.
- [ ] Build/deploy frontend (root `frontend`). Expected: no localhost fallback in production build.
- [ ] Verify frontend loads and can reach backend. Failure: check VITE_API_BASE_URL, CORS, HTTPS/mixed content.

### Public acceptance testing
- [ ] Grounded answer: ask in-scope question; expect answer + citations.
- [ ] Valid citations: citations resolve to retrieved/source metadata; no invalid source IDs.
- [ ] Appropriate abstention: ask out-of-scope/unanswerable question; expect explicit abstention.
- [ ] Invalid request handling: empty/malformed request returns clear 4xx/handled error.
- [ ] Backend unavailable handling: frontend shows clear error when backend down.
- [ ] Cold start: first request after spin-down succeeds within acceptable timeout or shows clear loading/error; record latency.
- [ ] Weaviate timeout behavior: observe logs/response on timeout; no silent fallback to Chroma.
- [ ] Memory measurement: capture Render metrics during cold start and query load.

### Deployment blockers (current)
- No explicit deploy approval.
- Render/Vercel account access/credentials unavailable in this environment.
- Actual Vercel production origin unknown.
- Render Free 512 MB fit unproven by deployment measurement.
- Weaviate gRPC reliability under Render networking unproven.

### Post-deployment quality risks (non-blockers)
- Retrieval-quality trade-off vs Chroma (Task 34).
- UNANS-11 abstention regression in BM25/hybrid (Tasks 34/35).
- Q07/Q12/Q14 citation-selection issues; Q08 multi-evidence coverage gap.
- Task 36 experimental abstention disabled/un-calibrated; do not enable in production without evidence.


## Task 40 — Publish Manifest Review (2026-11-26, manifest-only)

**Status:** No files staged/committed/pushed/deployed. Dirty worktree contains pre-existing unrelated changes; publish must use an exact reviewed allowlist only.

### Proposed exact publish allowlist (dirty/untracked candidates only)

| File | Action | Why |
|---|---|---|
| `.dockerignore` | INCLUDE | Container build excludes; review diff for unrelated changes. |
| `.gitignore` | INCLUDE | Repo hygiene; review diff for unrelated changes. |
| `Dockerfile` | INCLUDE | Weaviate Render production image. |
| `docker-compose.yml` | INCLUDE (review) | Dev compose; review for unintended drift before publish. |
| `render.yaml` | INCLUDE | Render Weaviate production blueprint. |
| `requirements.weaviate.runtime.txt` | INCLUDE | Minimal Weaviate runtime deps. |
| `frontend/src/App.jsx` | INCLUDE | Vercel production API-base guard. |
| `frontend/.env.example` | INCLUDE | Frontend build-time API URL example. |
| `src/rag_app/config.py` | INCLUDE | Weaviate/Gemini/config wiring. |
| `src/rag_app/schemas.py` | INCLUDE | API/response schema compatibility. |
| `src/rag_app/service.py` | INCLUDE | Weaviate routing, no silent fallback, lightweight health. |
| `src/rag_core/__init__.py` | INCLUDE | Lazy exports for lightweight Weaviate path. |
| `src/rag_core/retrieval_types.py` | INCLUDE | Shared lightweight retrieval types. |
| `src/rag_core/weaviate_retriever.py` | INCLUDE | Weaviate Cloud retriever/timeouts/modes. |
| `src/rag_core/pipeline.py` | INCLUDE | Grounded answer/abstention pipeline (Task 36 opt-in only). |
| `src/rag_core/generation.py` | INCLUDE (if dirty/required) | Gemini/extractive generation behavior; verify diff if present. |
| `src/rag_core/index.py` | INCLUDE (review) | Chroma/index support and lazy behavior; review diff carefully. |
| `src/rag_core/retrieval.py` | INCLUDE (review) | Chroma hybrid retrieval; review diff carefully. |
| `src/rag_core/evaluation.py` | INCLUDE (review) | Evaluation support used by tests/diagnostics; review diff. |
| `tests/test_api.py` | INCLUDE-TEST | API regression coverage. |
| `tests/test_weaviate_retriever.py` | INCLUDE-TEST | Weaviate retriever/import/safety/abstention tests. |
| `tests/test_citation_metadata.py` | INCLUDE-TEST | Citation regression. |
| `tests/test_fixed_chunker.py` | INCLUDE-TEST | Chunking regression. |
| `tests/test_rag_evaluation.py` | INCLUDE-TEST | Eval regression. |
| `tests/fixtures/exp001/*` | INCLUDE-TEST | Test fixtures; preserve. |
| `README.md` | INCLUDE-DOCS (review) | Review large diff for unrelated changes. |
| `docs/DEPLOYMENT.md` | INCLUDE-DOCS | Deployment checklist/manifest. |
| `docs/RUN_LOCAL.md` | INCLUDE-DOCS | Local run instructions. |
| `docs/ENGINEERING_LOG.md` | INCLUDE-DOCS | Audit trail.

### Explicitly excluded / owner review

- **EXCLUDE-LOCAL:** `.env`, `.agents/`, `chroma_db_production/`, `frontend/node_modules/`, `frontend/dist/`, caches, local generated data; `todo.txt` deletion preserved.
- **EXCLUDE-HISTORICAL:** `evaluation/results/`, `experiments/results/`, `requirements.weaviate.runtime.proposed.txt`, other generated artifacts.
- **REVIEW:** `evaluation/weaviate_eval.py`, `scripts/weaviate_ingest.py`, `requirements.weaviate.txt`, `evaluation/drafts/*`, all pre-existing deletions and unrelated dirty docs/src/tests from earlier tasks. Do not stage by default.

### Pre-publish verification

- Confirm allowlist file-by-file; stage individually (no `git add .`).
- Run: `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py tests/test_api.py -q` (known env-sensitive failure possible when `.env` supplies Weaviate creds: `test_missing_weaviate_credentials_fail_clearly`).
- Confirm no secrets in staged diff. Confirm Vercel origin and Render secrets are dashboard-only.
- Confirm `/ingest` remains disabled (`RAG_INGEST_TOKEN` unset).


## Task 41 — Resolved Publish Manifest (2026-11-26, manifest-only)

**Status:** No files staged/committed/pushed/deployed. This resolves Task 40 REVIEW items for the smallest complete Weaviate production publish (Render backend + Vercel frontend).

### Exact proposed publish allowlist (no unresolved REVIEW entries)

| File | Action | Why |
|---|---|---|
| `.dockerignore` | INCLUDE | Build-context hygiene; diff reviewed (adds chroma exp dirs/eval results ignores). |
| `.gitignore` | INCLUDE | Repo hygiene; diff reviewed (ignores env variants, caches, build outputs, eval results). |
| `.env.example` | INCLUDE | Template only, no secrets; adds Weaviate/CORS/ingest placeholders. |
| `Dockerfile` | INCLUDE | Weaviate lightweight Render image. |
| `render.yaml` | INCLUDE | Render Free Weaviate blueprint; CORS placeholder remains owner action. |
| `docker-compose.yml` | INCLUDE | Dev parity; adds conservative thread env vars only. |
| `requirements.weaviate.runtime.txt` | INCLUDE | Minimal Weaviate runtime deps. |
| `frontend/src/App.jsx` | INCLUDE | Production `VITE_API_BASE_URL` guard. |
| `frontend/.env.example` | INCLUDE | Frontend build-time API URL template. |
| `src/rag_app/config.py` | INCLUDE | Weaviate/generation/timeout config wiring. |
| `src/rag_app/schemas.py` | INCLUDE | API schema compatibility. |
| `src/rag_app/service.py` | INCLUDE | Weaviate routing, no silent fallback, lightweight health. |
| `src/rag_core/__init__.py` | INCLUDE | Lazy exports for lightweight path. |
| `src/rag_core/retrieval_types.py` | INCLUDE | Shared lightweight retrieval types. |
| `src/rag_core/weaviate_retriever.py` | INCLUDE | Weaviate Cloud retriever. |
| `src/rag_core/pipeline.py` | INCLUDE | Grounded answer/abstention; Task 36 opt-in default-off. |
| `src/rag_core/index.py` | INCLUDE | Lazy chromadb/type refactor; required by Chroma/index support/tests. |
| `src/rag_core/retrieval.py` | INCLUDE | Retrieval support; reviewed diff relevant to shared retrieval behavior. |
| `src/rag_core/evaluation.py` | INCLUDE | Evaluation/type import used by tests/diagnostics. |
| `tests/test_api.py` | INCLUDE-TEST | API regressions incl. lightweight health/ingest. |
| `tests/test_weaviate_retriever.py` | INCLUDE-TEST | Weaviate regressions (env-sensitive failure caveat below). |
| `tests/test_citation_metadata.py` | INCLUDE-TEST | Citation regressions. |
| `tests/test_fixed_chunker.py` | INCLUDE-TEST | Chunking regressions. |
| `tests/test_rag_evaluation.py` | INCLUDE-TEST | Evaluation regressions. |
| `tests/fixtures/exp001/chunks.json` | INCLUDE-TEST | Required fixture. |
| `tests/fixtures/exp001/exp001_ground_truth.json` | INCLUDE-TEST | Required fixture. |
| `tests/fixtures/exp001/exp001_questions.csv` | INCLUDE-TEST | Required fixture. |
| `README.md` | INCLUDE-DOCS | Project/deploy documentation; reviewed as doc-only. |
| `docs/DEPLOYMENT.md` | INCLUDE-DOCS | Deployment manifest/checklist. |
| `docs/RUN_LOCAL.md` | INCLUDE-DOCS | Run/test instructions including Weaviate/eval commands. |
| `docs/ENGINEERING_LOG.md` | INCLUDE-DOCS | Audit trail. |

**Must-include groups:** Dockerfile+render.yaml+requirements.weaviate.runtime.txt; config+service+weaviate_retriever+retrieval_types+pipeline; frontend App.jsx+frontend/.env.example; tests+fixtures with code they validate.

### Explicit exclusions (resolved)

| File/group | Action | Why |
|---|---|---|
| `.env`, `.agents/**`, `chroma_db*`, caches/build outputs, `todo.txt` deletion | EXCLUDE-LOCAL | Local/secret/generated; preserve locally. |
| `evaluation/drafts/*`, `requirements.weaviate.runtime.proposed.txt`, `experiments/**` deletions, `evaluation/results/**` | EXCLUDE-HISTORICAL | Historical/draft/generated artifacts, not runtime. |
| `evaluation/weaviate_eval.py`, `scripts/weaviate_ingest.py`, `requirements.weaviate.txt` | EXCLUDE | Useful ops/eval tooling, not required for minimal Render/Vercel runtime. Stage only in separate owner-approved ops PR if desired. |
| `evaluation/ci_eval.py` modified | EXCLUDE | Not runtime; avoid mixing CI/eval change into production publish. |
| Dirty docs not listed above (`docs/BACKLOG.md`, `CHUNKING_STRATEGY.md`, `CI_EVALUATION.md`, `EXP-*`, `M4_*`, `PRODUCTION_SLICE.md`, `PROJECT_BRIEF.md`) | EXCLUDE | Not required for production publish; keep for separate docs cleanup review. |
| Legacy deletions: `.github/workflows/rag_eval.yml`, `api/index.py`, `src/chunking/*`, `src/config/*`, `src/generation/models.py`, `src/memory/*`, `src/metrics/*`, `src/observability/*`, `src/restart.py`, `src/retrieval/*`, `tests/eval_pipeline.py`, `tests/golden_dataset.csv`, `tests/test_exp001.py` | EXCLUDE | Pre-existing cleanup; no runtime imports found in backend/src/rag_app/src/rag_core/scripts/evaluation. Staging deletions with production publish would broaden scope; owner may stage in separate cleanup PR. |

### Known test caveat (unchanged)

- `tests/test_weaviate_retriever.py::test_missing_weaviate_credentials_fail_clearly` fails when workspace `.env` provides Weaviate credentials (`WEAVIATE_URL/WEAVIATE_API_KEY/WEAVIATE_COLLECTION` set). Test calls `WeaviateRetrieverConfig.from_env()` without env isolation; env vars satisfy config so no `WeaviateConfigError` raised.
- **Proposed minimal fix (not implemented):** add `monkeypatch` parameter and clear Weaviate env vars (`WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION`, `RAG_WEAVIATE_URL`, `RAG_WEAVIATE_API_KEY`, `RAG_WEAVIATE_COLLECTION`) before calling `from_env()`. Do not alter assertions/behavior beyond isolation.

### Blockers / owner decisions

- Approve exact file-by-file staging of allowlist (no `git add .`).
- Set actual Vercel origin in Render `RAG_CORS_ORIGINS`.
- Provide Render/Vercel access and dashboard secrets (no secrets in repo).
- Confirm Gemini default-mode expectation (extractive default, Gemini only on explicit request).
- Keep `RAG_INGEST_TOKEN` unset in Weaviate production.
- Render Free 512MB fit and Weaviate gRPC reliability remain unverified until deployment smoke/memory evidence.


## Task 42 — Test Isolation and Manifest Refinement (2026-11-26, no-stage/no-deploy)

**Status:** No files staged/committed/pushed/deployed.

### Test isolation fix
- `tests/test_weaviate_retriever.py::test_missing_weaviate_credentials_fail_clearly`: added `monkeypatch` and cleared `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION`, `RAG_WEAVIATE_URL`, `RAG_WEAVIATE_API_KEY`, `RAG_WEAVIATE_COLLECTION` before calling `WeaviateRetrieverConfig.from_env()`. Assertion unchanged. No production behavior change. No credential values exposed.

### Validation
- Command: `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py tests/test_api.py -q`
- Result: **64 passed, 0 failed, 0 skipped, 4 warnings** (62.71s). Warnings: PydanticDeprecatedSince211 from `chromadb/types.py` in real-index smoke test.

### Manifest refinement
- **Removed from minimal production allowlist:** `docker-compose.yml` — local dev convenience only; no concrete Render/Vercel production dependency demonstrated.
- **Inspected diffs (kept):**
  - `src/rag_core/index.py` — moves `RetrievedChunk` to `retrieval_types`, lazy-imports `chromadb`; necessary for lightweight import boundary/shared types used by Weaviate and Chroma paths. Keep.
  - `src/rag_core/retrieval.py` — moves retrieval types to `retrieval_types`, lazy-imports `rank_bm25`/BM25Okapi and TYPE_CHECKING DenseIndex; necessary for lightweight boundary/shared types. Keep.
  - `src/rag_core/evaluation.py` — imports `RetrievedChunk` from `retrieval_types`; necessary consistency for refactor/tests. Keep.
  - `.gitignore` — hygiene only (ignores env variants/caches/build outputs/eval results/OS junk); no runtime dependency, but safe to include to prevent accidental commits. Keep as reviewed hygiene.
  - `.dockerignore` — deployment-relevant build-context hygiene (excludes chroma exp dirs/eval results/env files); reduces Docker/Render build context. Keep.

### Refined exact publish allowlist (no unresolved REVIEW entries)
`.dockerignore`, `.gitignore`, `.env.example`, `Dockerfile`, `render.yaml`, `requirements.weaviate.runtime.txt`, `frontend/src/App.jsx`, `frontend/.env.example`, `src/rag_app/config.py`, `src/rag_app/schemas.py`, `src/rag_app/service.py`, `src/rag_core/__init__.py`, `src/rag_core/retrieval_types.py`, `src/rag_core/weaviate_retriever.py`, `src/rag_core/pipeline.py`, `src/rag_core/index.py`, `src/rag_core/retrieval.py`, `src/rag_core/evaluation.py`, `tests/test_api.py`, `tests/test_citation_metadata.py`, `tests/test_fixed_chunker.py`, `tests/test_rag_evaluation.py`, `tests/test_weaviate_retriever.py`, `tests/fixtures/exp001/chunks.json`, `tests/fixtures/exp001/exp001_ground_truth.json`, `tests/fixtures/exp001/exp001_questions.csv`, `README.md`, `docs/DEPLOYMENT.md`, `docs/RUN_LOCAL.md`, `docs/ENGINEERING_LOG.md`.

### Remaining blockers
- Owner approval for exact file-by-file staging (no `git add .`).
- Actual Vercel origin for Render `RAG_CORS_ORIGINS`.
- Render/Vercel access and dashboard secrets.
- Gemini default-mode confirmation (extractive default; Gemini only on explicit request).
- Keep `RAG_INGEST_TOKEN` unset.
- Render Free 512MB actual fit and Weaviate gRPC reliability unverified until deployment smoke/memory evidence.


## Task 43 — Gemini Default Grounded Answer Generation (2026-11-26, no-stage/no-deploy)

**Status:** Implemented locally; no staging/commit/push/deploy.

### Product behavior (approved)
- Gemini is the default generation mode for normal user requests (`RAG_GENERATION_MODE=gemini`).
- Gemini receives only the user question and selected retrieved evidence; prompt contract remains grounded JSON.
- Citations remain app-resolved against retrieved chunks; no model-invented source metadata trusted.
- Insufficient evidence -> abstention (existing guard/pipeline behavior preserved).
- Gemini disabled/unavailable/error -> safe explicit fallback to grounded extractive (`generation.fallback_to_extractive=true`, explicit status/errors) or abstention where already abstained; never ungrounded/fabricated answer.
- Explicit `mode=extractive` remains supported. `RAG_ALLOW_GEMINI=false` remains supported.

### Files changed
- `src/rag_app/config.py` — default `default_generation_mode` / `RAG_GENERATION_MODE` -> `gemini`.
- `src/rag_app/schemas.py` — `QueryRequest.mode` and `GenerationModel.mode` defaults -> `gemini`.
- `src/rag_app/service.py` — request mode fallback defaults to `gemini`.
- `tests/test_api.py` — added Task 43 regression tests (default Gemini, extractive override, evidence passed, citations valid, abstention, disabled fallback, error no fabrication, API compatibility).
- `.env.example` — added `RAG_GENERATION_MODE=gemini`, `RAG_ALLOW_GEMINI=true`.
- `render.yaml` — `RAG_GENERATION_MODE=gemini`.
- `README.md` — updated default generation description.
- `docs/DEPLOYMENT.md`, `docs/ENGINEERING_LOG.md` — updated.

### Validation
- `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py tests/test_api.py tests/test_citation_metadata.py tests/test_rag_evaluation.py -q`
- **86 passed, 0 failed, 0 skipped, 31 warnings** (28.19s). Warnings: existing PydanticDeprecatedSince211 from `chromadb/types.py`.

### Limitations / blockers
- Real Gemini API key/quota/model availability not exercised in tests (mocked).
- Render/Vercel deployment not performed; Render Free 512MB fit and Weaviate gRPC reliability still unverified.
- Owner must set Render secrets/CORS origin and confirm deploy.


## Task 44 — Final implementation review and publish manifest reconciliation (2026-11-26, no-stage/no-deploy)

**Status:** Review-only reconciliation. No code behavior changes. No staging/commit/push/deploy.

### Task 43 review findings
- Verified defaults consistent: `RAG_GENERATION_MODE=gemini` in config/env template/render, `QueryRequest.mode`/`GenerationModel.mode` default `gemini`.
- Default omitted mode uses Gemini; explicit `mode=extractive` supported. Invalid API modes rejected by schema Literal (422).
- Gemini receives selected grounded statements/citations only; `assemble_generated_answer`/claim parsing validates IDs and resolves citations to supplied retrieved evidence. `citation_validation.app_resolved=true`.
- Insufficient evidence abstains via existing guard/pipeline path. Gemini disabled/unavailable/error returns explicit grounded extractive fallback (`fallback_to_extractive=true`, status/errors) or preserved abstention; no fabricated/ungrounded answer path observed.
- No concrete correctness defect requiring production behavior change identified.

### Manifest reconciliation
- Task 42 reported 31 files but enumerated 30 in DEPLOYMENT allowlist; corrected exact count is **30**.
- `docker-compose.yml` remains excluded (local dev convenience, no concrete Render production dependency).
- All 30 allowlist files verified present in current worktree.

### Validation
- Not re-run (no code changes). Last required command/result (Task 43): `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py tests/test_api.py tests/test_citation_metadata.py tests/test_rag_evaluation.py -q` => **86 passed, 0 failed, 0 skipped, 31 warnings**.

### Blockers unchanged
- Owner approval for exact file-by-file staging; Vercel origin for `RAG_CORS_ORIGINS`; Render/Vercel secrets/access; keep `RAG_INGEST_TOKEN` unset; Render Free 512MB and Weaviate gRPC reliability unverified until deploy smoke/memory evidence.
