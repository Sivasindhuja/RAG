# Deployment

## Target and constraints (Vercel Hobby)
- **Target:** publicly accessible application compatible with Vercel Hobby plan (per project constraints).
- **Observed constraints:** serverless execution limits, cold starts, function bundle size limits, no persistent local filesystem across invocations, request/response and timeout limits, Python runtime constraints, and secret management requirements.
- **Repository reality:** backend depends on `sentence-transformers`, `torch` (transitively), `chromadb`, `pypdf`, `rank_bm25`, etc., and loads a local embedding model (`all-MiniLM-L6-v2`) plus a persistent Chroma index. This is a poor fit for Vercel Hobby serverless functions (large dependencies/model artifacts, cold-start model load, non-persistent local storage).

## Implemented deployment artifacts (best-effort Vercel compatibility)
- `vercel.json`: routes `/api/*` to Python serverless entrypoint and frontend static build.
- `api/index.py`: Vercel Python entrypoint exposing FastAPI app from `backend.main`.
- Frontend: Vite app (`frontend/`) configurable via `VITE_API_BASE_URL`.
- Backend CORS configurable via `RAG_CORS_ORIGINS` (comma-separated explicit origins; local-dev defaults when unset; explicit `*` only when intentionally set; production must set actual Vercel origin).

These artifacts enable deployment attempts/local Vercel-compatible structure but **do not by themselves guarantee a working Hobby deployment** given dependency/model/storage constraints.

## Persistence strategy (required for serverless)
Local Chroma persistence (`chroma_db_production`) and local model loading cannot be assumed in serverless. Options and trade-offs:
1. **Bundled read-only index artifact + model download/cache:** include prebuilt Chroma SQLite/index files in deployment artifact and allow model download on cold start. Trade-offs: large bundle/cold start, ephemeral cache, possible size limit violations on Hobby. Cost: free-tier attempt possible but likely infeasible.
2. **External persistent vector store / hosted retrieval service (free tier where possible):** e.g. hosted Chroma-compatible or alternative free-tier vector DB/API; keep embeddings/index there. Trade-offs: network latency, auth/secrets, service limits, operational complexity. No paid infra added without approval.
3. **Separate backend host for RAG API (free tier where possible) + Vercel frontend:** deploy FastAPI backend on a free-tier-compatible container/VM host with persistent disk for index/model cache; deploy frontend on Vercel pointing to backend. Trade-offs: cold starts/sleep on some free tiers, resource limits, availability not guaranteed. Prior assessment noted Render free likely incompatible and HF Docker Spaces rejected under strict zero-cost personal Docker compute; Oracle Always Free VM is plausible but not confidently guaranteed/obtainable without tenant-specific verification (see `docs/M3_DEPLOYMENT_FEASIBILITY.md`, `docs/ORACLE_ALWAYS_FREE_INVESTIGATION.md`).

**Selected documented strategy:** for Hobby-compatible public deployment, prefer **Vercel frontend + separate free-tier backend host** if a verified free-tier host is available; otherwise document blocker and keep local/public-source deployment reproducible. No paid infrastructure incurred.

## Configuration and secrets
- Backend env: `RAG_PERSIST_DIR`, `RAG_COLLECTION_NAME`, `RAG_TOP_K`, `RAG_RETRIEVAL_MODE`, `RAG_GENERATION_MODE`, `RAG_GEMINI_MODEL`, `RAG_ALLOW_GEMINI`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_CORS_ORIGINS`, `GOOGLE_API_KEY`/`GEMINI_API_KEY` (optional), `COHERE_API_KEY`/`CO_API_KEY` (optional), `LANGFUSE_*` (optional), `RAG_INGEST_TOKEN` (required to enable `POST /ingest`; endpoint fails closed when unset), `RAG_DEFINITION_INTENT_BOOST` (optional, default `false`; opt-in definition-intent/heading-alignment selection boost — do not treat as default-promoted baseline without measured comparison).
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
- `GET /health` config/index/gemini/langfuse status
- `GET /metrics` live operational metrics
- `GET /dashboard` project dashboard (live + offline artifacts)
- `POST /query` extractive grounded answer with citations/abstention
- `POST /ingest` protected index rebuild (controlled environment; requires `RAG_INGEST_TOKEN` + `X-Ingest-Token`; expect 403 when disabled/unauthorized)

---

## Render Free + Vercel Hobby release attempt (2026-11-19, prepared; not publicly deployed)

### Render Free constraints (verified from official docs)
- **Source:** https://docs.render.com/free (fetched in-session)
- Free web services: **750 free instance hours/month**; service **spins down after 15 minutes idle** and spins up on next request (~1 minute).
- **Ephemeral filesystem:** local file changes lost on redeploy/restart/spin-down. **Free web services cannot attach persistent disks.** (Persistent disks require paid services.)
- Free Render Postgres expires 30 days (not used by this app). Bandwidth/pipeline-minute limits apply.
- Practical implication for this stack: no durable on-disk Chroma persistence across restarts/redeploys; cold starts include dependency/model load and possible index rebuild. Memory/CPU on free instances may be insufficient for `sentence-transformers` + PyTorch + ChromaDB (risk of OOM/slow startup) — must be validated by actual deployment.

### Reliability adaptations implemented (no RAG architecture/provider/model changes)
- **Automatic index recovery/re-ingestion:** `src/rag_app/service.py` rebuilds the configured Chroma index from `data/satcom-ngp.pdf` using configured chunking when the index is missing/empty (validated locally: rebuild produced 61 chunks and answered query with citations). This enables recovery on Render Free ephemeral filesystem when prebuilt index is not packaged.
- **Lightweight liveness endpoint:** `GET /healthz` added for platform health checks/cold starts; `GET /health` remains detailed (may trigger initialization/model load/index recovery).
- **Deployment artifacts:** `requirements.runtime.txt`, `render.yaml` (blueprint, `healthCheckPath=/healthz`, `sync:false` secrets), `Dockerfile` (CPU torch, runtime deps, `${PORT:-8000}`), `.dockerignore`, `docker-compose.yml`, `frontend/vercel.json` (canonical frontend-only Vercel config when Vercel project root is `frontend/`). Root `vercel.json` (untracked) is a legacy combined Vercel API+frontend config and is **not** the canonical choice for the Render-backend + Vercel-frontend split.

### Prebuilt index packaging status
- `chroma_db_production/` exists locally but is currently **gitignored** in `.gitignore`; `data/satcom-ngp.pdf` is tracked. For git-based Render deploys, prebuilt index may not be present in source unless explicitly included; automatic recovery from PDF is the fallback. If packaging the index in deployment artifacts is required, ensure deployment source/image includes `chroma_db_production` contents.

### Required Render environment variables (set in Render dashboard, do not commit)
- **App/config:** `RAG_PERSIST_DIR`, `RAG_COLLECTION_NAME=satcom_ngp_production_fixed_char_800_160`, `RAG_RETRIEVAL_MODE=hybrid`, `RAG_GENERATION_MODE=extractive`, `RAG_ALLOW_GEMINI=true`, `RAG_GEMINI_MODEL=gemini-flash-lite-latest`, `RAG_DEFINITION_INTENT_BOOST=false` (default; opt-in only), `RAG_CORS_ORIGINS` (**must be set to the actual Vercel frontend origin before public frontend integration**; placeholder/wildcard is not a production configuration).
- **Secrets:** `GEMINI_API_KEY` and/or `GOOGLE_API_KEY` (Gemini), `CO_API_KEY`/`COHERE_API_KEY` (rerank, optional), `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `LANGFUSE_ENABLED` (optional), `RAG_INGEST_TOKEN` (optional but recommended; required to enable protected `POST /ingest`, which fails closed when unset).

### Actual deployment status
- **Backend (Render):** not deployed in this environment (no Render account/API/CLI credentials or GitHub push/deploy access available).
- **Frontend (Vercel Hobby):** not deployed in this environment (no Vercel account/CLI/token or GitHub integration access available).
- **Public URLs:** none.
- **Public endpoint/frontend/cold-start/restart smoke tests:** not executed.

### User actions/credentials required to complete
1. Push deployment-ready code (including runtime/deploy files and, if desired, packaged index) to a GitHub repo/branch accessible by Render/Vercel (preserving unrelated pre-existing deletions as instructed).
2. Create/connect Render service from repo/blueprint (`render.yaml`), set secret env vars in Render dashboard, deploy, and capture backend URL.
3. Deploy frontend on Vercel Hobby with `VITE_API_BASE_URL=<render-backend-url>`, set any required env vars, and capture frontend URL.
4. Run public smoke tests: `/`, `/healthz`, `/health`, `/query`, `/metrics`, `/dashboard`, frontend query/citation/abstention/error flows, cold-start behavior, index availability/recovery, and post-restart/redeploy behavior; record evidence and any Render resource failures (OOM, timeout, build failure).
