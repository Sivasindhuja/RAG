# M3 — Deployment Feasibility & Architecture Decision Report (Evidence-Based)

**Date:** 2026-11-15  
**Status:** assessment complete for review; **no consequential infrastructure changes implemented**.  
**Scope:** strict zero-cost deployment feasibility for the current SATCOM NGP RAG application (`data/satcom-ngp.pdf`) while preserving validated RAG behavior (fixed-char chunking, corrected production index, EXP-003 default selection, app-validated statement-governing citations, abstention/answerability guard, extractive default; optional Gemini via `GeminiGenerator`, M1 not accepted for quality).

**Strict zero-cost requirement applied:** no paid plans/subscriptions/upgrades/recurring charges, no paid infra even if small, no free trials/promotional/expiring credits, no architecture requiring payment after a limited period, only genuinely $0 services/tiers for intended usage with verified eligibility/quotas/restrictions, no unexpected-charge risk accepted, no paid Gemini billing/model. If unsuitable, explain limitation and propose a different zero-cost architecture without requesting paid approval.

---

## 1. Current implementation baseline (local, verified)

| Item | Evidence (repo, 2026-11-15) | Implication |
|---|---|---|
| Backend | FastAPI app `backend/main.py` + `src/rag_app/*` + `src/rag_core/*` (Python). Endpoints: `/`, `/health`, `/query`, `/ask`. Extractive default, optional Gemini. | Single HTTP API service; can be containerized or adapted to ASGI/serverless entrypoints. |
| Frontend | React/Vite SPA `frontend/` (build output ~296 KB / 301 KB in `frontend/dist`, `npm run build` passes). Calls `${VITE_API_BASE_URL}/query` and `/health`. | Static assets deployable separately; can also be served by backend/container. |
| Data/index | `data/satcom-ngp.pdf` 68 KB; corrected production Chroma index `chroma_db_production/` **~3.89 MB** (49 chunks, `satcom_ngp_production_fixed_char`). | Index is small enough to bundle as read-only artifact in image/repo; must not rely on writable persistent local FS in ephemeral/serverless runtimes for canonical index. |
| Embeddings/query | `sentence-transformers/all-MiniLM-L6-v2` via `DenseIndex`/`SentenceTransformer` (`src/rag_core/index.py`, `src/rag_app/service.py`). | Runtime requires model weights + inference libraries; query-time CPU embedding on each request/cold start unless cached in warm process. |
| Vector store | `chromadb.PersistentClient` (local path). | Works with bundled read-only directory in long-lived/container FS; not suitable for multi-instance mutable serverless storage; serverless FS is ephemeral/read-only patterns vary. |
| Python deps (declared/installed) | `requirements.txt` includes heavy/legacy/eval deps (`sentence-transformers`, `chromadb`, `torch`, `onnxruntime` via chroma, `ragas`, `langchain-*`, `pandas`, `datasets`, etc.). Installed examples: `torch 2.12.0`, `sentence-transformers 3.0.1`, `chromadb 0.5.23`, `onnxruntime 1.24.4`. | **Major deployment risk:** ML/runtime dependency footprint is large (torch wheels/models commonly hundreds of MB to >1 GB uncompressed depending on build); current `requirements.txt` is not a minimal production runtime manifest. Bundle/memory/cold-start feasibility must be measured. |
| Validation | WP2/WP3 local: backend+frontend running, API/CORS flow verified for Q12/Q05/normal/repeat/health/backend-unavailable scenarios; `python -m pytest tests -q` **108 passed**; frontend build **pass**. | Functional baseline to preserve; deployment must not alter RAG behavior. |

**Not measured in this assessment (would require build/test):** exact minimal runtime bundle size, container image size, peak RSS with MiniLM+Chroma loaded, cold-start time on target platforms, and per-request embedding latency on constrained CPU/memory. Claims below distinguish authoritative limits from untested fit.

---

## 2. Evidence sources (authoritative / primary where available)

- **Vercel:** Limits/Functions/pricing docs (web search + `https://vercel.com/docs/limits` fetched; search results include Functions limitations, Hobby allowances, Python 500 MB uncompressed, 300s/2GB Hobby, 4 active CPU-hrs/360 GB-hrs/1M invocations included, no additional usage purchase on Hobby, non-commercial Hobby note on some pages) — **verify current plan terms in dashboard at deploy time**.
- **Hugging Face:** Spaces overview/pricing pages (search results citing official docs), HF pricing page search results, HF Spaces skill/docs excerpts: **“Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts, Team or Enterprise for organizations. Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU.”**; CPU Basic listed $0/hr but gated for Gradio/Docker creation on free accounts; 50 GB non-persistent disk, sleep on free hardware, 2 vCPU/16 GB for CPU Basic.
- **Gemini API:** `https://ai.google.dev/gemini-api/docs/rate-limits` and pricing pages/search results: Free tier exists for eligible models with RPM/TPM/RPD quotas (model/tier/project-specific, viewed in AI Studio), RPD resets midnight Pacific, spend-based limits N/A for Free, free-tier content may be used to improve products, some features unavailable on free tier, no billing required to remain on Free tier (billing required to upgrade tiers); exact active limits vary by project/model/time.
- **Render:** search results citing official free/web-service docs: free web services (512 MB RAM/0.1 CPU free plan), spin down after 15 min idle (~1 min spin-up), 750 free instance hours/workspace/month, ephemeral FS (no persistent disk on free), outbound bandwidth/build-minute limits with suspension risk if exceeded without payment method, possible suspension for high outbound traffic, SMTP ports blocked, not for production per docs. (Direct `render.com/docs/*` fetch failed in this environment; evidence based on consistent search-result excerpts of official docs — **reconfirm on Render docs/dashboard before use**.)

---

## 3. Target-by-target assessment

### 3.1 Vercel Hobby (target) — **not verifiably compatible for current backend as-is; conditional/untested at best, likely infeasible**

| Criterion | Evidence | Assessment |
|---|---|---|
| Python serverless runtime | Vercel supports Python functions (Fluid compute). Standard uncompressed bundle limit **500 MB** for Python; large functions beta up to 5 GB (requires Fluid/Active CPU, excludes some features; availability/eligibility must be verified). | Backend is Python/FastAPI-style; can be adapted to Vercel Python function/ASGI patterns, but packaging is non-trivial. |
| Execution/memory | Hobby: **300s max duration**, **2 GB / 1 vCPU** default and max (memory not configurable on Hobby per Vercel plugin docs). | 300s is sufficient for request/response; 2 GB may be tight for `torch + sentence-transformers + chromadb + onnxruntime` loaded concurrently — **untested**. |
| Cold starts | Serverless cold starts include runtime + dependency/model initialization; in-memory caches not guaranteed across invocations. | `SentenceTransformer`/model load on cold path can be slow and memory-heavy; **likely poor UX/reliability without measurement**. |
| Bundle size | 500 MB uncompressed Python limit (standard); build/static limits apply separately. Current deps include `torch`, `sentence-transformers`, `chromadb`, `onnxruntime`, `transformers/tokenizers` transitively, plus unused heavy libs in `requirements.txt`. | **High risk of exceeding 500 MB** or approaching limits even with pruning; MiniLM weights + torch CPU runtime alone are substantial. **Cannot claim fit without a measured minimal bundle.** |
| Chroma persistent index | Serverless FS ephemeral; bundled files in deployment are read-only/static artifact. Prod index ~3.9 MB. | **Feasible in principle** if bundled read-only and opened appropriately, but depends on Chroma/SQLite behavior on read-only bundled paths and cold initialization — **untested**. |
| MiniLM query embedding | CPU inference via sentence-transformers/torch (or alternative backend). | Technically possible in CPU function if memory/bundle fit; cold-start/model-load cost likely significant. No rewrite requested now, but feasibility is constrained. |
| Free-tier compatibility/cost | Hobby $0 with included allotments (e.g. 4 active CPU-hrs, 360 GB-hrs, 1M invocations, bandwidth limits per docs); cannot buy extra usage on Hobby; overages may pause features/require waiting period. Some Vercel materials note Hobby for personal/non-commercial use — **verify current terms/eligibility**. | **Monetarily $0 if within limits and terms**, but operationally constrained; risk of suspension/pause on overuse (not monetary charge). |
| Frontend+backend | Vercel can host static frontend and serverless functions in one project. | Architecturally convenient **if backend fits**. |

**Verdict (strict):** **Reject as “verified zero-cost compatible” for the current backend.** Vercel Hobby may be monetarily free within allotments, but current ML stack compatibility (bundle/memory/cold start/model+Chroma loading) is **unproven and likely problematic**. Do not claim Vercel compatibility without building/measuring a minimal deployment artifact and runtime behavior.

### 3.2 Hugging Face Docker Spaces (evaluated) — **ineligible under strict zero-cost for intended FastAPI Docker deployment**

| Required check | Evidence | Result |
|---|---|---|
| Free-account / PRO requirement for Docker Spaces | HF Spaces overview/pricing/skills: “Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts, Team or Enterprise for organizations.” Free personal accounts in good standing may host up to **2 ZeroGPU Gradio Spaces** only (exception). CPU Basic is $0/hr **but compute Spaces are gated**; skills note `cpu-basic` “needs a paid plan” for creation on free accounts. | **Not genuinely free.** Docker Space on CPU Basic requires **PRO ($9/month)** for personal accounts (or paid org plan). **Reject** under strict zero-cost. |
| Run FastAPI and serve built React in one Docker container | Docker SDK allows arbitrary container; FastAPI can serve API and static frontend files. | **Technically yes** (if paid/eligible compute were allowed), but irrelevant given cost gate. |
| CPU Basic availability/eligibility | CPU Basic 2 vCPU/16 GB/50 GB non-persistent, $0/hr listed; creation/hosting of Docker/Gradio compute Spaces gated by paid plan (except ZeroGPU Gradio exception). | **$0 hourly rate does not imply free eligibility** for Docker Spaces. |
| RAM/startup for Chroma + MiniLM | 16 GB CPU Basic would likely be sufficient in memory *if* runnable, but startup includes image boot + dependency/model load; 50 GB non-persistent disk can hold image/layers but resets on restart/rebuild contexts. | Potentially feasible technically on CPU Basic hardware class; **blocked by plan requirement and zero-cost rule**. |
| Prebuilt read-only index packaging/download | ~3.9 MB index can be committed/baked into Docker image or downloaded at startup from public artifact/repo; disk non-persistent across restarts but image contents available. | Feasible technically. |
| Non-persistent disk/storage alternatives | 50 GB ephemeral; persistent storage via buckets is paid (`canPay`/user confirmation per HF skills). | No zero-cost persistent writable storage; read-only baked assets fine. |
| Secrets/public API security | Space secrets supported for env vars (HF); public Space exposes app endpoint — must protect admin/debug surfaces, CORS, rate limiting, and not leak keys/debug info. | Manageable with discipline, but public exposure increases abuse/cost-risk considerations for Gemini quotas. |
| Actual deployment/operating cost | PRO required for Docker Space on CPU Basic → **$9/month** (personal) plus any paid hardware/storage if used. | **Violates strict zero-cost. Reject.** |

**ZeroGPU note:** ZeroGPU is Gradio-only, PyTorch-first, quota-limited, and unsuitable for hosting our existing FastAPI + Chroma + sentence-transformers CPU request path as-is; it does **not** satisfy “preserve existing RAG implementation with least unnecessary change”.

### 3.3 Other genuinely free alternatives (assessed)

| Option | Genuinely $0? | Compatibility with current implementation | Key limitations/risks | Verdict |
|---|---|---|---|---|
| **Render free web services** | Service type offered free (750 free instance hrs/workspace/month, no stated subscription fee); over-bandwidth/build overage can be billed if payment method on file else suspended; possible suspension for high outbound traffic. **Must confirm no payment method required and current terms in dashboard/docs.** | Docker/native Python supported; but **512 MB RAM / 0.1 CPU** on free plan, spin down after 15 min, ~1 min cold start, ephemeral FS, no persistent disk, no shell, outbound restrictions. | **Very likely insufficient RAM/CPU** for torch+sentence-transformers+Chroma; poor latency due to spin-down/cold starts; index/model would need to be in image and loaded in tiny memory — **unproven, likely infeasible**. | **Not recommended / likely incompatible** for current ML backend; only conditionally “$0” if it runs within limits and no paid overage path triggered. |
| **Cloudflare Workers/Pages** | Pages/Workers have free tiers, but Python ML stack (torch/sentence-transformers/Chroma) not supported in current form. | Requires major rewrite (e.g. external inference/vector services) — conflicts with “preserve existing implementation”, and external services may not be zero-cost. | Architectural mismatch. | **Reject for current implementation.** |
| **PythonAnywhere (free)** | Free tier exists but outbound internet restricted to allowlist. | Gemini API calls likely blocked; dependency/install constraints. | Cannot reliably call Gemini; limited compute. | **Reject.** |
| **Google Cloud Run / AWS / Azure always-free tiers** | Often require billing account enablement and have complex quota/egress/model risks; easy to incur charges if misconfigured. | Could run containers in principle. | **Unexpected-charge risk / billing-account requirement** conflicts with strict zero-cost guarantee without careful verified controls. | **Reject under strict zero-cost confidence requirement** (unless proven otherwise with hard billing safeguards, not established here). |
| **Oracle Cloud Always Free (compute VM)** | Advertised as “Always Free” (not expiring trial credits) in eligible regions/tenancies; typically requires credit card for signup verification; capacity/reclamation policies apply. | Can host current FastAPI+frontend+Chroma+MiniLM in one VM/container with minimal code change (closest to local). | **Eligibility/availability/payment-method-on-file caveats, account enforcement/reclamation risk, ops burden/security patching**; “no unexpected charges” depends on staying strictly within Always Free resources and Oracle policy — **high verification burden**. | **Possible technically closest zero-cost-capable path** *only if* tenancy/resources are confirmed Always Free with no billing/upgrade, but **not confidently “guaranteed zero-cost”** without tenant-specific verification. Present as constrained alternative, not approved. |
| **Static frontend-only hosting (GitHub Pages/Cloudflare Pages/Vercel static) + no public backend** | Static hosting can be $0. | Backend/API would be unavailable publicly. | Fails “publicly accessible, functional RAG application”. | **Reject** (unless paired with a compliant backend, none identified). |

**Conclusion:** Under strict zero-cost rules and current implementation constraints, **no assessed managed platform provides a clearly eligible, reliable, zero-cost deployment for the current Python ML backend**. Vercel Hobby is monetarily plausible but **technically unverified/likely incompatible** for the heavy embedding stack; Hugging Face Docker Spaces are **plan-ineligible ($9/mo PRO)**; Render free is **resource-constrained**.

---

## 4. Component-specific feasibility

### 4.1 FastAPI backend on Vercel Hobby
- **Possible in form:** Python/ASGI serverless function with 300s/2 GB (Hobby).  
- **Main blockers/unknowns:** dependency bundle size (500 MB std), cold start/model load, memory headroom, adapting uvicorn/FastAPI app to Vercel entrypoint, read-only access pattern for bundled Chroma SQLite files.  
- **Evidence status:** limits known; **end-to-end compatibility not tested**.

### 4.2 Chroma persistent index (~3.9 MB) access
- **Preservation strategy (no rewrite):** keep prebuilt `chroma_db_production/` as **read-only deployment artifact** baked into container/image or serverless bundle; configure `RAG_PERSIST_DIR`/collection to point to it; ensure no runtime writes required (current query path reads).  
- **Risk:** serverless ephemeral FS means any writes lost (acceptable if read-only); Chroma/SQLite on read-only media may need flags/config validation in target runtime.

### 4.3 SentenceTransformer/MiniLM query embedding
- **Current:** in-process CPU embedding via `sentence-transformers` (`all-MiniLM-L6-v2`) on each query; model loaded lazily/cached per process.  
- **Serverless impact:** cold processes reload model; memory and latency spikes.  
- **Zero-cost-preserving alternatives (would be architectural changes, not approved now):** precompute query embeddings client-side impossible for arbitrary queries; use a genuinely free embedding API (must be verified $0, no billing, quotas, CORS/secrets) — **not identified/validated here**. Preserving current layer means hosting compute capable of running it.

### 4.4 Gemini API (zero-cost usage)
- **Free tier:** available for eligible models/projects with per-model/project RPM/TPM/RPD quotas (view in AI Studio), RPD resets midnight Pacific; no spend-based limit for Free tier. Must **not** enable paid billing, not use paid-only features, and stay within quotas/rate limits. Free-tier content may be used to improve products (data-handling consideration).  
- **Operational controls:** keep extractive default, make Gemini optional and explicitly rate-limited/disabled on quota exhaustion, handle `429 RESOURCE_EXHAUSTED`, avoid grounding/paid features, monitor usage in AI Studio, never store/expose keys in frontend.  
- **Current app:** M1 not accepted for quality; extractive mode is grounded baseline and can operate without Gemini (zero external API cost) if desired.

### 4.5 Secrets, CORS, rate limiting, public security
- Store `GOOGLE_API_KEY`/`GEMINI_API_KEY` as platform secrets/env vars only; never ship to frontend.  
- CORS: restrict origins to deployed frontend domain(s) in production (`RAG_CORS_ORIGINS`).  
- Add/rationalize rate limiting and abuse protection at edge/app level where feasible (platform-dependent); public endpoint can be abused causing quota exhaustion/availability issues.  
- Do not expose internal debug/config; keep `/health` minimal in public deployment.

---

## 5. Deployment architecture options (comparison)

| Option | Monetary cost (strict) | Complexity | Reliability/cold start | Compatibility (current code) | Migration effort |
|---|---:|---:|---|---|---|
| **A. Vercel Hobby: static frontend + Python serverless backend (bundled read-only index/model)** | $0 if within allotments/terms (verify) | Medium–High (packaging, entrypoint, size tuning) | Poor–Unknown (cold starts, 2 GB/1 vCPU, model load) | **Unverified/likely low** due to deps/model | Medium (build config, minimal runtime manifest, measure) |
| **B. Hugging Face Docker Space (FastAPI + static frontend, CPU Basic)** | **Not $0** for personal Docker Space (PRO $9/mo required) | Low–Medium | Moderate (sleep on idle; 16 GB/2 vCPU) | **High** technically | Low if allowed — **rejected by cost rule** |
| **C. Hugging Face ZeroGPU Gradio Space** | $0 for eligible free accounts (2 spaces, quotas) but Gradio-only/GPU quota semantics | High (rewrite to Gradio + GPU-decorated flow) | Quota/queue/cold behavior varies | **Low** (major app/runtime change) | High — **rejected (preserve implementation)** |
| **D. Render free web service (Docker/Python)** | Potentially $0 within limits, but suspension/operational risks; must avoid overage billing | Medium | Poor (512 MB/0.1 CPU, 15 min spin-down) | **Low–Unknown** (RAM/CPU) | Medium — **likely infeasible** |
| **E. Oracle Always Free VM (single VM, Docker/FastAPI+frontend)** | Advertised Always Free (if eligible resources/tenancy compliant); signup/payment-method/ops caveats | Medium (VM ops, TLS, domain, hardening) | Good once running (no forced spin-down; user-managed) | **Highest** (closest to local) | Low–Medium — **zero-cost not confidently guaranteed without tenant verification** |
| **F. Local/self-hosted demo only (no public managed hosting)** | $0 | Low | N/A (not public cloud) | **Full** | None — but fails “publicly accessible” goal |

---

## 6. Recommended approach (evidence-based, strict zero-cost)

### 6.1 Recommendation
**Do not proceed with Vercel Hobby or Hugging Face Docker Spaces deployment at this time under the strict zero-cost requirement.**
- **Vercel Hobby:** cannot be claimed compatible without measured bundle/memory/cold-start validation for the current ML stack; likely infeasible as-is.
- **Hugging Face Docker Spaces:** **ineligible** for zero-cost personal deployment (requires PRO plan despite $0/hr CPU Basic listing).

**Preferred zero-cost-preserving path (to preserve existing RAG implementation and correctness):**
1. **Keep the current locally working application as the source of truth** (already validated). Provide a **reproducible public demo via local/self-hosted execution** (runbook + recorded demo/video/screenshots or evaluator-run instructions) as an immediately achievable **$0** outcome.
2. **If a publicly hosted URL is mandatory:** perform **tenant-specific verification** of an **Oracle Cloud Always Free** VM (confirm Always Free-eligible shapes, no billing, no trial credits, no upgrade triggers) and, only if verified $0, deploy the existing stack with minimal changes (Dockerized FastAPI serving API + built frontend, bundled read-only index/model cache strategy). This is the **least architectural change** option but carries eligibility/ops caveats outside this repo’s control.
3. **In parallel (design only, no rewrite now):** specify a “minimal runtime” dependency split (remove eval/langchain/ragas/cohere/langfuse from production image) and a **measurement plan** (image/bundle size, RSS, cold/warm latency) so Vercel/Render feasibility can be empirically decided later **without guessing**.

### 6.2 Why this best satisfies priorities
- Prioritizes **zero monetary cost** and avoids false compatibility claims.
- **Preserves** retrieval/chunking/EXP-003/citations/abstention and validated code.
- Avoids insecure/correctness-compromising shortcuts.

---

## 7. Risks and limitations

| Risk | Impact | Mitigation |
|---|---|---|
| False “free” assumption (HF Docker $0/hr, platform overages) | Policy violation/unexpected cost or blocked deploy | Require plan/eligibility verification in-account; reject gated/paid options. |
| ML dependency size/memory on serverless/free tiers | Deploy failure, OOM, slow/timeout | Minimal runtime deps, measure artifacts, keep index read-only/bundled, avoid runtime model downloads where possible. |
| Gemini free-tier quotas/RPD reset/429/data-use terms | Generation unavailable/throttled; privacy trade-off on free tier | Keep extractive default, graceful fallback/abstention, rate limiting, user notice, no billing enablement. |
| Public endpoint abuse | Quota exhaustion, availability issues | CORS restriction, rate limiting, minimal health responses, no debug leakage. |
| Chroma/SQLite on read-only FS | Runtime errors in bundled deployment | Test read-only open mode in target environment before committing. |
| Oracle Always Free eligibility/reclamation (if chosen) | Service interruption or policy constraints | Tenant verification, backups/config-as-code, accept ops risk only with confirmed $0 terms. |

---

## 8. Implementation steps (post-review, conditional)

### 8.1 No-regret steps (safe to do after review, preserve behavior)
1. **Create minimal production runtime manifest** (separate from dev/eval): only FastAPI/uvicorn/pydantic, `sentence-transformers`+CPU torch build strategy, `chromadb`, `numpy`, `google-generativeai`, `python-dotenv` as needed; exclude ragas/langchain/langfuse/cohere/pandas/datasets unless required at runtime.
2. **Add deployment packaging assets (design/build-ready, not deployed):** `Dockerfile` (multi-stage, non-root, bakes `frontend/dist` + `chroma_db_production` + app code; configurable model cache), `.dockerignore`, optional `docker-compose.yml` for local parity.
3. **Measurement harness:** script/notes to record image size, container RSS, cold/warm `/health` and `/query` latency, model load time.
4. **Security hardening for public exposure:** tighten CORS allowlist via env, disable verbose debug in production responses/config, basic rate limiting middleware (if dependency-free/simple) — **verify no paid service introduced**.
5. **Reproducible deployment runbook draft** for selected verified-$0 target (or local public-demo procedure).

### 8.2 Conditional deployment steps (only if a target is verified $0)
- **If Oracle Always Free verified:** provision VM, install Docker/runtime, deploy container, configure reverse proxy/TLS (e.g. Caddy) and firewall, set secrets, smoke-test Q05/Q12/normal/health, monitor quotas/resources.
- **If empirical Vercel/Render measurement proves fit within free limits:** implement platform config and deploy; otherwise do not proceed.

### 8.3 Explicitly deferred (per instructions)
- No DB migration, no embedding-layer rewrite, no RAG behavior/threshold/experiment changes, no paid upgrades/services/billing.

---

## 9. Work that can proceed alongside deployment planning (no infra migration)

| Stream | Can proceed now? | Rationale |
|---|---|---|
| **M4 evaluation planning** | **Yes** | Expand 15q set toward 50–200 manually verified Q/A, define labeling schema (evidence/citations/abstention/correctness/faithfulness/completeness), design offline harness and baseline-derived thresholds — independent of hosting. |
| **M5 observability design/implementation (app-level)** | **Yes** | Extend structured traces/metrics (request IDs already present), aggregate latency/token/citation/abstention/error signals, export/log formats, investigation workflow; choose storage/export compatible with eventual free target later. |
| **CI quality gates** | **Yes** | Add CI for `pytest`, frontend build/lint, and (later) evaluation subsets/gates — runs in existing CI environments; keep costs $0. |
| **Minimal runtime dependency split & packaging tests** | **Yes** | Improves portability and measurement for any target; preserves behavior. |
| **Security hardening (CORS/config/rate-limit design)** | **Yes** | Required before public exposure. |
| **Actual public deployment** | **Blocked pending verified $0 target** | No eligible/verified platform confirmed for current backend. |

---

## 10. Decision required (review)

- **Accept assessment:** Vercel Hobby **not claimed compatible** without measurement; **HF Docker Spaces rejected** (paid plan required); no managed zero-cost platform confirmed for current ML backend.
- **Choose direction:** (a) $0 public-demo via local/reproducible run only, (b) tenant-verified Oracle Always Free VM path, or (c) authorize measurement/build spike for minimal Vercel/Render artifact (still no deployment claim without evidence).
- **No consequential infrastructure changes will be made without your review/approval.**
