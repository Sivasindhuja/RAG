# Oracle Cloud Always Free — Deployment Investigation (2026-11-15)

**Purpose:** investigate Oracle Cloud Infrastructure (OCI) Always Free as a potential *genuinely $0* public deployment target for the existing FastAPI + React (static build) + Chroma + sentence-transformers (all-MiniLM-L6-v2) stack, with **minimal changes**.  
**Status:** research/investigation only. **No account created, no resources provisioned, no billing enabled, no deployment performed. No consequential infrastructure changes made.**  
**Constraint:** EVERYTHING MUST REMAIN COMPLETELY FREE (no subscriptions, paid plans, trials/promotional credits, or future payment requirements). Local application remains the validated source of truth.

> **Evidence limitation (recorded):** In this session, direct authoritative page retrieval was blocked/unavailable (`websearch` unavailable; `webfetch` to `docs.oracle.com` transport error and `oracle.com/cloud/free` returned 403). Findings below are therefore **best-effort from widely documented OCI Always Free policies known up to early 2026** and must be **re-verified against current official OCI documentation/console terms at the time of any approved action**. Where current eligibility/quotas/availability cannot be confirmed here, they are marked **UNVERIFIED (verify before action)**.

## 1. Always Free eligibility and typical resource quotas (UNVERIFIED — verify current)
Commonly documented Always Free resources (subject to change by Oracle):
- **Compute (Ampere A1 / ARM):** up to 4 OCPUs and 24 GB memory total for Always Free Ampere A1 instances (flex shapes). Availability is region/tenancy/capacity dependent.
- **Compute (AMD/Intel Micro):** up to 2 VM.Standard.E2.1.Micro instances (1/8 OCPU, 1 GB memory each) in eligible regions.
- **Block storage:** typically 200 GB total block volume storage (including boot volumes) for Always Free; boot volumes consume quota.
- **Object storage:** typically 20 GB (or region/account-specific allowance) — **UNVERIFIED**.
- **Networking:** 1 VCN, public IPs, load balancer allowances vary; **outbound data transfer** has a monthly Always Free allowance (commonly cited ~10 TB/month aggregate in some docs historically) — **UNVERIFIED current limit/terms**.
- **Autonomous Database:** 2 Always Free ADB instances possible — not required for current stack (we use local Chroma persistence).

**Action required before approval:** confirm exact current Always Free shapes, per-resource quotas, and which are marked "Always Free" (not Free Trial credits) in the target home region and tenancy.

## 2. Can a new eligible account actually obtain required VM resources?
- **UNVERIFIED / risk:** Ampere A1 capacity is frequently reported as constrained in popular regions; "Out of host capacity" can prevent instance creation even for eligible accounts. Micro instances may be easier but likely underpowered.
- **Reclamation/idle policies:** OCI has policies to reclaim/restart/stop idle Always Free compute instances based on utilization thresholds over time (historically based on CPU/network/memory metrics over a 7-day window) — **UNVERIFIED current thresholds/policy**. This can interrupt a low-traffic demo.
- **Practical implication:** obtaining and retaining a suitable Always Free VM is **not guaranteed** by eligibility alone; it depends on region capacity and ongoing utilization.

## 3. Fit with existing implementation (FastAPI + React + Chroma + MiniLM)
- **Architecture compatibility:** a single Always Free VM can host backend (uvicorn/FastAPI) and serve frontend static build (nginx/caddy or FastAPI static) with minimal code changes. Chroma persistent dir (`chroma_db_production`, ~3.89 MB) and source PDF are small.
- **Compute/RAM feasibility (estimate, must be measured):**
  - `sentence-transformers/all-MiniLM-L6-v2` + runtime deps (torch/sentence-transformers/chromadb/fastapi) have non-trivial memory footprint; model load causes large cold-start latency (observed ~17s first retrieval in M4 baseline run in this environment). A 1 GB Micro instance is **likely insufficient**; a 2–4 OCPU / 12–24 GB Ampere A1 instance is **plausibly feasible** for low traffic but **must be validated by measurement** (RSS, load time, concurrent requests).
  - Chroma persistent client over local disk is suitable for a single VM.
- **Storage:** boot + OS + Python venv + model cache + index/PDF should fit within typical 200 GB block quota, but model/cache location and image size must be managed.
- **Minimal-change path (if approved and feasible):** package backend with pinned deps, bake/download model cache at build or first start (network egress/time considerations), serve frontend build, run behind reverse proxy with TLS termination strategy compatible with free constraints; no RAG logic changes.

## 4. Signup/payment-method requirements and risk of charges
- **Payment verification:** OCI account creation commonly requires identity/payment-method verification (credit/debit card) even for Always Free — **UNVERIFIED current requirement by country/account type**. Verification can place temporary authorization holds depending on issuer.
- **Charge risk:** "Always Free" resources must remain within Always Free-eligible shapes/quotas/regions and account must avoid upgrading to paid/using non-Free resources. Misconfiguration (wrong shape, extra storage, paid services, exceeding quotas, or leaving Free Trial/PAYG transitions enabled depending on account state) can incur charges. **Cannot be confidently guaranteed $0 without careful tenancy-specific verification and guardrails.**
- **Required safeguards if ever approved:** use only resources explicitly labeled Always Free, set budgets/alerts where available (if free), monitor usage, avoid paid PaaS/services, and confirm account type/upgrade state. Even then, absolute guarantee depends on provider terms/behavior outside project control.

## 5. Networking and operational constraints
- Public endpoint exposure requires security list/NSG rules and a TLS strategy (e.g. reverse proxy + certificate). Let's Encrypt/certbot can be $0 but adds operational steps; exposing plain HTTP is insecure for public demo.
- Single-VM deployment has no managed HA/scaling; uptime depends on instance health, idle reclamation, and host capacity events.

## 6. Recommendation (investigation conclusion)
- **Not approved for deployment at this time (per decisions).** Oracle Always Free is the most plausible *infrastructure-level* $0 VM option for the current stack compared to rejected/unsuitable alternatives, but it is **not confidently guaranteed $0 nor guaranteed obtainable** without tenant-specific verification.
- **Before any provisioning (explicit approval required):** (1) re-verify current official Always Free resources/quotas, idle reclamation, and signup/payment terms; (2) confirm home region capacity for Ampere A1 (or validate Micro infeasibility by measurement); (3) run a local resource measurement (peak RSS, model load time, disk footprint, steady-state latency/concurrency) to choose shape; (4) document a $0-only architecture/guardrail checklist and rollback plan.
- **No RAG behavior changes required** for VM hosting in principle; deployment packaging/ops work only, and only after approval. Local application remains the validated source of truth.