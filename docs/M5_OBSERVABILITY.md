# M5 — Application Observability Progress (2026-11-15)

**Status:** lightweight in-process observability implemented and tested. **No mandatory paid dependencies. No retrieval/chunking/EXP-003/threshold/production-behavior changes.** Compatible with genuinely free/local deployment (in-memory, exportable via JSON endpoints).

## Implemented

### Request tracing (`src/rag_app/observability.py`)
- Extended `RequestTrace` with `details`, `add_detail()`, and `total_ms()`.
- Captures per-stage timings/events/errors (`retrieval`, `selection_and_extractive`, `gemini_generation`, etc.) and structured details:
  - **retrieval:** retrieved chunks (rank/chunk_id/section/pages/offsets/distance)
  - **selection:** selected evidence IDs, statement/citation counts, selected lexical support, pipeline status/reason
  - **abstention:** abstained flag and reason (including WP1.1 guard reason when applied)
  - **citation validation:** app-resolved status/invalid source IDs/uncited claims/errors (including Gemini validation path details)
  - **generation (when attempted):** status/model/prompt version/claim and error counts and token metadata when available
- Trace snapshot included in response debug as `debug.trace` for export/inspection (`request_id`, timings, events, errors, details, total_ms).

### Aggregated metrics (`src/rag_app/metrics.py`)
- `MetricsCollector` (thread-safe, in-memory, bounded recent window, default 1000 records) — no external service required.
- Records per request: request_id, timestamp, status, abstained, mode, total/stage timings, retrieved/selected/citation/statement counts, citation validation error count, error/failure flags, generation attempt/status, token usage and estimated cost fields when available.
- Aggregates: totals (requests/failures/abstentions/answered/errors/generation attempts/errors), rates (abstention rate, failure rate, answered citation coverage rate proxy), latency summaries (count/mean/p50/p95) for total/retrieval/selection/generation, recent records export.

### Service/API integration
- `src/rag_app/service.py`: records trace details at retrieval/selection/abstention/citation-validation/generation points; emits metrics for answered/abstained/error responses (metrics recording never breaks request handling).
- `backend/main.py`: added `GET /metrics` endpoint returning aggregated metrics snapshot (JSON, exportable). Existing `/`, `/health`, `/query`, `/ask` preserved.

## Tests (executed)
- `tests/test_api.py`: added `test_metrics_endpoint`; full API suite **8 passed, 4 warnings**.
- Regression: `python -m pytest tests -q -k "not real_index"` **108 passed, 1 deselected, 29 warnings** (existing environment warnings only).

## Captured fields (when available)
- Request IDs, stage timings, retrieval results metadata, selected evidence IDs, citation validation outcomes, abstention reasons, errors, model/prompt/generation status.
- Token usage and estimated cost: supported in schema/collector fields and populated from generation metadata when provided by generator (`prompt_token_count/candidates_token_count/total_token_count` or input/output tokens, `estimated_cost_usd`) — **currently Gemini path may not supply cost; values recorded as null/unknown when unavailable (M1 cost not computed)**.

## Limitations / next steps
- In-memory metrics reset on process restart and are per-instance (acceptable for local/free/single-instance; for multi-instance free deployment, export/scrape `/metrics` externally if needed).
- Latency aggregates can include cold-start/model-load spikes; steady-state and warm-up reporting should be distinguished in future harness runs.
- "Citation coverage" aggregate is a proxy (answered responses with >=1 citation); combine with M4 harness evidence-overlap metrics for quality gates.
- No dashboards/tracing backend added (by design, zero-cost). Future: optional JSONL/export sink, Prometheus-compatible text export, or free local dashboard — only if genuinely free and approved.
- CI integration pending: add reproducible eval command and measured-threshold gates once M4 expanded baselines/thresholds exist.