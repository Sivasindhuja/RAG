"""M1 — Validated Gemini answer path evaluation.

Uses the frozen 15-question dataset and accepted EXP-003 extractive selection as
the evidence-selection baseline. Retrieval, chunks, embeddings and ground truth
are unchanged.

Gemini generation (when available) runs over selected verbatim evidence with a
versioned prompt and strict JSON claim contract. Citation IDs are validated and
resolved by the application against actual selected-evidence metadata; the model
must not be trusted for page/section/source references.

If no Gemini API key is available, the script records a blocked generation run
but still produces extractive baseline artifacts and sufficiency/abstention
analysis.

Run:
    python -m experiments.m1_gemini_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentence_transformers import SentenceTransformer

from experiments.production_slice import (
    CHUNK_SIZE,
    COLLECTION_NAME as PRODUCTION_COLLECTION,
    EMBEDDING_MODEL,
    GROUND_TRUTH_PATH,
    OVERLAP,
    PDF_PATH,
    PERSIST_DIR as PRODUCTION_PERSIST_DIR,
    QUESTIONS_CSV,
    TOP_K,
    UNANSWERABLE_CSV,
    _load_unanswerable,
    _overlapping_evidence,
    _write_json,
)
from src.rag_core import (
    ABSTAINED,
    ANSWERED,
    AbstentionPolicy,
    DenseIndex,
    DenseIndexConfig,
    FixedCharacterChunker,
    GEMINI_PROMPT_VERSION,
    GeminiGenerator,
    GroundedPipeline,
    audit_source_coverage,
    compute_retrieval_metrics,
    content_tokens,
    evaluate_answer_support,
    evidence_for_question,
    extract_source_document,
    load_exp001_evidence,
    load_questions,
    token_support,
)
from src.rag_core.generation import (
    DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_TEMPERATURE,
    EVIDENCE_INSUFFICIENT,
    GENERATION_ERROR,
    INSUFFICIENT_EVIDENCE,
    MODEL_ABSTAINED,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ID = "M1-gemini"
RESULTS_DIR = ROOT / "experiments" / "results" / "m1_gemini"

# M1 trustworthiness gate (DR-014): abstain before generation when selected
# evidence is lexically insufficient for the question. This catches in-corpus
# retrieval-failure cases like Q05 where pipeline distance/support heuristics
# allow an answer but selected statements do not address the question.
DEFAULT_MIN_SELECTED_EVIDENCE_SUPPORT = 0.5

MODEL_PREFERENCE = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
]


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", (text or "").lower())


def _selected_evidence_support(question: str, statements: Sequence[Any]) -> float:
    class _Doc:
        def __init__(self, text: str):
            self.text = text

    text = " ".join(s.text for s in statements)
    return token_support(question, [_Doc(text)]) if text.strip() else 0.0


def _evidence_sufficient(question: str, extractive_answer: Any, min_support: float) -> Tuple[bool, Optional[str]]:
    if extractive_answer.status == ABSTAINED:
        return False, extractive_answer.reason or "pipeline_abstained"
    if not extractive_answer.statements:
        return False, "no_selected_evidence"
    support = _selected_evidence_support(question, extractive_answer.statements)
    if support < min_support:
        return False, f"{EVIDENCE_INSUFFICIENT}:selected_support={support:.4f}<{min_support}"
    return True, None


def _claim_grounded(claim: Dict[str, Any], statements_by_id: Dict[int, Any]) -> Tuple[bool, float]:
    claim_tokens = set(_tokenize(claim.get("text", "")))
    cited = claim.get("source_ids", [])
    evidence_text = " ".join(statements_by_id[sid].text for sid in cited if sid in statements_by_id)
    evidence_tokens = set(_tokenize(evidence_text))
    if not claim_tokens:
        return False, 0.0
    coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    return coverage >= 0.80, round(coverage, 4)


def _evaluate_generated(row: Dict[str, Any]) -> Dict[str, Any]:
    statements_by_id = {i + 1: s for i, s in enumerate(row.get("evidence_statements", []))}
    claims = row.get("claims", [])
    claim_eval = []
    for claim in claims:
        grounded, coverage = _claim_grounded(claim, statements_by_id)
        claim_eval.append(
            {
                "text": claim.get("text"),
                "source_ids": claim.get("source_ids", []),
                "markers": claim.get("markers", []),
                "invalid_source_ids": claim.get("invalid_source_ids", []),
                "cited_token_coverage": coverage,
                "grounded_by_token_overlap": grounded,
                "uncited": not claim.get("markers"),
            }
        )
    grounded_count = sum(1 for c in claim_eval if c["grounded_by_token_overlap"])
    uncited_count = sum(1 for c in claim_eval if c["uncited"])
    invalid_count = sum(len(c.get("invalid_source_ids", [])) for c in claims)
    return {
        "claim_count": len(claims),
        "grounded_claim_count": grounded_count,
        "grounded_claim_rate": round(grounded_count / len(claims), 4) if claims else 0.0,
        "uncited_claim_count": uncited_count,
        "invalid_source_id_count": invalid_count,
        "claim_evaluations": claim_eval,
    }


def _has_gemini_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _list_gemini_models() -> List[str]:
    import google.generativeai as genai

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        genai.configure(api_key=key)
    models = []
    for m in genai.list_models():
        if "generateContent" in getattr(m, "supported_generation_methods", []):
            models.append(m.name.replace("models/", ""))
    return sorted(models)


def _select_gemini_model(available: Sequence[str], preferred: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    if preferred:
        if preferred in available:
            return preferred, {"method": "explicit", "selected": preferred, "available_count": len(available)}
        return None, {"method": "explicit", "selected": None, "requested": preferred, "available_count": len(available)}
    for name in MODEL_PREFERENCE:
        if name in available:
            return name, {"method": "preference_list", "selected": name, "preference": MODEL_PREFERENCE}
    # fallback: first flash-like, else first available
    for name in available:
        if "flash" in name:
            return name, {"method": "fallback_flash", "selected": name}
    return (available[0] if available else None), {"method": "fallback_first", "selected": available[0] if available else None}


def _probe_model(model_name: str) -> Dict[str, Any]:
    import google.generativeai as genai

    started = time.perf_counter()
    model = genai.GenerativeModel(
        f"models/{model_name}" if not model_name.startswith("models/") else model_name,
        generation_config={"temperature": 0.0, "max_output_tokens": 64, "response_mime_type": "application/json"},
    )
    prompt = 'Return valid JSON only: {"sentences":[]}'
    try:
        resp = model.generate_content(prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(resp, "usage_metadata", None)
        return {
            "ok": True,
            "latency_ms": round(latency_ms, 2),
            "text_preview": (getattr(resp, "text", "") or "")[:120],
            "prompt_token_count": getattr(usage, "prompt_token_count", None) if usage else None,
            "candidates_token_count": getattr(usage, "candidates_token_count", None) if usage else None,
            "total_token_count": getattr(usage, "total_token_count", None) if usage else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2), "error": f"{type(exc).__name__}: {exc}"}


def _summary_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# M1 — Validated Gemini Answer Path",
        "",
        f"- Experiment: `{payload.get('experiment_id')}`",
        f"- Gemini available: `{payload['gemini'].get('available')}`",
        f"- Selected model: `{payload['gemini'].get('selected_model')}`",
        f"- Prompt version: `{payload['config']['generation']['prompt_version']}`",
        f"- Evidence sufficiency gate: `selected_support >= {payload['config']['m1_policy']['min_selected_evidence_support']}`",
        "",
        "## Extractive baseline (EXP-003 selection, unchanged)",
        "",
    ]
    b = payload["extractive_summary"]
    lines.append(f"- Answered: {b['answered']}/{b['question_count']}, supported answers: {b['supported_answers']}")
    lines.append(f"- Supported statements: {b['supported_statement_count']}/{b['statement_count']} ({b['statement_support_rate']:.4f})")
    lines.append(f"- Supported citations: {b['supported_citation_count']}/{b['citation_count']} ({b['citation_support_rate']:.4f})")
    lines += ["", "## Gemini generation", ""]
    g = payload.get("gemini_summary")
    if g:
        lines.append(
            f"- Answered: {g['answered']}, model abstained: {g['model_abstained']}, "
            f"insufficient-evidence abstained: {g['insufficient_evidence_abstained']}, "
            f"blocked: {g.get('blocked', 0)}, errors: {g['errors']}"
        )
        if g.get("claim_count") is not None:
            lines.append(
                f"- Claims: {g['claim_count']}, grounded claims (>=0.80 overlap): "
                f"{g['grounded_claim_count']} ({g.get('grounded_claim_rate', 0):.4f})"
            )
            lines.append(
                f"- Uncited claims: {g['uncited_claim_count']}, invalid source IDs: {g['invalid_source_id_count']}"
            )
    else:
        lines.append("- Blocked/unavailable. See `gemini` in config/results.")
    lines += ["", "## Abstention (out-of-corpus)", ""]
    a = payload.get("abstention", {})
    for arm, s in a.items():
        o = s.get("overall", {})
        lines.append(f"- {arm}: {o.get('abstained')}/{o.get('count')} abstained")
    lines += ["", "## Notes", "", "- Retrieval/chunking/embeddings/ground truth unchanged from accepted baseline.", "- Use current 15-question set as initial evaluation only; not general production readiness.", "- Manual correctness/completeness/faithfulness/citation audit required for Gemini outputs.", ""]
    return "\n".join(lines)


def run(
    pdf_path: Path = PDF_PATH,
    questions_csv: Path = QUESTIONS_CSV,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    unanswerable_csv: Path = UNANSWERABLE_CSV,
    results_dir: Path = RESULTS_DIR,
    persist_dir: Path = PRODUCTION_PERSIST_DIR,
    collection_name: str = PRODUCTION_COLLECTION,
    top_k: int = TOP_K,
    model: Optional[str] = None,
    min_selected_evidence_support: float = DEFAULT_MIN_SELECTED_EVIDENCE_SUPPORT,
) -> Dict[str, Any]:
    policy = AbstentionPolicy()
    document = extract_source_document(pdf_path)
    chunks = FixedCharacterChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP).chunk(document)
    audit = audit_source_coverage(document, chunks, max_chars=CHUNK_SIZE, allow_overlap=True)
    if not audit["ok"]:
        raise RuntimeError(f"Chunk audit failed: {audit['issues']}")

    questions = load_questions(questions_csv)
    units, problems = load_exp001_evidence(ground_truth_path, document)
    if problems:
        raise RuntimeError(f"Ground-truth mapping failed: {problems[0]}")

    st_model = SentenceTransformer(EMBEDDING_MODEL)
    index = DenseIndex(
        DenseIndexConfig(collection_name=collection_name, persist_dir=persist_dir, embedding_model=EMBEDDING_MODEL),
        model=st_model,
    )
    pipeline = GroundedPipeline(index=index, top_k=top_k, policy=policy, sibling_score_ratio=0.5)

    # Gemini availability / selection
    gemini_info: Dict[str, Any] = {"available": _has_gemini_key(), "key_present": _has_gemini_key()}
    generator: Optional[GeminiGenerator] = None
    if gemini_info["available"]:
        try:
            available = _list_gemini_models()
            gemini_info["available_models"] = available
            selected, selection_meta = _select_gemini_model(available, preferred=model)
            gemini_info["selection"] = selection_meta
            gemini_info["selected_model"] = selected
            if selected:
                probe = _probe_model(selected)
                gemini_info["probe"] = probe
                if probe.get("ok"):
                    generator = GeminiGenerator(model=selected)
                else:
                    gemini_info["available"] = False
                    gemini_info["blocked_reason"] = "probe_failed"
            else:
                gemini_info["available"] = False
                gemini_info["blocked_reason"] = "no_suitable_model"
        except Exception as exc:  # noqa: BLE001
            gemini_info["available"] = False
            gemini_info["blocked_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        gemini_info["blocked_reason"] = "missing_api_key"
        gemini_info["selection_note"] = (
            "No GOOGLE_API_KEY/GEMINI_API_KEY found. Model selection requires "
            "actual availability; intended preference order recorded in config."
        )

    retrieval_rows = []
    extractive_rows = []
    gemini_rows = []
    for q in questions:
        evidence = evidence_for_question(units, q.question_id)
        hits = index.search(q.question, k=top_k)
        metrics = compute_retrieval_metrics(hits, evidence, k=top_k)
        retrieval_rows.append({
            "question_id": q.question_id,
            "question": q.question,
            "retrieved_chunks": [
                {
                    "rank": h.rank, "chunk_id": h.chunk_id, "section": h.section,
                    "start_offset": h.start_offset, "end_offset": h.end_offset,
                    "distance": h.distance, "evidence_ids": _overlapping_evidence(h.start_offset, h.end_offset, evidence),
                }
                for h in hits
            ],
            **{k: metrics[k] for k in metrics if k.startswith(("recall@", "precision@", "mrr@"))},
            "missing_evidence_ids": list(metrics.get("missing_evidence_ids", [])),
        })

        ex = pipeline.answer_from_hits(q.question, hits)
        ex_eval = evaluate_answer_support(ex, evidence)
        stmt_rate = round(ex_eval["supported_statement_count"] / ex_eval["statement_count"], 4) if ex_eval["statement_count"] else 0.0
        cit_rate = round(ex_eval["supported_citation_count"] / ex_eval["citation_count"], 4) if ex_eval["citation_count"] else 0.0
        extractive_rows.append({
            "question_id": q.question_id,
            "question": q.question,
            "expected_answer": q.expected_answer,
            "status": ex.status,
            "reason": ex.reason,
            "text": ex.text,
            "token_support": ex.token_support,
            "best_distance": ex.best_distance,
            "statements": ex_eval["statements"],
            "citations": ex_eval["citations"],
            "statement_count": ex_eval["statement_count"],
            "supported_statement_count": ex_eval["supported_statement_count"],
            "statement_support_rate": stmt_rate,
            "citation_count": ex_eval["citation_count"],
            "supported_citation_count": ex_eval["supported_citation_count"],
            "citation_support_rate": cit_rate,
            "answer_supported": ex_eval["answer_supported"],
        })

        selected_support = _selected_evidence_support(q.question, ex.statements)
        sufficient, insuff_reason = _evidence_sufficient(q.question, ex, min_selected_evidence_support)
        if generator is not None:
            g = generator.generate(
                q.question, ex.statements, ex.citations,
                evidence_sufficient=sufficient, insufficiency_reason=insuff_reason,
            )
            grow = g.to_dict()
        else:
            if not sufficient or ex.status == ABSTAINED or not ex.statements:
                grow = {
                    "status": MODEL_ABSTAINED,
                    "question": q.question,
                    "text": INSUFFICIENT_EVIDENCE if not sufficient or not ex.statements else ex.text,
                    "claims": [],
                    "citations": [],
                    "evidence_statements": [s.to_dict() for s in ex.statements],
                    "raw_response": "",
                    "model": gemini_info.get("selected_model"),
                    "latency_ms": 0.0,
                    "errors": [],
                    "metadata": {
                        "blocked": not gemini_info.get("available", False),
                        "blocked_reason": gemini_info.get("blocked_reason"),
                        "evidence_sufficient": sufficient,
                        "insufficiency_reason": insuff_reason,
                        "selected_evidence_support": round(selected_support, 4),
                        "prompt_version": GEMINI_PROMPT_VERSION,
                    },
                }
            else:
                grow = {
                    "status": GENERATION_ERROR,
                    "question": q.question,
                    "text": "",
                    "claims": [],
                    "citations": [],
                    "evidence_statements": [s.to_dict() for s in ex.statements],
                    "raw_response": "",
                    "model": gemini_info.get("selected_model"),
                    "latency_ms": 0.0,
                    "errors": [f"gemini_unavailable:{gemini_info.get('blocked_reason')}"],
                    "metadata": {
                        "blocked": True,
                        "blocked_reason": gemini_info.get("blocked_reason"),
                        "evidence_sufficient": sufficient,
                        "insufficiency_reason": insuff_reason,
                        "selected_evidence_support": round(selected_support, 4),
                        "prompt_version": GEMINI_PROMPT_VERSION,
                    },
                }
        grow.update({
            "question_id": q.question_id,
            "expected_answer": q.expected_answer,
            "selected_evidence_support": round(selected_support, 4),
            "evidence_sufficient_for_generation": sufficient,
            "insufficiency_reason": insuff_reason,
        })
        grow.update(_evaluate_generated(grow))
        gemini_rows.append(grow)

    # Summaries
    ex_answered = [r for r in extractive_rows if r["status"] == ANSWERED]
    extractive_summary = {
        "question_count": len(extractive_rows),
        "answered": len(ex_answered),
        "abstained": len(extractive_rows) - len(ex_answered),
        "supported_answers": sum(1 for r in ex_answered if r.get("answer_supported")),
        "statement_count": sum(r["statement_count"] for r in ex_answered),
        "supported_statement_count": sum(r["supported_statement_count"] for r in ex_answered),
        "statement_support_rate": round(sum(r["supported_statement_count"] for r in ex_answered) / sum(r["statement_count"] for r in ex_answered), 4) if sum(r["statement_count"] for r in ex_answered) else 0.0,
        "citation_count": sum(r["citation_count"] for r in ex_answered),
        "supported_citation_count": sum(r["supported_citation_count"] for r in ex_answered),
        "citation_support_rate": round(sum(r["supported_citation_count"] for r in ex_answered) / sum(r["citation_count"] for r in ex_answered), 4) if sum(r["citation_count"] for r in ex_answered) else 0.0,
    }

    g_answered = [r for r in gemini_rows if r.get("status") == ANSWERED]
    g_abstained = [r for r in gemini_rows if r.get("status") == MODEL_ABSTAINED]
    g_errors = [r for r in gemini_rows if r.get("status") == GENERATION_ERROR and not r.get("metadata", {}).get("blocked")]
    g_blocked = [r for r in gemini_rows if r.get("metadata", {}).get("blocked")]
    all_claims = sum(r.get("claim_count", 0) for r in gemini_rows)
    grounded_claims = sum(r.get("grounded_claim_count", 0) for r in gemini_rows)
    gemini_summary = {
        "answered": len(g_answered),
        "model_abstained": len([
            r for r in g_abstained
            if r.get("evidence_sufficient_for_generation", True) and not r.get("metadata", {}).get("blocked")
        ]),
        "insufficient_evidence_abstained": len([
            r for r in g_abstained if not r.get("evidence_sufficient_for_generation", True)
        ]),
        "blocked": len(g_blocked),
        "errors": len(g_errors),
        "claim_count": all_claims,
        "grounded_claim_count": grounded_claims,
        "grounded_claim_rate": round(grounded_claims / all_claims, 4) if all_claims else 0.0,
        "uncited_claim_count": sum(r.get("uncited_claim_count", 0) for r in gemini_rows),
        "invalid_source_id_count": sum(r.get("invalid_source_id_count", 0) for r in gemini_rows),
    }

    # Abstention evaluation
    unanswerable = _load_unanswerable(unanswerable_csv)
    abstention = {"extractive": {"results": []}, "gemini": {"results": []}}
    for item in unanswerable:
        ex = pipeline.answer(item["question"])
        abstention["extractive"]["results"].append({**item, "status": ex.status, "abstained": ex.status == ABSTAINED, "reason": ex.reason})
        # Gemini path: if extractive abstains or no statements, abstain without generation (and no API call possible in blocked mode)
        ex_sufficient, ex_reason = _evidence_sufficient(item["question"], ex, min_selected_evidence_support)
        g_abst = (ex.status == ABSTAINED) or (not ex_sufficient) or (not ex.statements)
        abstention["gemini"]["results"].append({
            **item, "status": MODEL_ABSTAINED if g_abst else ANSWERED, "abstained": g_abst,
            "reason": ex.reason or ex_reason, "selected_evidence_support": round(_selected_evidence_support(item["question"], ex.statements), 4),
        })
    for arm in abstention:
        rows = abstention[arm]["results"]
        by_split = {}
        for split in sorted({r["split"] for r in rows}):
            srows = [r for r in rows if r["split"] == split]
            n = sum(1 for r in srows if r["abstained"])
            by_split[split] = {"count": len(srows), "abstained": n, "abstention_rate": round(n / len(srows), 4)}
        overall_n = sum(1 for r in rows if r["abstained"])
        abstention[arm]["summary"] = {"by_split": by_split, "overall": {"count": len(rows), "abstained": overall_n, "abstention_rate": round(overall_n / len(rows), 4)}}

    # Comparison vs EXP-004 if available
    exp004_compare = None
    exp004_path = ROOT / "experiments/results/exp004/comparison.json"
    if exp004_path.exists():
        try:
            exp004 = json.loads(exp004_path.read_text(encoding="utf-8"))
            exp004_compare = {
                "available": True,
                "exp004_decision": exp004.get("decision"),
                "note": "EXP-004 used local qwen2.5:1.5b over same selected evidence; compare claim grounding/abstention/errors and manual audit qualitatively.",
            }
        except Exception as exc:  # noqa: BLE001
            exp004_compare = {"available": False, "error": str(exc)}

    config = {
        "experiment_id": EXPERIMENT_ID,
        "pdf_path": str(pdf_path),
        "questions_csv": str(questions_csv),
        "ground_truth": str(ground_truth_path),
        "unanswerable_csv": str(unanswerable_csv),
        "embedding_model": EMBEDDING_MODEL,
        "chunking": {"strategy": "fixed_character", "chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "chunk_count": len(chunks)},
        "retrieval": {"method": "dense_vector_similarity", "top_k": top_k, "changed": False},
        "selection": {"method": "accepted_extractive_pipeline", "sibling_score_ratio": 0.5, "max_answer_sentences": 3, "changed": False},
        "generation": {
            "backend": "google_gemini",
            "selected_model": gemini_info.get("selected_model"),
            "available": gemini_info.get("available", False),
            "prompt_version": GEMINI_PROMPT_VERSION,
            "temperature": DEFAULT_GEMINI_TEMPERATURE,
            "max_output_tokens": DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
            "input": "selected_verbatim_evidence_only",
            "citation_resolution": "application_validated_source_ids_to_actual_metadata",
            "model_preference": MODEL_PREFERENCE,
            "changed": True,
        },
        "m1_policy": {
            "min_selected_evidence_support": min_selected_evidence_support,
            "abstain_if_pipeline_abstained": True,
            "abstain_if_no_selected_statements": True,
            "abstain_if_selected_evidence_insufficient": True,
            "rationale": "Trustworthy M1 path must not generate when selected evidence is insufficient (e.g. Q05 in-corpus retrieval failure).",
        },
        "gemini": gemini_info,
        "exp004_comparison": exp004_compare,
    }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "extractive_summary": extractive_summary,
        "gemini_summary": gemini_summary,
        "abstention": {k: v["summary"] for k, v in abstention.items()},
        "gemini": gemini_info,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "config.json", config)
    _write_json(results_dir / "integrity_audit.json", audit)
    _write_json(results_dir / "retrieval_results.json", {"experiment_id": EXPERIMENT_ID, "retrieval_results": retrieval_rows})
    _write_json(results_dir / "answers_extractive.json", {"experiment_id": EXPERIMENT_ID, "summary": extractive_summary, "answers": extractive_rows})
    _write_json(results_dir / "answers_gemini.json", {"experiment_id": EXPERIMENT_ID, "summary": gemini_summary, "answers": gemini_rows})
    _write_json(results_dir / "abstention_results.json", abstention)

    # Manual audit template
    template_path = results_dir / "manual_audit_template.csv"
    with open(template_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "question", "expected_answer", "gemini_status", "gemini_text", "correctness(correct/partial/incorrect)", "completeness(complete/partial/missing)", "faithfulness(grounded/partial/unfaithful)", "citation_quality(correct/partial/incorrect/na)", "notes"])
        for r in gemini_rows:
            qtext = next((q.question for q in questions if q.question_id == r["question_id"]), "")
            w.writerow([r["question_id"], qtext, r.get("expected_answer", ""), r.get("status"), r.get("text", ""), "", "", "", "", ""])

    # Per-question comparison CSV
    with open(results_dir / "per_question_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "retrieval_recall@5", "extractive_status", "extractive_supported", "extractive_supported_statements", "selected_evidence_support", "evidence_sufficient_for_generation", "gemini_status", "gemini_claims", "gemini_grounded_claim_rate", "gemini_uncited_claims", "gemini_invalid_source_ids"])
        ret = {r["question_id"]: r for r in retrieval_rows}
        ex = {r["question_id"]: r for r in extractive_rows}
        gm = {r["question_id"]: r for r in gemini_rows}
        for q in questions:
            qid = q.question_id
            w.writerow([
                qid, ret[qid].get(f"recall@{top_k}"), ex[qid]["status"], ex[qid]["answer_supported"],
                ex[qid]["supported_statement_count"], gm[qid]["selected_evidence_support"],
                gm[qid]["evidence_sufficient_for_generation"], gm[qid]["status"], gm[qid].get("claim_count", 0),
                gm[qid].get("grounded_claim_rate", 0), gm[qid].get("uncited_claim_count", 0), gm[qid].get("invalid_source_id_count", 0),
            ])

    summary = _summary_md(payload)
    (results_dir / "summary.md").write_text(summary, encoding="utf-8")
    _write_json(results_dir / "summary.json", payload)

    print(json.dumps(payload, indent=2, default=str))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Gemini model override")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--min-selected-evidence-support", type=float, default=DEFAULT_MIN_SELECTED_EVIDENCE_SUPPORT)
    args = parser.parse_args()
    run(model=args.model, results_dir=Path(args.results_dir), min_selected_evidence_support=args.min_selected_evidence_support)


if __name__ == "__main__":
    main()
