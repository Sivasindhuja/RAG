"""EXP-004: constrained generator evaluation over selected evidence.

Hypothesis
----------
A small local model can rewrite the already-selected verbatim evidence into a
shorter, more direct answer while preserving citation validity and without
adding unsupported policy claims.

Controlled variable
-------------------
Answer synthesis only:

- baseline: accepted extractive answer (verbatim selected sentences);
- candidate: local Ollama model rewrites only those selected sentences into
  JSON claims; each claim must name the selected evidence sentence(s) it uses.

Retrieval, chunking, embedding, selection, answer length cap, abstention policy
and evidence annotations are held fixed.  The candidate never sees unselected
retrieved chunks, so it cannot recover evidence the selector missed.

Pre-registered mechanical gate
------------------------------
1. all 12 out-of-corpus questions abstain (12/12);
2. no generation errors and all 15 positive responses parse as JSON;
3. zero invalid source IDs and zero uncited claims;
4. at least 90% of claims have >=0.80 content-token coverage by their cited
   selected evidence;
5. at least 12/15 positives receive a generated answer (the rest may abstain).

A manual audit then checks correctness and material faithfulness against the
expected answer and selected evidence.  Passing the mechanical gate alone does
not make the generator the production default.

Run:
    python -m experiments.exp004
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from sentence_transformers import SentenceTransformer

from experiments.production_slice import (
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GROUND_TRUTH_PATH,
    OVERLAP,
    PDF_PATH,
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
    AnswerStatement,
    DenseIndex,
    DenseIndexConfig,
    FixedCharacterChunker,
    GENERATION_ERROR,
    GroundedPipeline,
    MODEL_ABSTAINED,
    OllamaGenerator,
    audit_source_coverage,
    compute_retrieval_metrics,
    content_tokens,
    evaluate_answer_support,
    evidence_for_question,
    extract_source_document,
    load_exp001_evidence,
    load_questions,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ID = "EXP-004"
RESULTS_DIR = ROOT / "experiments" / "results" / "exp004"
PERSIST_DIR = ROOT / "chroma_db_exp004"
COLLECTION_NAME = "satcom_ngp_exp004_constrained_generator"
GENERATOR_MODEL = "qwen2.5:1.5b"
GENERATOR_OPTIONS = {
    "temperature": 0,
    "top_p": 0.9,
    "num_ctx": 4096,
    "num_predict": 220,
    "repeat_penalty": 1.1,
}
POLICY_ABSTAINED = "policy_abstained"
GROUNDING_THRESHOLD = 0.80
GROUNDING_RATE_THRESHOLD = 0.90


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _word_windows(text: str, size: int = 8) -> List[str]:
    words = text.lower().split()
    if len(words) < size:
        return [" ".join(words)] if words else []
    return [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]


def _verbatim_span_match(claim: str, source: str, size: int = 8) -> bool:
    windows = _word_windows(claim, size)
    if not windows:
        return False
    source_words = source.lower()
    return any(window in source_words for window in windows)


def _token_coverage(text: str, source: str) -> float:
    claim_tokens = set(content_tokens(text))
    if not claim_tokens:
        return 0.0
    source_tokens = set(content_tokens(source))
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def _expected_token_recall(text: str, expected: str) -> float:
    expected_tokens = set(content_tokens(expected))
    if not expected_tokens:
        return 1.0
    return len(expected_tokens & set(content_tokens(text))) / len(expected_tokens)


def _citation_overlaps_evidence(citation, evidence) -> bool:
    return any(
        unit.start_offset < citation.end_offset and citation.start_offset < unit.end_offset
        for unit in evidence
    )


def _evaluate_generated(
    generated,
    evidence,
    statements: Sequence[AnswerStatement],
) -> Dict[str, Any]:
    if generated.status != ANSWERED:
        return {
            "status": generated.status,
            "claim_count": 0,
            "cited_claim_count": 0,
            "uncited_claim_count": 0,
            "invalid_source_count": 0,
            "duplicate_claim_count": 0,
            "claims_with_coverage_ge_0_8": 0,
            "claim_grounding_rate": 0.0,
            "mean_cited_coverage": 0.0,
            "min_cited_coverage": 0.0,
            "verbatim_claim_count": 0,
            "citation_count": 0,
            "evidence_overlapping_citation_count": 0,
            "citation_evidence_rate": 0.0,
            "expected_token_recall": 0.0,
            "word_count": 0,
            "truncated": generated.metadata.get("done_reason") == "length",
            "errors": list(generated.errors),
            "claims": [],
        }

    claim_rows: List[Dict[str, Any]] = []
    seen_claims: set[str] = set()
    duplicate_claims = 0
    for index, claim in enumerate(generated.claims, 1):
        valid_statements = [
            statements[source_id - 1]
            for source_id in claim.source_ids
            if 1 <= source_id <= len(statements)
        ]
        cited_text = " ".join(statement.text for statement in valid_statements)
        all_text = " ".join(statement.text for statement in statements)
        coverage = _token_coverage(claim.text, cited_text)
        normalised = _normalise(claim.text)
        if normalised in seen_claims:
            duplicate_claims += 1
        seen_claims.add(normalised)
        claim_rows.append(
            {
                "claim_index": index,
                "text": claim.text,
                "source_ids": list(claim.source_ids),
                "markers": list(claim.markers),
                "invalid_source_ids": list(claim.invalid_source_ids),
                "cited_chunk_ids": sorted({statement.chunk_id for statement in valid_statements}),
                "cited_coverage": round(coverage, 4),
                "all_selected_coverage": round(_token_coverage(claim.text, all_text), 4),
                "verbatim_span_match": _verbatim_span_match(claim.text, cited_text),
            }
        )

    cited_claims = [row for row in claim_rows if row["markers"]]
    coverages = [row["cited_coverage"] for row in claim_rows]
    evidence_overlap = sum(
        1 for citation in generated.citations if _citation_overlaps_evidence(citation, evidence)
    )
    citation_count = len(generated.citations)
    return {
        "status": generated.status,
        "claim_count": len(claim_rows),
        "cited_claim_count": len(cited_claims),
        "uncited_claim_count": len(claim_rows) - len(cited_claims),
        "invalid_source_count": sum(len(row["invalid_source_ids"]) for row in claim_rows),
        "duplicate_claim_count": duplicate_claims,
        "claims_with_coverage_ge_0_8": sum(
            1 for row in claim_rows if row["cited_coverage"] >= GROUNDING_THRESHOLD
        ),
        "claim_grounding_rate": (
            sum(1 for row in claim_rows if row["cited_coverage"] >= GROUNDING_THRESHOLD)
            / len(claim_rows)
            if claim_rows
            else 0.0
        ),
        "mean_cited_coverage": round(sum(coverages) / len(coverages), 4) if coverages else 0.0,
        "min_cited_coverage": round(min(coverages), 4) if coverages else 0.0,
        "verbatim_claim_count": sum(1 for row in claim_rows if row["verbatim_span_match"]),
        "citation_count": citation_count,
        "evidence_overlapping_citation_count": evidence_overlap,
        "citation_evidence_rate": round(evidence_overlap / citation_count, 4) if citation_count else 0.0,
        "expected_token_recall": 0.0,
        "word_count": len(generated.text.split()),
        "truncated": generated.metadata.get("done_reason") == "length",
        "errors": list(generated.errors),
        "claims": claim_rows,
    }


def _answer_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answered = [row for row in rows if row["status"] == ANSWERED]
    statement_count = sum(row["statement_count"] for row in answered)
    supported_statement_count = sum(row["supported_statement_count"] for row in answered)
    citation_count = sum(row["citation_count"] for row in answered)
    supported_citation_count = sum(row["supported_citation_count"] for row in answered)
    return {
        "question_count": len(rows),
        "answered": len(answered),
        "abstained": len(rows) - len(answered),
        "supported_answers": sum(1 for row in answered if row["answer_supported"]),
        "statement_count": statement_count,
        "supported_statement_count": supported_statement_count,
        "statement_support_rate": supported_statement_count / statement_count if statement_count else 0.0,
        "citation_count": citation_count,
        "supported_citation_count": supported_citation_count,
        "citation_support_rate": supported_citation_count / citation_count if citation_count else 0.0,
        "mean_statements": statement_count / len(answered) if answered else 0.0,
        "mean_citations": citation_count / len(answered) if answered else 0.0,
    }


def _generator_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answered = [row for row in rows if row["evaluation"]["status"] == ANSWERED]
    claim_count = sum(row["evaluation"]["claim_count"] for row in answered)
    cited_claim_count = sum(row["evaluation"]["cited_claim_count"] for row in answered)
    return {
        "question_count": len(rows),
        "answered": len(answered),
        "model_abstained": sum(1 for row in rows if row["evaluation"]["status"] == MODEL_ABSTAINED),
        "generation_errors": sum(1 for row in rows if row["evaluation"]["status"] == GENERATION_ERROR),
        "claim_count": claim_count,
        "cited_claim_count": cited_claim_count,
        "uncited_claim_count": sum(row["evaluation"]["uncited_claim_count"] for row in answered),
        "invalid_source_count": sum(row["evaluation"]["invalid_source_count"] for row in answered),
        "duplicate_claim_count": sum(row["evaluation"]["duplicate_claim_count"] for row in answered),
        "claims_with_coverage_ge_0_8": sum(
            row["evaluation"]["claims_with_coverage_ge_0_8"] for row in answered
        ),
        "claim_grounding_rate": (
            sum(row["evaluation"]["claims_with_coverage_ge_0_8"] for row in answered) / claim_count
            if claim_count
            else 0.0
        ),
        "mean_cited_coverage": round(
            sum(row["evaluation"]["mean_cited_coverage"] for row in answered) / len(answered), 4
        )
        if answered
        else 0.0,
        "citation_count": sum(row["evaluation"]["citation_count"] for row in answered),
        "evidence_overlapping_citation_count": sum(
            row["evaluation"]["evidence_overlapping_citation_count"] for row in answered
        ),
        "expected_token_recall": round(
            sum(row["evaluation"]["expected_token_recall"] for row in answered) / len(answered), 4
        )
        if answered
        else 0.0,
        "mean_words": round(
            sum(row["evaluation"]["word_count"] for row in answered) / len(answered), 2
        )
        if answered
        else 0.0,
        "truncated": sum(1 for row in answered if row["evaluation"]["truncated"]),
    }


def _abstention_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, Dict[str, Any]] = {}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        abstained = sum(1 for row in split_rows if row["abstained"])
        by_split[split] = {
            "count": len(split_rows),
            "abstained": abstained,
            "abstention_rate": round(abstained / len(split_rows), 4) if split_rows else 0.0,
        }
    total = sum(1 for row in rows if row["abstained"])
    return {
        "by_split": by_split,
        "overall": {
            "count": len(rows),
            "abstained": total,
            "abstention_rate": round(total / len(rows), 4) if rows else 0.0,
        },
    }


def _mechanical_decision(
    generator_summary: Dict[str, Any],
    abstention: Dict[str, Any],
) -> Dict[str, Any]:
    checks = {
        "negative_abstention_12_of_12": (
            abstention["overall"]["count"] == 12 and abstention["overall"]["abstained"] == 12
        ),
        "no_generation_errors": generator_summary["generation_errors"] == 0,
        "all_positives_parse": (
            generator_summary["answered"] + generator_summary["model_abstained"]
            == generator_summary["question_count"]
        ),
        "zero_invalid_sources": generator_summary["invalid_source_count"] == 0,
        "zero_uncited_claims": generator_summary["uncited_claim_count"] == 0,
        "claim_grounding_rate_ge_0_9": (
            generator_summary["claim_grounding_rate"] >= GROUNDING_RATE_THRESHOLD
        ),
        "at_least_12_positives_answered": generator_summary["answered"] >= 12,
    }
    return {
        "mechanical_pass": all(checks.values()),
        "checks": checks,
        "manual_audit_required": True,
        "note": "Passing the mechanical gate does not establish correctness or faithfulness.",
    }


def _summary(
    config: Dict[str, Any],
    extractive_summary: Dict[str, Any],
    generator_summary: Dict[str, Any],
    abstention: Dict[str, Any],
    decision: Dict[str, Any],
    comparison: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "# EXP-004 — Constrained generator evaluation",
        "",
        f"- Experiment id: `{EXPERIMENT_ID}`",
        f"- Generator: `{config['generator']['model']}` (local Ollama, selected evidence only)",
        "- Retrieval/selection: accepted PROD-001/EXP-003 path, unchanged",
        "- Candidate input: only the extractive selector's verbatim evidence sentences",
        "- Mechanical gate: **" + ("PASS" if decision["mechanical_pass"] else "FAIL") + "**",
        "- Manual correctness/faithfulness audit: see `manual_audit.json` / `manual_audit_summary.json`",
        "",
        "## Extractive baseline vs generated synthesis",
        "",
        "| Measure | Extractive | Generated |",
        "|---|---:|---:|",
        f"| Answered | {extractive_summary['answered']}/15 | {generator_summary['answered']}/15 |",
        f"| Model abstained | — | {generator_summary['model_abstained']}/15 |",
        f"| Generation errors | — | {generator_summary['generation_errors']}/15 |",
        f"| Statements / claims | {extractive_summary['statement_count']} | {generator_summary['claim_count']} |",
        f"| Citations | {extractive_summary['citation_count']} | {generator_summary['citation_count']} |",
        f"| Mean words per answer | — | {generator_summary['mean_words']:.2f} |",
        f"| Mean cited-token coverage | — | {generator_summary['mean_cited_coverage']:.4f} |",
        f"| Claims grounded at >=0.80 | — | {generator_summary['claims_with_coverage_ge_0_8']}/{generator_summary['claim_count']} |",
        f"| Uncited claims | — | {generator_summary['uncited_claim_count']} |",
        f"| Invalid source IDs | — | {generator_summary['invalid_source_count']} |",
        f"| Duplicate claims | — | {generator_summary['duplicate_claim_count']} |",
        f"| Truncated generations | — | {generator_summary['truncated']} |",
        "",
        "## Mechanical gate",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in decision["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    recall_key = f"recall@{config['retrieval']['top_k']}"
    lines += [
        "",
        "## Per-question mechanical view",
        "",
        f"| ID | Recall@{config['retrieval']['top_k']} | Extractive supported | Gen status | Claims/cited | Grounded >=0.80 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['question_id']} | {row[recall_key]:.4f} | "
            f"{row['extractive_supported_statements']}/{row['extractive_statements']} | "
            f"{row['generator_status']} | {row['generator_cited_claims']}/{row['generator_claims']} | "
            f"{row['generator_claims_grounded']}/{row['generator_claims']} |"
        )
    lines += [
        "",
        "## Abstention smoke test",
        "",
        f"Extractive: {abstention['overall']['abstained']}/{abstention['overall']['count']} abstained.",
        "",
        "The generator is never called when the shared abstention policy already declines",
        "the question, so the negative set measures the same gate for both arms.",
        "",
        "## Interpretation limits",
        "",
        "- The generator sees only selected sentences. It cannot fix Q05/Q08 retrieval",
        "  misses or Q10's wrong selection; those remain upstream failures.",
        "- Cited-token coverage is a lexical proxy, not entailment or human faithfulness.",
        "- Expected-answer token recall is a weak surface metric and is not used as a",
        "  decision gate.",
        "- The manual audit in `manual_audit.json` is a single-reviewer judgment, not",
        "  an independent gold standard.",
        "- A local 1.5B model is evaluated here; results do not generalise to larger",
        "  hosted models.",
        "",
        "Artifacts: `answers_extractive.json`, `answers_generated.json`,",
        "`comparison.json`, `per_question_metrics.csv`, `manual_audit_template.csv`.",
        "",
    ]
    return "\n".join(lines)


def run(
    pdf_path: Path = PDF_PATH,
    questions_csv: Path = QUESTIONS_CSV,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    unanswerable_csv: Path = UNANSWERABLE_CSV,
    results_dir: Path = RESULTS_DIR,
    persist_dir: Path = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    top_k: int = TOP_K,
    build_index: bool = True,
    policy: Optional[AbstentionPolicy] = None,
    generator_model: str = GENERATOR_MODEL,
    generator_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy = policy or AbstentionPolicy()
    generator_options = dict(generator_options or GENERATOR_OPTIONS)
    document = extract_source_document(pdf_path)
    chunks = FixedCharacterChunker(chunk_size=chunk_size, overlap=overlap).chunk(document)
    audit = audit_source_coverage(document, chunks, max_chars=chunk_size, allow_overlap=True)
    if not audit["ok"]:
        raise RuntimeError(f"Fixed-character chunk audit failed: {audit['issues']}")

    questions = load_questions(questions_csv)
    units, problems = load_exp001_evidence(ground_truth_path, document)
    if problems:
        raise RuntimeError(f"Ground-truth mapping failed: {problems[0]}")
    if {unit.question_id for unit in units} != {question.question_id for question in questions}:
        raise RuntimeError("Evidence manifest question ids do not match the approved question set")

    model = SentenceTransformer(EMBEDDING_MODEL)
    index = DenseIndex(
        DenseIndexConfig(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_model=EMBEDDING_MODEL,
        ),
        model=model,
    )
    build_stats = index.build(chunks, reset=True) if build_index else {"chunks": index.count()}

    extractive_pipeline = GroundedPipeline(index=index, top_k=top_k, policy=policy)
    generator = OllamaGenerator(model=generator_model, options=generator_options)

    retrieval_rows: List[Dict[str, Any]] = []
    extractive_rows: List[Dict[str, Any]] = []
    generated_rows: List[Dict[str, Any]] = []
    comparison: List[Dict[str, Any]] = []
    extractive_latencies: List[float] = []
    generation_latencies: List[float] = []

    for question in questions:
        evidence = evidence_for_question(units, question.question_id)
        if not evidence:
            raise RuntimeError(f"No verified evidence for {question.question_id}")

        hits = index.search(question.question, k=top_k)
        metrics = compute_retrieval_metrics(hits, evidence, k=top_k)
        retrieval_rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "retrieved_chunks": [
                    {
                        "rank": hit.rank,
                        "chunk_id": hit.chunk_id,
                        "section": hit.section,
                        "start_page": hit.start_page,
                        "end_page": hit.end_page,
                        "start_offset": hit.start_offset,
                        "end_offset": hit.end_offset,
                        "distance": hit.distance,
                        "evidence_ids": _overlapping_evidence(hit.start_offset, hit.end_offset, evidence),
                    }
                    for hit in hits
                ],
                **{key: metrics[key] for key in metrics if key.startswith(("recall@", "precision@", "mrr@"))},
            }
        )

        started = time.perf_counter()
        extractive = extractive_pipeline.answer_from_hits(question.question, hits)
        extractive_latencies.append((time.perf_counter() - started) * 1000.0)
        extractive_row = {
            "question_id": question.question_id,
            "question": question.question,
            "expected_answer": question.expected_answer,
            "relevant_clause_ids": list(question.relevant_clause_ids),
            **evaluate_answer_support(extractive, evidence),
        }
        extractive_rows.append(extractive_row)

        if extractive.status == ABSTAINED:
            generated_row = {
                "question_id": question.question_id,
                "question": question.question,
                "expected_answer": question.expected_answer,
                "relevant_clause_ids": list(question.relevant_clause_ids),
                "generator": {
                    "status": POLICY_ABSTAINED,
                    "text": "",
                    "claims": [],
                    "citations": [],
                    "errors": [],
                    "metadata": {},
                },
                "evaluation": _evaluate_generated(
                    SimpleNamespace(
                        status=POLICY_ABSTAINED,
                        text="",
                        claims=[],
                        citations=[],
                        errors=[],
                        metadata={},
                    ),
                    evidence,
                    [],
                ),
            }
            generated_rows.append(generated_row)
            comparison.append(
                {
                    "question_id": question.question_id,
                    f"recall@{top_k}": metrics[f"recall@{top_k}"],
                    "extractive_status": extractive.status,
                    "extractive_statements": extractive_row["statement_count"],
                    "extractive_supported_statements": extractive_row["supported_statement_count"],
                    "generator_status": POLICY_ABSTAINED,
                    "generator_claims": 0,
                    "generator_cited_claims": 0,
                    "generator_claims_grounded": 0,
                }
            )
            continue

        if not extractive.statements:
            generated_eval = _evaluate_generated(
                SimpleNamespace(
                    status=GENERATION_ERROR,
                    text="",
                    claims=[],
                    citations=[],
                    errors=["extractive pipeline returned no selected evidence"],
                    metadata={},
                ),
                evidence,
                [],
            )
            generated_row = {
                "question_id": question.question_id,
                "question": question.question,
                "expected_answer": question.expected_answer,
                "relevant_clause_ids": list(question.relevant_clause_ids),
                "generator": {
                    "status": GENERATION_ERROR,
                    "text": "",
                    "claims": [],
                    "citations": [],
                    "errors": ["extractive pipeline returned no selected evidence"],
                    "metadata": {},
                },
                "evaluation": generated_eval,
            }
            generated_rows.append(generated_row)
            comparison.append(
                {
                    "question_id": question.question_id,
                    f"recall@{top_k}": metrics[f"recall@{top_k}"],
                    "extractive_status": extractive.status,
                    "extractive_statements": extractive_row["statement_count"],
                    "extractive_supported_statements": extractive_row["supported_statement_count"],
                    "generator_status": GENERATION_ERROR,
                    "generator_claims": 0,
                    "generator_cited_claims": 0,
                    "generator_claims_grounded": 0,
                }
            )
            continue

        generated = generator.generate(question.question, extractive.statements, extractive.citations)
        generation_latencies.append(generated.latency_ms)
        generated_eval = _evaluate_generated(generated, evidence, extractive.statements)
        generated_eval["expected_token_recall"] = round(
            _expected_token_recall(generated.text, question.expected_answer), 4
        )
        generated_row = {
            "question_id": question.question_id,
            "question": question.question,
            "expected_answer": question.expected_answer,
            "relevant_clause_ids": list(question.relevant_clause_ids),
            "generator": generated.to_dict(),
            "evaluation": generated_eval,
        }
        generated_rows.append(generated_row)
        comparison.append(
            {
                "question_id": question.question_id,
                f"recall@{top_k}": metrics[f"recall@{top_k}"],
                "extractive_status": extractive.status,
                "extractive_statements": extractive_row["statement_count"],
                "extractive_supported_statements": extractive_row["supported_statement_count"],
                "generator_status": generated.status,
                "generator_claims": generated_eval["claim_count"],
                "generator_cited_claims": generated_eval["cited_claim_count"],
                "generator_claims_grounded": generated_eval["claims_with_coverage_ge_0_8"],
                "generator_expected_token_recall": generated_eval["expected_token_recall"],
                "generator_words": generated_eval["word_count"],
            }
        )

    unanswerable = _load_unanswerable(unanswerable_csv)
    abstention_rows: List[Dict[str, Any]] = []
    for item in unanswerable:
        started = time.perf_counter()
        answer = extractive_pipeline.answer(item["question"])
        latency = (time.perf_counter() - started) * 1000.0
        abstention_rows.append(
            {
                **item,
                "status": answer.status,
                "abstained": answer.status == ABSTAINED,
                "reason": answer.reason,
                "best_distance": answer.best_distance,
                "token_support": round(answer.token_support, 4),
                "latency_ms": round(latency, 2),
            }
        )
    abstention = _abstention_summary(abstention_rows)

    extractive_summary = _answer_summary(extractive_rows)
    generator_summary = _generator_summary(generated_rows)
    decision = _mechanical_decision(generator_summary, abstention)

    config = {
        "experiment_id": EXPERIMENT_ID,
        "pdf_path": str(pdf_path),
        "questions_csv": str(questions_csv),
        "ground_truth": str(ground_truth_path),
        "unanswerable_csv": str(unanswerable_csv),
        "embedding_model": EMBEDDING_MODEL,
        "chunking": {
            "strategy": "fixed_character",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "step": chunk_size - overlap,
            "per_page": True,
            "chunk_count": len(chunks),
        },
        "retrieval": {"method": "dense_vector_similarity", "top_k": top_k, "changed": False},
        "selection": {
            "method": "accepted_extractive_pipeline",
            "sibling_score_ratio": extractive_pipeline.sibling_score_ratio,
            "max_answer_sentences": extractive_pipeline.max_answer_sentences,
            "changed": False,
        },
        "generator": {
            "backend": "ollama",
            "model": generator_model,
            "options": generator_options,
            "input": "selected_verbatim_evidence_only",
            "contract": {
                "format": "json",
                "schema": {"sentences": [{"text": "string", "sources": ["evidence number"]}]},
            },
            "changed": True,
        },
        "answering_policy": asdict(policy),
        "decision_rule": {
            "negative_abstention": "12/12",
            "generation_errors": 0,
            "invalid_source_ids": 0,
            "uncited_claims": 0,
            "claim_grounding_threshold": GROUNDING_THRESHOLD,
            "claim_grounding_rate_min": GROUNDING_RATE_THRESHOLD,
            "minimum_answered_positives": 12,
            "manual_audit_required": True,
        },
        "vector_db": {
            "backend": "chroma",
            "collection_name": collection_name,
            "persist_directory": str(persist_dir),
        },
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "config.json", config)
    _write_json(results_dir / "integrity_audit.json", audit)
    _write_json(
        results_dir / "retrieval_results.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "build_stats": build_stats,
            "retrieval_results": retrieval_rows,
        },
    )
    _write_json(
        results_dir / "answers_extractive.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "arm": "extractive_baseline",
            "summary": extractive_summary,
            "answers": extractive_rows,
        },
    )
    _write_json(
        results_dir / "answers_generated.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "arm": "selected_evidence_generator",
            "generator": {"model": generator_model, "options": generator_options},
            "summary": generator_summary,
            "answers": generated_rows,
        },
    )
    _write_json(
        results_dir / "abstention_results.json",
        {"experiment_id": EXPERIMENT_ID, "summary": abstention, "results": abstention_rows},
    )
    _write_json(
        results_dir / "comparison.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "extractive_summary": extractive_summary,
            "generator_summary": generator_summary,
            "mechanical_decision": decision,
            "per_question": comparison,
        },
    )

    with open(results_dir / "per_question_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_id",
                f"recall@{top_k}",
                "extractive_supported_statements",
                "extractive_statements",
                "generator_status",
                "generator_claims",
                "generator_cited_claims",
                "generator_claims_grounded",
                "generator_expected_token_recall",
                "generator_words",
            ]
        )
        for row in comparison:
            writer.writerow(
                [
                    row["question_id"],
                    row[f"recall@{top_k}"],
                    row["extractive_supported_statements"],
                    row["extractive_statements"],
                    row["generator_status"],
                    row["generator_claims"],
                    row["generator_cited_claims"],
                    row["generator_claims_grounded"],
                    row.get("generator_expected_token_recall", 0.0),
                    row.get("generator_words", 0),
                ]
            )

    with open(results_dir / "manual_audit_template.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_id",
                "question",
                "expected_answer",
                "extractive_answer",
                "generator_answer",
                "generator_claims",
                "baseline_correctness",
                "generator_correctness",
                "baseline_faithfulness",
                "generator_faithfulness",
                "notes",
            ]
        )
        extractive_by_id = {row["question_id"]: row for row in extractive_rows}
        for generated_row in generated_rows:
            question_id = generated_row["question_id"]
            extractive_row = extractive_by_id[question_id]
            writer.writerow(
                [
                    question_id,
                    generated_row["question"],
                    generated_row["expected_answer"],
                    extractive_row["text"],
                    generated_row["generator"]["text"],
                    json.dumps(generated_row["generator"]["claims"], ensure_ascii=False),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

    (results_dir / "summary.md").write_text(
        _summary(config, extractive_summary, generator_summary, abstention, decision, comparison),
        encoding="utf-8",
    )

    latency_summary = {
        "extractive": {
            "mean_ms": round(sum(extractive_latencies) / len(extractive_latencies), 2)
            if extractive_latencies
            else 0.0,
            "median_ms": round(float(statistics.median(extractive_latencies)), 2)
            if extractive_latencies
            else 0.0,
            "max_ms": round(max(extractive_latencies), 2) if extractive_latencies else 0.0,
        },
        "generator": {
            "mean_ms": round(sum(generation_latencies) / len(generation_latencies), 2)
            if generation_latencies
            else 0.0,
            "median_ms": round(float(statistics.median(generation_latencies)), 2)
            if generation_latencies
            else 0.0,
            "max_ms": round(max(generation_latencies), 2) if generation_latencies else 0.0,
        },
    }

    _write_json(
        results_dir / "latency.json",
        {"experiment_id": EXPERIMENT_ID, **latency_summary},
    )

    return {
        "experiment_id": EXPERIMENT_ID,
        "mechanical_decision": decision,
        "extractive_summary": extractive_summary,
        "generator_summary": generator_summary,
        "abstention": abstention,
        "latency": latency_summary,
        "build_stats": build_stats,
        "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-004 constrained generator evaluation")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--questions-csv", type=Path, default=QUESTIONS_CSV)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--unanswerable-csv", type=Path, default=UNANSWERABLE_CSV)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--generator-model", default=GENERATOR_MODEL)
    args = parser.parse_args()

    result = run(
        pdf_path=args.pdf,
        questions_csv=args.questions_csv,
        ground_truth_path=args.ground_truth,
        unanswerable_csv=args.unanswerable_csv,
        results_dir=args.results_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        generator_model=args.generator_model,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
