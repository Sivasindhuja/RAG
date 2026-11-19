"""Constrained, evidence-grounded generation over selected evidence.

This module does **not** retrieve or select evidence.  It takes verbatim
evidence units selected by :class:`~src.rag_core.pipeline.GroundedPipeline` and
asks a model to answer under a strict JSON contract:

- every generated claim must name one or more supplied evidence IDs;
- source IDs are validated by the application against the supplied evidence;
- citation markers/metadata are resolved by the application from the selected
  evidence (the model must not emit or be trusted for section/page/source refs);
- if evidence is insufficient, the result is abstention (not a guess).

Generators: :class:`OllamaGenerator` (EXP-004, local/experimental, rejected for
production) and :class:`GeminiGenerator` (M1 validated Gemini path). Faithfulness
is not guaranteed by generation; claims must be evaluated against evidence.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pipeline import AnswerStatement, Citation

ANSWERED = "answered"
MODEL_ABSTAINED = "model_abstained"
GENERATION_ERROR = "generation_error"

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
EVIDENCE_INSUFFICIENT = "evidence_insufficient"
DEFAULT_GENERATOR_MODEL = "qwen2.5:1.5b"

GEMINI_PROMPT_VERSION = "grounded_json_v1"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GEMINI_TEMPERATURE = 0.0
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 512
DEFAULT_GENERATOR_OPTIONS: Dict[str, Any] = {
    "temperature": 0,
    "top_p": 0.9,
    "num_ctx": 4096,
    "num_predict": 220,
    "repeat_penalty": 1.1,
}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT_DIR / "prompts"


def _load_prompt_template(name: str = "generation_grounded_v1.txt") -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Prompt template is empty: {path}")
    return text


class GenerationContractError(ValueError):
    """Raised when the model response does not follow the claim contract."""


@dataclass(frozen=True)
class GeneratedClaim:
    """One generated claim and the evidence sentences it names."""

    text: str
    source_ids: Tuple[int, ...]
    markers: Tuple[str, ...]
    invalid_source_ids: Tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_ids": list(self.source_ids),
            "markers": list(self.markers),
            "invalid_source_ids": list(self.invalid_source_ids),
        }


@dataclass
class GeneratedAnswer:
    """A generated answer plus the evidence/citations it used.

    ``status`` is one of ``answered``, ``model_abstained`` or
    ``generation_error``.  A ``generation_error`` answer keeps the raw model
    response and error message for audit; it is never silently replaced by an
    extractive answer in an experiment.
    """

    status: str
    question: str
    text: str
    claims: List[GeneratedClaim]
    citations: List[Citation]
    evidence_statements: List[AnswerStatement]
    raw_response: str
    model: str
    latency_ms: float
    errors: List[str]
    metadata: Dict[str, Any]

    @property
    def answered(self) -> bool:
        return self.status == ANSWERED

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "question": self.question,
            "text": self.text,
            "claims": [claim.to_dict() for claim in self.claims],
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence_statements": [statement.to_dict() for statement in self.evidence_statements],
            "raw_response": self.raw_response,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


def build_selected_evidence_prompt(question: str, statements: Sequence[AnswerStatement]) -> str:
    """Build the JSON-contract prompt for the selected evidence sentences.

    Kept for backward compatibility (EXP-004). New Gemini path uses
    :func:`build_gemini_evidence_prompt`.
    """
    if not statements:
        raise ValueError("at least one selected evidence statement is required")
    evidence = "\n".join(f"[{index}] {statement.text}" for index, statement in enumerate(statements, 1))
    maximum = len(statements)
    return f"""You are a careful policy analyst. Write a concise answer using ONLY the evidence sentences below.

Return valid JSON only, with this shape:
{{"sentences": [{{"text": "...", "sources": [1]}}]}}

Rules:
- Answer the exact question asked. Ignore evidence that does not answer it.
- Each sentence must state one factual claim from the evidence and list every evidence number that supports it.
- Use evidence numbers from 1 to {maximum} only. Do not invent numbers.
- Do not include page numbers, section numbers, evidence labels, background, or facts not in the evidence.
- Do not repeat a sentence or claim. Use at most 3 sentences.
- If the evidence does not answer the question, return {{"sentences": []}}.
- Do not change the policy meaning.

Evidence:
{evidence}

Question: {question}
"""


def build_gemini_evidence_prompt(
    question: str,
    statements: Sequence[AnswerStatement],
    prompt_version: str = GEMINI_PROMPT_VERSION,
    template_name: str = "generation_grounded_v1.txt",
) -> str:
    """Build the versioned Gemini prompt from configurable prompt templates.

    Evidence is identified only by opaque numeric IDs [1..N]. The application
    validates cited IDs and resolves them to actual source metadata; the model
    must not emit or rely on page/section/source references.
    """
    if not statements:
        raise ValueError("at least one selected evidence statement is required")
    evidence = "\n".join(f"[{index}] {statement.text}" for index, statement in enumerate(statements, 1))
    maximum = len(statements)
    template = _load_prompt_template(template_name)
    rendered = (
        template.replace("{{prompt_version}}", str(prompt_version))
        .replace("{{max_source_id}}", str(maximum))
        .replace("{{evidence}}", evidence)
        .replace("{{question}}", question)
    )
    # Ensure placeholders resolved (ignore JSON braces).
    import re as _re
    unresolved = _re.findall(r"\{\{[^}]+\}\}", rendered)
    if unresolved:
        raise ValueError(f"Unresolved prompt placeholders: {sorted(set(unresolved))}")
    return rendered


def _strip_fence(raw: str) -> str:
    return _FENCE_RE.sub("", raw.strip())


def _loads_with_duplicate_detection(raw: str) -> Tuple[Any, List[str]]:
    duplicates: List[str] = []

    def hook(pairs: Sequence[Tuple[str, Any]]) -> dict:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    return json.loads(_strip_fence(raw), object_pairs_hook=hook), duplicates


def parse_generated_claims(
    raw_response: str,
    statement_count: int,
) -> Tuple[List[GeneratedClaim], List[str]]:
    """Parse the JSON contract and validate source IDs.

    Invalid source IDs are preserved on the claim so an experiment can count
    them; they are never mapped to a citation.  Structural failures raise
    :class:`GenerationContractError`.
    """
    if statement_count <= 0:
        raise ValueError("statement_count must be positive")
    try:
        payload, duplicates = _loads_with_duplicate_detection(raw_response)
    except json.JSONDecodeError as exc:
        raise GenerationContractError(f"invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise GenerationContractError("top-level JSON value must be an object")
    sentences = payload.get("sentences")
    if not isinstance(sentences, list):
        raise GenerationContractError("missing list field 'sentences'")

    errors: List[str] = []
    if duplicates:
        errors.append("duplicate JSON keys: " + ", ".join(sorted(set(duplicates))))

    claims: List[GeneratedClaim] = []
    for index, item in enumerate(sentences):
        if not isinstance(item, dict):
            raise GenerationContractError(f"sentences[{index}] must be an object")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise GenerationContractError(f"sentences[{index}].text must be a non-empty string")
        sources = item.get("sources")
        if not isinstance(sources, list):
            raise GenerationContractError(f"sentences[{index}].sources must be a list")

        source_ids: List[int] = []
        invalid: List[int] = []
        for value in sources:
            if isinstance(value, bool):
                invalid.append(-1)
                continue
            if isinstance(value, int):
                source_id = value
            elif isinstance(value, str) and value.strip().isdigit():
                source_id = int(value.strip())
            else:
                invalid.append(-1)
                continue
            if 1 <= source_id <= statement_count:
                if source_id not in source_ids:
                    source_ids.append(source_id)
            elif source_id not in invalid:
                invalid.append(source_id)

        claims.append(
            GeneratedClaim(
                text=" ".join(text.split()),
                source_ids=tuple(source_ids),
                markers=(),
                invalid_source_ids=tuple(invalid),
            )
        )
    return claims, errors


def _claim_markers(claim: GeneratedClaim, statements: Sequence[AnswerStatement]) -> Tuple[str, ...]:
    markers: List[str] = []
    for source_id in claim.source_ids:
        marker = statements[source_id - 1].marker
        if marker not in markers:
            markers.append(marker)
    return tuple(markers)


def assemble_generated_answer(
    question: str,
    statements: Sequence[AnswerStatement],
    citations: Sequence[Citation],
    raw_response: str,
    *,
    model: str,
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> GeneratedAnswer:
    """Turn a model JSON response into a cited answer or a recorded failure."""
    if not statements:
        raise ValueError("at least one selected evidence statement is required")
    try:
        parsed, errors = parse_generated_claims(raw_response, len(statements))
    except GenerationContractError as exc:
        return GeneratedAnswer(
            status=GENERATION_ERROR,
            question=question,
            text="",
            claims=[],
            citations=[],
            evidence_statements=list(statements),
            raw_response=raw_response,
            model=model,
            latency_ms=latency_ms,
            errors=[str(exc)],
            metadata=dict(metadata or {}),
        )

    if not parsed:
        return GeneratedAnswer(
            status=MODEL_ABSTAINED,
            question=question,
            text=INSUFFICIENT_EVIDENCE,
            claims=[],
            citations=[],
            evidence_statements=list(statements),
            raw_response=raw_response,
            model=model,
            latency_ms=latency_ms,
            errors=errors,
            metadata=dict(metadata or {}),
        )

    claims: List[GeneratedClaim] = []
    pieces: List[str] = []
    used_markers: List[str] = []
    for claim in parsed:
        markers = _claim_markers(claim, statements)
        claim = GeneratedClaim(
            text=claim.text,
            source_ids=claim.source_ids,
            markers=markers,
            invalid_source_ids=claim.invalid_source_ids,
        )
        claims.append(claim)
        if claim.invalid_source_ids:
            errors.append(f"claim cites invalid evidence numbers: {list(claim.invalid_source_ids)}")
        if not markers:
            errors.append("claim has no valid evidence source")
        for marker in markers:
            if marker not in used_markers:
                used_markers.append(marker)
        suffix = f" {' '.join(markers)}" if markers else ""
        pieces.append(f"{claim.text}{suffix}")

    used_citations = [citation for citation in citations if citation.marker in used_markers]
    return GeneratedAnswer(
        status=ANSWERED,
        question=question,
        text=" ".join(pieces),
        claims=claims,
        citations=used_citations,
        evidence_statements=list(statements),
        raw_response=raw_response,
        model=model,
        latency_ms=latency_ms,
        errors=errors,
        metadata=dict(metadata or {}),
    )


class OllamaGenerator:
    """Small local-model client for the selected-evidence contract (EXP-004).

    Importing this module does not require Ollama.  The dependency is imported
    only when a generation is actually requested.
    """

    def __init__(
        self,
        model: str = DEFAULT_GENERATOR_MODEL,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.options = dict(DEFAULT_GENERATOR_OPTIONS)
        if options:
            self.options.update(options)

    def generate(
        self,
        question: str,
        statements: Sequence[AnswerStatement],
        citations: Sequence[Citation],
    ) -> GeneratedAnswer:
        prompt = build_selected_evidence_prompt(question, statements)
        started = time.perf_counter()
        try:
            import ollama

            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                format="json",
                options=self.options,
            )
        except Exception as exc:  # noqa: BLE001 - recorded as an experiment failure
            return GeneratedAnswer(
                status=GENERATION_ERROR,
                question=question,
                text="",
                claims=[],
                citations=[],
                evidence_statements=list(statements),
                raw_response="",
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                errors=[f"{type(exc).__name__}: {exc}"],
                metadata={},
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        metadata = {
            "eval_count": response.get("eval_count"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "done_reason": response.get("done_reason"),
            "total_duration_ns": response.get("total_duration"),
        }
        return assemble_generated_answer(
            question,
            statements,
            citations,
            response.get("response", ""),
            model=self.model,
            latency_ms=latency_ms,
            metadata=metadata,
        )


class GeminiGenerator:
    """Gemini client for the supplied-evidence-only JSON claim contract (M1).

    The application, not the model, validates source IDs and resolves citations
    to actual source metadata from selected evidence. The model must not emit
    page/section/source references; any such text is ignored by parsing.

    Requires `GOOGLE_API_KEY` or `GEMINI_API_KEY` in the environment unless an
    already-configured `google.generativeai` client is usable.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = DEFAULT_GEMINI_TEMPERATURE,
        max_output_tokens: int = DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
        prompt_version: str = GEMINI_PROMPT_VERSION,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.prompt_version = prompt_version
        self.api_key = api_key

    def _configure(self) -> None:
        import os

        import google.generativeai as genai

        key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if key:
            genai.configure(api_key=key)

    def generate(
        self,
        question: str,
        statements: Sequence[AnswerStatement],
        citations: Sequence[Citation],
        *,
        evidence_sufficient: bool = True,
        insufficiency_reason: Optional[str] = None,
    ) -> GeneratedAnswer:
        if not statements:
            return GeneratedAnswer(
                status=MODEL_ABSTAINED,
                question=question,
                text=INSUFFICIENT_EVIDENCE,
                claims=[],
                citations=[],
                evidence_statements=[],
                raw_response="",
                model=self.model,
                latency_ms=0.0,
                errors=[],
                metadata={"reason": insufficiency_reason or "no_selected_evidence", "prompt_version": self.prompt_version},
            )
        if not evidence_sufficient:
            return GeneratedAnswer(
                status=MODEL_ABSTAINED,
                question=question,
                text=INSUFFICIENT_EVIDENCE,
                claims=[],
                citations=[],
                evidence_statements=list(statements),
                raw_response="",
                model=self.model,
                latency_ms=0.0,
                errors=[],
                metadata={"reason": insufficiency_reason or EVIDENCE_INSUFFICIENT, "prompt_version": self.prompt_version},
            )

        prompt = build_gemini_evidence_prompt(question, statements, prompt_version=self.prompt_version)
        started = time.perf_counter()
        try:
            self._configure()
            import google.generativeai as genai

            api_model = self.model if self.model.startswith("models/") else f"models/{self.model}"
            model = genai.GenerativeModel(
                api_model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "response_mime_type": "application/json",
                },
            )
            response = model.generate_content(prompt)
            raw = (getattr(response, "text", None) or "").strip()
            latency_ms = (time.perf_counter() - started) * 1000.0
            metadata = {
                "prompt_version": self.prompt_version,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                metadata["prompt_token_count"] = getattr(usage, "prompt_token_count", None)
                metadata["candidates_token_count"] = getattr(usage, "candidates_token_count", None)
                metadata["total_token_count"] = getattr(usage, "total_token_count", None)
            return assemble_generated_answer(
                question,
                statements,
                citations,
                raw,
                model=self.model,
                latency_ms=latency_ms,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - recorded as an experiment failure
            return GeneratedAnswer(
                status=GENERATION_ERROR,
                question=question,
                text="",
                claims=[],
                citations=[],
                evidence_statements=list(statements),
                raw_response="",
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                errors=[f"{type(exc).__name__}: {exc}"],
                metadata={"prompt_version": self.prompt_version},
            )
