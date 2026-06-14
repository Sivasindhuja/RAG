"""LLM calls with optional per-task routing, cost tracking, and token usage."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai

from src.config import SETTINGS, PROMPTS


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Session-level token/cost estimates."""

    calls: list[dict] = field(default_factory=list)

    def record(self, task: str, result: LLMResult):
        self.calls.append(
            {
                "task": task,
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
            }
        )

    def summary(self) -> dict:
        if not self.calls:
            return {"total_cost_usd": 0, "total_input_tokens": 0, "total_output_tokens": 0, "by_task": {}}
        by_task: dict[str, dict] = {}
        for c in self.calls:
            t = c["task"]
            by_task.setdefault(t, {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0})
            by_task[t]["calls"] += 1
            by_task[t]["cost_usd"] += c["cost_usd"]
            by_task[t]["input_tokens"] += c["input_tokens"]
            by_task[t]["output_tokens"] += c["output_tokens"]
        return {
            "total_cost_usd": round(sum(c["cost_usd"] for c in self.calls), 6),
            "total_input_tokens": sum(c["input_tokens"] for c in self.calls),
            "total_output_tokens": sum(c["output_tokens"] for c in self.calls),
            "by_task": by_task,
        }


def _estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+", text)))


def _usage_from_response(response: Any, prompt: str, text: str) -> tuple[int, int]:
    meta = getattr(response, "usage_metadata", None)
    if meta:
        return (
            getattr(meta, "prompt_token_count", None) or _estimate_tokens(prompt),
            getattr(meta, "candidates_token_count", None) or _estimate_tokens(text),
        )
    return _estimate_tokens(prompt), _estimate_tokens(text)


def _cost_for_model(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = SETTINGS.get("pricing_usd_per_1m", {}).get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def _resolve_model(task: str) -> str:
    models = SETTINGS["models"]
    if SETTINGS["features"].get("model_routing"):
        return models.get(task, models["single"])
    return models["single"]


def generate(task: str, prompt: str, tracker: CostTracker | None = None) -> LLMResult:
    model_name = _resolve_model(task)
    api_model = f"models/{model_name}" if not model_name.startswith("models/") else model_name
    model = genai.GenerativeModel(api_model)
    response = model.generate_content(prompt)
    text = (response.text or "").strip()
    in_tok, out_tok = _usage_from_response(response, prompt, text)
    cost = _cost_for_model(model_name.replace("models/", ""), in_tok, out_tok)
    result = LLMResult(text=text, model=model_name, input_tokens=in_tok, output_tokens=out_tok, cost_usd=round(cost, 6))
    if tracker is not None:
        tracker.record(task, result)
    return result


def _heuristic_expand(question: str) -> str:
    q = question.strip()
    acronyms = ["GMPCS", "INSAT", "CAISS", "WPC", "DoT", "ITU", "ICC", "TAG", "FDI"]
    extra = [a for a in acronyms if a.lower() in q.lower() or a in q]
    if "license" in q.lower() or "licence" in q.lower():
        extra.append("operating license authorization")
    if "satellite" in q.lower() and "indian" in q.lower():
        extra.append("Indian satellite system authorization")
    if extra:
        return f"{q} {' '.join(dict.fromkeys(extra))}"
    return q


def expand_query(question: str, memory_context: str = "", tracker: CostTracker | None = None) -> LLMResult:
    prompt = PROMPTS["query_expansion"].format(question=question, memory_context=memory_context or "None")
    try:
        return generate("query_expansion", prompt, tracker)
    except Exception:
        text = _heuristic_expand(question)
        return LLMResult(text=text, model="heuristic", input_tokens=0, output_tokens=0, cost_usd=0.0)


def answer_question(question: str, context: str, memory_context: str = "", tracker: CostTracker | None = None) -> LLMResult:
    prompt = PROMPTS["rag_answer"].format(
        context=context,
        question=question,
        memory_context=memory_context or "None",
    )
    return generate("generation", prompt, tracker)
