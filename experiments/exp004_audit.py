"""Validate and summarize the manual EXP-004 audit.

The manual labels are reviewer judgements, not automated metrics.  This script
only checks that every question has both arms, that labels come from the fixed
vocabulary, and that the answers under review match the stored artifacts.  It
then writes a machine-readable summary and a short decision note.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results" / "exp004"

CORRECTNESS_LABELS = {"correct", "partial", "incorrect"}
FAITHFULNESS_LABELS = {"grounded", "partially_grounded", "ungrounded"}


def _validate_label(value: Any, allowed: set, field: str, question_id: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{question_id}: invalid {field} label {value!r}")
    return value


def _counts(
    entries: List[Dict[str, Any]],
    arm: str,
    field: str,
    labels: List[str],
) -> Dict[str, int]:
    counter = Counter(entry[arm][field] for entry in entries)
    return {label: counter.get(label, 0) for label in labels}


def run(results_dir: Path = RESULTS_DIR) -> Dict[str, Any]:
    results_dir = Path(results_dir)
    audit = json.loads((results_dir / "manual_audit.json").read_text(encoding="utf-8"))
    generated = json.loads((results_dir / "answers_generated.json").read_text(encoding="utf-8"))
    extractive = json.loads((results_dir / "answers_extractive.json").read_text(encoding="utf-8"))
    comparison = json.loads((results_dir / "comparison.json").read_text(encoding="utf-8"))

    generated_rows = {row["question_id"]: row for row in generated["answers"]}
    extractive_rows = {row["question_id"]: row for row in extractive["answers"]}
    entries = audit.get("questions", [])
    if len(entries) != 15:
        raise ValueError(f"expected 15 audited questions, found {len(entries)}")
    if {entry["question_id"] for entry in entries} != set(generated_rows):
        raise ValueError("audit question ids do not match the generated answers")
    if {entry["question_id"] for entry in entries} != set(extractive_rows):
        raise ValueError("audit question ids do not match the extractive answers")

    for entry in entries:
        question_id = entry["question_id"]
        if entry["question"] != generated_rows[question_id]["question"]:
            raise ValueError(f"{question_id}: question text differs from generated artifact")
        if entry["expected_answer"] != generated_rows[question_id]["expected_answer"]:
            raise ValueError(f"{question_id}: expected answer differs from generated artifact")
        for arm in ("baseline", "generator"):
            _validate_label(entry[arm]["correctness"], CORRECTNESS_LABELS, "correctness", question_id)
            _validate_label(entry[arm]["faithfulness"], FAITHFULNESS_LABELS, "faithfulness", question_id)
            if not isinstance(entry[arm].get("notes"), str) or not entry[arm]["notes"].strip():
                raise ValueError(f"{question_id}: {arm} notes are required")

    baseline_correctness = _counts(
        entries, "baseline", "correctness", ["correct", "partial", "incorrect"]
    )
    generator_correctness = _counts(
        entries, "generator", "correctness", ["correct", "partial", "incorrect"]
    )
    baseline_faithfulness = _counts(
        entries, "baseline", "faithfulness", ["grounded", "partially_grounded", "ungrounded"]
    )
    generator_faithfulness = _counts(
        entries, "generator", "faithfulness", ["grounded", "partially_grounded", "ungrounded"]
    )
    mechanical = comparison["mechanical_decision"]

    reasons = []
    if not mechanical["mechanical_pass"]:
        failed = [name for name, passed in mechanical["checks"].items() if not passed]
        reasons.append("pre-registered mechanical gate failed: " + ", ".join(failed))
    if generator_correctness.get("incorrect", 0) > 0:
        reasons.append(
            f"{generator_correctness['incorrect']}/15 generated answers were judged incorrect"
        )
    if generator_correctness.get("correct", 0) <= baseline_correctness.get("correct", 0):
        reasons.append("manual audit found no correctness improvement over the extractive baseline")
    if generator_faithfulness.get("ungrounded", 0) > 0:
        reasons.append("manual audit found ungrounded generated claims")
    if not reasons:
        reasons.append("mechanical gate and manual audit both passed")

    summary = {
        "experiment_id": audit.get("experiment_id", "EXP-004"),
        "reviewer": audit.get("reviewer"),
        "reviewed_at": audit.get("reviewed_at"),
        "question_count": len(entries),
        "baseline": {
            "correctness": baseline_correctness,
            "faithfulness": baseline_faithfulness,
        },
        "generator": {
            "correctness": generator_correctness,
            "faithfulness": generator_faithfulness,
        },
        "mechanical_decision": mechanical,
        "production_ready": False,
        "decision": "reject_production_default",
        "reasons": reasons,
        "note": (
            "The production default remains the accepted extractive pipeline. "
            "This experiment did not establish a correctness or faithfulness gain "
            "large enough to justify changing it."
        ),
    }

    (results_dir / "manual_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# EXP-004 decision",
        "",
        f"- Mechanical gate: **{'PASS' if mechanical['mechanical_pass'] else 'FAIL'}**",
        f"- Production default: **unchanged** (extractive pipeline)",
        "- Generator: **rejected for production** by this experiment",
        "",
        "## Manual audit counts (15 positive questions)",
        "",
        "| Correctness | Extractive | Generated |",
        "|---|---:|---:|",
    ]
    for label in ("correct", "partial", "incorrect"):
        lines.append(
            f"| {label} | {baseline_correctness.get(label, 0)} | "
            f"{generator_correctness.get(label, 0)} |"
        )
    lines += [
        "",
        "| Faithfulness | Extractive | Generated |",
        "|---|---:|---:|",
    ]
    for label in ("grounded", "partially_grounded", "ungrounded"):
        lines.append(
            f"| {label} | {baseline_faithfulness.get(label, 0)} | "
            f"{generator_faithfulness.get(label, 0)} |"
        )
    lines += [
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    lines += [
        "",
        "## Interpretation",
        "",
        "- The generator was constrained to the extractive selector's own evidence,",
        "  so it could not fix retrieval or selection failures.",
        "- The audit found no fabricated material claims, but four answers were still",
        "  incorrect because the selected evidence was wrong or incomplete.",
        "- A fluent rewrite of an incomplete heading (Q04, Q12) is not an improvement",
        "  in answer quality.",
        "- The pre-registered grounding proxy also failed (14/16 claims at >=0.80",
        "  token coverage, rate 0.875 vs required 0.90).",
        "",
    ]
    (results_dir / "decision.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    summary = run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
