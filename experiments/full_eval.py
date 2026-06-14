"""
Full evaluation: retrieval ablations + end-to-end Ragas + latency/cost aggregates.

Outputs: experiments/results/eval_report.json

Run:
  python -m experiments.full_eval --sample 12
  python -m experiments.full_eval --ragas --sample 5   # needs GEMINI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

from src.config import SETTINGS
from src.generation.models import CostTracker, expand_query
from src.metrics.retrieval_eval import evaluate_retrieval, keyword_recall_in_context
from src.observability.tracer import get_tracer
from src.retrieval.hierarchical import get_retriever
from src.retrieval.hybrid import HybridRetriever

PDF_PATH = ROOT / "data" / "satcom-ngp.pdf"
GOLDEN_PATH = ROOT / "tests" / "golden_dataset.csv"
RESULTS_DIR = ROOT / "experiments" / "results"


def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    return [
        Document(page_content=(p.extract_text() or "").strip(), metadata={"source": str(path), "page": i + 1})
        for i, p in enumerate(reader.pages)
        if (p.extract_text() or "").strip()
    ]


def index_paths(tag: str) -> tuple[Path, Path, Path]:
    base = ROOT / "indices" / tag
    return base / "chroma", base / "parent_docstore.pkl", base / "child_chunks.pkl"


def build_stack(use_structure: bool, force_rebuild: bool) -> HybridRetriever:
    tag = "structure" if use_structure else "recursive"
    persist, docstore, chunks = index_paths(tag)
    if force_rebuild and persist.parent.exists():
        shutil.rmtree(persist.parent)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    hier = get_retriever(
        emb, load_pdf(PDF_PATH), persist, docstore, chunks,
        force_rebuild=force_rebuild, use_structure_chunking=use_structure,
    )
    return HybridRetriever(hier, hier.all_children)


def _subset(df: pd.DataFrame, sample: int, category: str | None) -> pd.DataFrame:
    if category:
        sub = df[df["category"] == category]
    else:
        sub = df
    return sub.head(sample) if sample else sub


def retrieval_ablations(df: pd.DataFrame, force_rebuild: bool) -> dict:
    questions = df["question"].tolist()
    truths = df["answer"].tolist()
    out: dict = {}

    for use_structure, label in [(False, "recursive"), (True, "structure")]:
        hybrid = build_stack(use_structure, force_rebuild)
        dense = evaluate_retrieval(questions, truths, lambda q, h=hybrid: h.retrieve(q, use_hybrid=False))
        hybrid_m = evaluate_retrieval(questions, truths, lambda q, h=hybrid: h.retrieve(q, use_hybrid=True))
        out[f"chunking_{label}"] = {"dense": dense, "hybrid": hybrid_m}

    hybrid = build_stack(True, False)
    base = evaluate_retrieval(questions, truths, lambda q: hybrid.retrieve(q, use_hybrid=False))
    hybrid_only = evaluate_retrieval(questions, truths, lambda q: hybrid.retrieve(q, use_hybrid=True))

    def expand_retrieve(q):
        expanded = expand_query(q).text
        return hybrid.retrieve(expanded, use_hybrid=True)

    expanded = evaluate_retrieval(questions, truths, expand_retrieve)
    out["hybrid_ablation"] = {
        "dense_only": base,
        "hybrid": hybrid_only,
        "hybrid_plus_expansion": expanded,
        "delta_hybrid_vs_dense": round(hybrid_only["mean_keyword_recall"] - base["mean_keyword_recall"], 4),
        "delta_expansion_vs_hybrid": round(expanded["mean_keyword_recall"] - hybrid_only["mean_keyword_recall"], 4),
    }
    return out


def retrieval_by_category(df: pd.DataFrame, force_rebuild: bool) -> dict:
    hybrid = build_stack(True, force_rebuild)
    results = {}
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        q, t = sub["question"].tolist(), sub["answer"].tolist()
        dense = evaluate_retrieval(q, t, lambda qy, h=hybrid: h.retrieve(qy, use_hybrid=False))
        hyb = evaluate_retrieval(q, t, lambda qy, h=hybrid: h.retrieve(qy, use_hybrid=True))

        def vague_expand(qy):
            return hybrid.retrieve(expand_query(qy).text, use_hybrid=True)

        exp = evaluate_retrieval(q, t, vague_expand) if cat == "vague" else {}
        results[cat] = {"dense": dense, "hybrid": hyb}
        if exp:
            results[cat]["hybrid_expansion"] = exp
            results[cat]["expansion_gain"] = round(
                exp["mean_keyword_recall"] - hyb["mean_keyword_recall"], 4
            )
    return results


def pipeline_ablation(df: pd.DataFrame) -> dict:
    """End-to-end: context recall proxy + citation + cost (no Ragas)."""
    from src.rag import ask_question, ensure_index

    ensure_index()
    configs = [
        ("dense_no_expand", False, False),
        ("hybrid_no_expand", True, False),
        ("hybrid_expand", True, True),
    ]
    results = {}
    latencies = []

    for name, use_hybrid, use_expansion in configs:
        recalls, citations, costs = [], [], []
        for _, row in df.iterrows():
            t0 = time.perf_counter()
            try:
                ans, docs, m = ask_question(
                    row["question"],
                    use_hybrid=use_hybrid,
                    use_expansion=use_expansion,
                    use_rerank=SETTINGS["features"]["reranking"],
                )
                latencies.append((time.perf_counter() - t0) * 1000)
                recalls.append(keyword_recall_in_context(docs, row["answer"]))
                citations.append(m.get("citation_coverage", 0))
                costs.append(m.get("cost_usd", 0))
            except Exception as exc:
                print(f"  [{name}] skip: {exc}")
        n = max(len(recalls), 1)
        results[name] = {
            "mean_context_keyword_recall": round(sum(recalls) / n, 4),
            "mean_citation_coverage": round(sum(citations) / n, 4),
            "mean_cost_usd": round(sum(costs) / n, 6),
            "n": len(recalls),
        }

    tracer = get_tracer()
    results["_latency_percentiles_ms"] = tracer.latency.percentiles()
    if latencies:
        s = sorted(latencies)
        n = len(s)
        results["_pipeline_latency_ms"] = {
            "p50": round(s[int(0.5 * (n - 1))], 2),
            "p95": round(s[int(0.95 * (n - 1))], 2),
        }
    return results


def run_ragas(df: pd.DataFrame) -> dict:
    from src.rag import ask_question, ensure_index

    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, AnswerCorrectness, LLMContextRecall
    from ragas.run_config import RunConfig
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    ensure_index()
    rows = []
    for _, row in df.iterrows():
        ans, docs, m = ask_question(row["question"], use_hybrid=True, use_expansion=True)
        rows.append({
            "question": row["question"],
            "answer": ans,
            "contexts": [d.page_content for d in docs],
            "ground_truth": row["answer"],
            "citation_coverage": m.get("citation_coverage", 0),
        })
        time.sleep(2)

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="models/gemini-2.0-flash"))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    ev = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerCorrectness(), LLMContextRecall()],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=1, timeout=180),
    )
    def avg(key):
        vals = ev[key]
        return round(sum(vals) / len(vals), 4) if vals else 0

    return {
        "faithfulness": avg("faithfulness"),
        "answer_correctness": avg("answer_correctness"),
        "context_recall": avg("context_recall"),
        "mean_citation_coverage": round(sum(r["citation_coverage"] for r in rows) / len(rows), 4),
        "n": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--pipeline", action="store_true", help="Run end-to-end pipeline ablation (uses API)")
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()

    if not PDF_PATH.exists():
        raise SystemExit(f"Missing PDF: {PDF_PATH}")

    df = pd.read_csv(GOLDEN_PATH)
    if "category" not in df.columns:
        df["category"] = "factual"
    df = df.head(args.sample)

    report = {"sample_n": len(df), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if not args.skip_retrieval:
        print("Running retrieval ablations...")
        report["retrieval"] = retrieval_ablations(df, args.rebuild_index)
        report["retrieval_by_category"] = retrieval_by_category(df, args.rebuild_index)

    if args.pipeline:
        print("Running pipeline ablation (API calls)...")
        sub = df.head(min(8, len(df)))
        report["pipeline"] = pipeline_ablation(sub)

    if args.ragas:
        if not os.getenv("GEMINI_API_KEY"):
            print("Skipping Ragas: GEMINI_API_KEY not set")
        else:
            print("Running Ragas (slow)...")
            report["ragas"] = run_ragas(df.head(min(5, len(df))))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
