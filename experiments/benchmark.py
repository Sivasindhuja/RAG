"""
Ablation benchmarks for portfolio metrics.

Measures retrieval before/after:
  - chunking: recursive vs structure-aware
  - query expansion: off vs on
  - hybrid retrieval: dense-only vs hybrid

Run: python -m experiments.benchmark --rebuild-index
"""

from __future__ import annotations

import argparse
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

from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain_core.documents import Document

from src.config import SETTINGS
from src.generation.models import CostTracker, expand_query
from src.metrics.retrieval_eval import evaluate_retrieval
from src.retrieval.hierarchical import get_retriever
from src.retrieval.hybrid import HybridRetriever

DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "satcom-ngp.pdf"
GOLDEN_PATH = ROOT / "tests" / "golden_dataset.csv"


def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": str(path), "page": i + 1}))
    return docs


def index_paths(tag: str) -> tuple[Path, Path, Path]:
    base = ROOT / "indices" / tag
    return base / "chroma", base / "parent_docstore.pkl", base / "child_chunks.pkl"


def build_stack(use_structure: bool, force_rebuild: bool):
    tag = "structure" if use_structure else "recursive"
    persist, docstore, chunks = index_paths(tag)
    if force_rebuild and persist.parent.exists():
        shutil.rmtree(persist.parent)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    hier = get_retriever(
        embeddings,
        load_pdf(PDF_PATH),
        persist,
        docstore,
        chunks,
        force_rebuild=force_rebuild,
        use_structure_chunking=use_structure,
    )
    return HybridRetriever(hier, hier.all_children)


def run_ablations(sample_n: int = 15, force_rebuild: bool = False):
    df = pd.read_csv(GOLDEN_PATH).head(sample_n)
    questions = df["question"].tolist()
    truths = df["answer"].tolist()

    print("\n=== 1. CHUNKING: recursive vs structure-aware (dense retrieval) ===\n")
    results_chunk = {}
    for use_structure, label in [(False, "recursive"), (True, "structure")]:
        hybrid = build_stack(use_structure, force_rebuild)
        t0 = time.perf_counter()

        def retrieve_parents(q, h=hybrid):
            return h.retrieve(q, use_hybrid=False)

        def retrieve_children(q, h=hybrid):
            return h.hier.get_child_matches(q)

        metrics = evaluate_retrieval(questions, truths, retrieve_parents)
        metrics_children = evaluate_retrieval(questions, truths, retrieve_children)
        metrics["latency_s"] = round(time.perf_counter() - t0, 2)
        results_chunk[label] = metrics
        print(f"  {label} (parents):  {metrics}")
        print(f"  {label} (children): {metrics_children}")

    print("\n=== 2. HYBRID RETRIEVAL (structure index) ===\n")
    hybrid = build_stack(True, force_rebuild=False)
    dense = evaluate_retrieval(
        questions,
        truths,
        lambda q: hybrid.retrieve(q, use_hybrid=False),
    )
    hybrid_m = evaluate_retrieval(
        questions,
        truths,
        lambda q: hybrid.retrieve(q, use_hybrid=True),
    )
    print(f"  dense only: {dense}")
    print(f"  hybrid:     {hybrid_m}")

    print("\n=== 3. QUERY EXPANSION (structure index, dense-only) ===\n")
    from src.generation.models import _heuristic_expand

    tracker = CostTracker()

    def no_expand(q):
        return hybrid.retrieve(q, use_hybrid=False)

    def with_heuristic_expand(q):
        return hybrid.retrieve(_heuristic_expand(q), use_hybrid=False)

    def with_llm_expand(q):
        return hybrid.retrieve(expand_query(q, tracker=tracker).text, use_hybrid=False)

    base = evaluate_retrieval(questions, truths, no_expand)
    heuristic = evaluate_retrieval(questions, truths, with_heuristic_expand)
    print(f"  without expansion:     {base}")
    print(f"  heuristic expansion: {heuristic}")
    try:
        llm_exp = evaluate_retrieval(questions, truths, with_llm_expand)
        print(f"  LLM expansion:         {llm_exp}")
        print(f"  expansion cost (est.): {tracker.summary()}")
        expansion_delta = llm_exp["mean_keyword_recall"] - base["mean_keyword_recall"]
    except Exception as exc:
        print(f"  LLM expansion skipped: {exc}")
        expansion_delta = heuristic["mean_keyword_recall"] - base["mean_keyword_recall"]

    print("\n=== SUMMARY (deltas) ===\n")
    rec_struct = results_chunk["structure"]["mean_keyword_recall"]
    rec_rec = results_chunk["recursive"]["mean_keyword_recall"]
    print(f"  structure vs recursive keyword recall: {rec_struct - rec_rec:+.4f}")
    print(f"  query expansion keyword recall: {expansion_delta:+.4f}")
    print(
        f"  hybrid vs dense keyword recall: "
        f"{hybrid_m['mean_keyword_recall'] - dense['mean_keyword_recall']:+.4f}"
    )


def run_model_cost_demo():
    """Compare single-model vs routed models on a few expansion+answer calls."""
    from src.generation.models import answer_question

    SETTINGS["features"]["model_routing"] = False
    t1 = CostTracker()
    q = "Who chairs CAISS?"
    expand_query(q, tracker=t1)
    answer_question(q, "CAISS is chaired by the Secretary, Department of Space.", tracker=t1)
    single = t1.summary()

    SETTINGS["features"]["model_routing"] = True
    t2 = CostTracker()
    expand_query(q, tracker=t2)
    answer_question(q, "CAISS is chaired by the Secretary, Department of Space.", tracker=t2)
    routed = t2.summary()

    print("\n=== 4. MODEL ROUTING COST (sample, estimated) ===\n")
    print(f"  single model:  {single}")
    print(f"  routed models: {routed}")
    SETTINGS["features"]["model_routing"] = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--sample", type=int, default=15)
    parser.add_argument("--skip-llm-cost", action="store_true")
    args = parser.parse_args()
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF missing: {PDF_PATH}")
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set; query expansion / cost demo will fail.")
    run_ablations(sample_n=args.sample, force_rebuild=args.rebuild_index)
    if not args.skip_llm_cost and os.getenv("GEMINI_API_KEY"):
        run_model_cost_demo()
