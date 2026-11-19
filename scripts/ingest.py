"""Reproducible ingestion/index build command.

Usage:
  python scripts/ingest.py
  python scripts/ingest.py --chunk-size 800 --chunk-overlap 160 --collection satcom_ngp_production_fixed_char_800_160
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from src.rag_app.config import AppConfig
from src.rag_core import (
    DenseIndex,
    DenseIndexConfig,
    FixedCharacterChunker,
    StructureAwareChunker,
    audit_source_coverage,
    extract_source_document,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--persist-dir", type=Path)
    ap.add_argument("--collection", type=str)
    ap.add_argument("--chunk-strategy", choices=["fixed_character", "structure_aware"])
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--chunk-overlap", type=int)
    args = ap.parse_args()

    cfg = AppConfig.from_env()
    pdf = args.pdf or cfg.pdf_path
    persist_dir = args.persist_dir or cfg.persist_dir
    collection = args.collection or cfg.collection_name
    strategy = args.chunk_strategy or cfg.chunk_strategy
    chunk_size = args.chunk_size or cfg.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else cfg.chunk_overlap

    doc = extract_source_document(pdf)
    if strategy == "structure_aware":
        chunker = StructureAwareChunker(max_chars=chunk_size)
        chunks = chunker.chunk(doc)
        audit = audit_source_coverage(doc, chunks, max_chars=chunk_size, allow_overlap=False)
    else:
        chunker = FixedCharacterChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        chunks = chunker.chunk(doc)
        audit = audit_source_coverage(doc, chunks, max_chars=chunk_size, allow_overlap=True)

    if not audit.get("ok"):
        print(json.dumps({"status": "error", "issues": audit.get("issues"), "audit": audit}, indent=2))
        raise SystemExit(2)

    index = DenseIndex(
        DenseIndexConfig(collection_name=collection, persist_dir=persist_dir, embedding_model=cfg.embedding_model)
    )
    stats = index.build(chunks, reset=True)
    out = {
        "status": "ok",
        "pdf": str(pdf),
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap if strategy == "fixed_character" else 0,
        "chunk_count": len(chunks),
        "collection": collection,
        "persist_dir": str(persist_dir),
        "embedding_model": cfg.embedding_model,
        "audit": {
            "ok": audit.get("ok"),
            "coverage_percent": audit.get("coverage_percent"),
            "overlap_chars": audit.get("overlap_chars"),
            "uncovered_non_whitespace_chars": audit.get("uncovered_non_whitespace_chars"),
            "total_chars": audit.get("total_chars"),
        },
        "index": stats,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
