"""Vercel serverless entrypoint for SATCOM NGP RAG API.

NOTE: Full production deployment on Vercel Hobby is constrained by serverless
bundle size/cold start limits and lack of persistent filesystem for local
Chroma/model artifacts. This entrypoint enables deployment attempts and local
Vercel dev compatibility; see docs/DEPLOYMENT.md for architecture, trade-offs,
and required external persistence/configuration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default to /tmp for any writable runtime artifacts in serverless environments.
os.environ.setdefault("RAG_PERSIST_DIR", os.getenv("RAG_PERSIST_DIR", str(ROOT / "chroma_db_production")))

from backend.main import app  # noqa: E402

# Vercel Python runtime looks for `app`/`handler` in many setups.
handler = app
