"""Convenience evaluation command for the fixed 15-question CI dataset."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "evaluation.ci_eval", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
