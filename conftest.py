"""Ensure the repository root is importable so tests can import `app` and `scripts`."""

import os
import sys
from pathlib import Path

# CI / local tests must never pull the 1.7B SAGE or 16 GB T-lite checkpoints.
os.environ.setdefault("CORRECTOR_MODEL", "ai-forever/sage-fredt5-distilled-95m")
os.environ.setdefault("SAGE_MODEL", "ai-forever/sage-fredt5-distilled-95m")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
