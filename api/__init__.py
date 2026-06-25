"""
FastAPI service for ScienceQ — a thin HTTP layer over the LangGraph agent.

The agent/ and pipeline/ packages use bare module imports internally
(e.g. ``from rag_chain import ...``), so their directories must be on
``sys.path`` before any api submodule imports the agent. Doing it here in the
package __init__ guarantees the bridge is established no matter which submodule
is imported first (the uvicorn entrypoint, tests, or a direct import).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("agent", "pipeline"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
