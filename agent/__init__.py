"""
Package initializer for the ScienceQ agent.

`agent/` was historically a namespace package (no __init__), which let callers
reach into it two incompatible ways:

  - the API imports the agent by the *bare* name — ``from agent import
    YouTubeQAAgent`` — because ``agent/`` is placed on ``sys.path`` (see
    ``api/__init__.py``), so ``agent.py`` resolves as a top-level module;
  - the test suite imports the *submodule* — ``from agent.agent import ...`` —
    treating ``agent/`` as a package.

Both cannot bind ``sys.modules['agent']`` in the same interpreter, so a process
that mixes the two (e.g. ``pytest tests/`` collecting an API test alongside an
agent-node test) breaks on whichever style loads second.

Re-exporting the public API here makes ``agent`` a genuine package for which
``from agent import YouTubeQAAgent`` also works, so both conventions converge on
the package form. In production the bare-module resolution still wins (``agent/``
sits ahead of the repo root on ``sys.path``), so this file is inert there — it
only takes effect when ``agent`` is imported as a package.
"""

from __future__ import annotations

from .agent import (  # noqa: F401 — re-exported for `from agent import ...`
    GenerationProvenance,
    YouTubeQAAgent,
    _derive_provenance,
)

__all__ = ["YouTubeQAAgent", "GenerationProvenance", "_derive_provenance"]
