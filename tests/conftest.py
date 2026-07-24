"""
Test-suite import bootstrap.

Pins ``agent`` to its *package* form before any test module imports it, so the
whole session agrees on one binding of ``sys.modules['agent']``. Without this,
collection order decides whether ``agent`` resolves to the package (tests do
``from agent.agent import ...``) or the bare ``agent.py`` module (``api.service``
does ``from agent import ...``) — and the two are mutually exclusive within one
interpreter.

Putting the repo root first on ``sys.path`` and importing the package here —
before ``api/__init__`` or ``agent.py`` can prepend ``agent/`` and flip
resolution to the bare module — locks in the package form for the run. The
package's ``__init__`` re-exports the public API, so ``from agent import
YouTubeQAAgent`` (used by ``api.service``) keeps working against it.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import agent  # noqa: E402,F401 — pin sys.modules['agent'] to the package
