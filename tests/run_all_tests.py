# tests/run_all_tests.py

"""
run_all_tests.py
----------------
Runs the full test battery and prints a summary.
Usage:
  python tests/run_all_tests.py

Discovery over ``tests/``, never a curated module list. An enumerated list is a
second place to register a test file, and the registration silently rots: the
list this replaced ran 85 of 106 tests green because the issue #16 excerpt suite
was never added to it. Discovery is also what CI runs
(``.github/workflows/ci.yml``), so local and CI runs collect the same set by
construction rather than by maintenance. Do not reintroduce a list.

There is deliberately no passthrough of extra arguments to pytest. ``tests/``
would have to be passed first, so a path argument would *widen* the run instead
of narrowing it — ``run_all_tests.py tests/test_cleaner.py`` would quietly run
all of them. Anyone wanting to select tests should call pytest directly.

``cwd`` is pinned to the repo root because ``tests/`` above is a relative path;
imports do not need it (``conftest.py`` puts the root on ``sys.path`` off
``__file__``). So this works from any directory, not just the root.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=ROOT,
    )
    sys.exit(result.returncode)
