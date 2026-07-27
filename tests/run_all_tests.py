# tests/run_all_tests.py

"""
run_all_tests.py
----------------
Runs the full test battery and prints a summary.
Usage:
  python tests/run_all_tests.py            # whole suite
  python tests/run_all_tests.py -k excerpt # extra args go straight to pytest

Discovery over ``tests/``, never a curated module list. An enumerated list is a
second place to register a test file, and the registration silently rots: the
list this replaced ran 85 of 106 tests green because the issue #16 excerpt suite
was never added to it. Discovery is also what CI runs
(``.github/workflows/ci.yml``), so local and CI runs collect the same set by
construction rather than by maintenance. Do not reintroduce a list.

``cwd`` is pinned to the repo root because the suite's imports resolve the
``agent/`` and ``pipeline/`` packages relative to it — so this works from any
directory, not just the root.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", *sys.argv[1:]],
        cwd=ROOT,
    )
    sys.exit(result.returncode)
