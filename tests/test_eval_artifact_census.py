# tests/test_eval_artifact_census.py

"""
``COMPLIANCE.md``'s tripwire 2 census must match what is actually tracked (#124).

The 2026-07-31 recorded assessment concludes that Art. 50(4) does not attach to
the generated answers committed under ``eval/results/``. That conclusion rests
on a purpose test, not on volume — but the entry states its scope as a count:
**562 answers across eight files**, split 112 across five ``run_*.json``
checkpoints and 450 across three sweep files. An assessment that names its
scope is only auditable while the named scope is the real one.

Nothing held the two together, and the gap is not theoretical. Seven untracked
``run_*.json`` files sit in the working tree of the machine that produced them.
#117's ``.gitignore`` rule ends ``eval/results/*`` with ``!eval/results/run_*.json``
— deliberately, so that every *future* checkpoint is tracked — which means those
seven are un-ignored and visible to ``git add -A``. One such command commits all
seven, takes the census from eight files to fifteen, and adds well over a
hundred answers to a scope figure that still reads 562. The assessment would be
stale the moment it was contradicted, and green everywhere.

``docs/EVALUATION.md`` also records a *decision* about those same seven files:
committing them was considered while writing that section and declined, on the
grounds that they add answers and scores but not the configuration that would
make a row re-runnable. A stray ``git add -A`` reverses a documented decision
silently. This file is what makes it cost a failing test instead.


What a failure here means
-------------------------

Not "you have done something wrong". Committing a new eval artifact is a normal
thing to do — every run from ``run_20260730_191842.json`` onward is tracked by
design. It means the census in ``docs/COMPLIANCE.md`` no longer describes the
tracked set, so the recorded assessment needs re-reading and its figures
updating before the commit lands. That is the point: the scope of a compliance
conclusion should move deliberately, in a diff someone reviewed, rather than as
a side effect of a wildcard add.

The one failure that is a genuine finding is a *new shape* — a tracked file
under ``eval/results/`` that is neither a ``run_*.json`` checkpoint nor one of
the three closed-set sweep files. The census reasons about exactly two shapes
and concludes that both qualify "in exactly the same sense". A third shape has
not been assessed, so ``test_no_unassessed_file_shapes`` fails rather than
silently folding it into a total.


Counting answers without knowing the schema
-------------------------------------------

The four shapes in this directory disagree. ``run_*.json`` holds a flat
``results`` list; ``reranker_sweep_*.json`` holds ``runs``; the two
``sweep_retrieval_stage*.json`` files hold ``combos`` keyed by combination
label, each with its own nested results. Writing four readers would mean a
fifth shape counts as zero — the failure mode that is indistinguishable from
"nothing changed".

So the count is structural instead: walk the parsed JSON and count every object
carrying a string ``answer`` field, wherever it sits. That is what the census
means by "holds generated answers", it needs no per-file knowledge, and it
reproduces all five published figures exactly — 1 + 25 + 25 + 25 + 36 = 112
checkpoints (``run_20260308_160914.json`` is an aborted run holding one answer,
which is why five files carry 112 rather than 125) and 50 + 275 + 125 = 450
sweeps. ``TestGuardIsNotVacuous`` pins the walker against a fixture holding all
four nestings, so a schema change fails as a wrong number rather than as a
quietly skipped file.


Why ``git ls-files`` and not a directory listing
-------------------------------------------------

The census counts what is **tracked**, which is the whole question — the seven
untracked files are precisely the ones it must not count. Globbing the
directory would count them on the machine that has them and not in CI, i.e.
fail locally for the wrong reason and pass in CI for the wrong reason.

Reading the index rather than ``HEAD`` is also deliberate: it catches a
``git add -A`` at the moment of staging, before the commit exists, which is the
cheapest point to notice. It does mean this suite needs ``git`` on PATH and a
real checkout — true in CI (``actions/checkout@v7``) and on any machine with a
working tree, false in a source tarball. The tarball case fails loudly here
rather than passing vacuously; see ``test_the_index_is_readable``.
"""

import json
import re
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent

_COMPLIANCE = _REPO_ROOT / "docs" / "COMPLIANCE.md"
_RESULTS_DIR = "eval/results"

# The three sweep files the census calls "a closed set predating the rule". No
# future sweep output is tracked by #117's `.gitignore` rule, so this list is
# expected to stay exactly this length — a fourth would be a new tracked shape
# and is caught as one.
_SWEEP_FILES = {
    "reranker_sweep_20260409_133416.json",
    "sweep_retrieval_stage1_20260410_100224.json",
    "sweep_retrieval_stage2_20260410_114731.json",
}

# The two aggregate-score files the census explicitly excludes: "carry aggregate
# scores only and no generated text".
_SCORE_ONLY_FILES = {
    "sweep_retrieval_stage1_20260410_100224.csv",
    "sweep_retrieval_stage2_20260410_114731.csv",
}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


# ── Reading what is tracked ────────────────────────────────────────────────────

def _tracked_files() -> list[str]:
    """Basenames of every file the index holds under ``eval/results/``."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", _RESULTS_DIR],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [entry.rsplit("/", 1)[-1] for entry in out.split("\0") if entry]


def _count_answers(node) -> int:
    """
    Every object carrying a string ``answer``, at any depth.

    Shape-agnostic on purpose — see the module docstring. Counts the field, not
    the container, so a list of results and a dict of combos both work and
    neither needs naming here.
    """
    if isinstance(node, dict):
        found = 1 if isinstance(node.get("answer"), str) else 0
        return found + sum(_count_answers(value) for value in node.values())
    if isinstance(node, list):
        return sum(_count_answers(item) for item in node)
    return 0


def _answers_in(name: str) -> int:
    path = _REPO_ROOT / _RESULTS_DIR / name
    return _count_answers(json.loads(path.read_text(encoding="utf-8")))


def _tracked_census() -> dict[str, int]:
    """``{checkpoint_files, checkpoint_answers, sweep_files, sweep_answers}``."""
    tracked = [name for name in _tracked_files() if name.endswith(".json")]
    checkpoints = [name for name in tracked if name.startswith("run_")]
    sweeps = [name for name in tracked if name in _SWEEP_FILES]
    return {
        "checkpoint_files": len(checkpoints),
        "checkpoint_answers": sum(_answers_in(name) for name in checkpoints),
        "sweep_files": len(sweeps),
        "sweep_answers": sum(_answers_in(name) for name in sweeps),
    }


# ── Reading what is claimed ────────────────────────────────────────────────────

def _recorded_census() -> dict[str, int]:
    """
    The five figures the 2026-07-31 assessment publishes.

    Parsed rather than duplicated here, so that updating the entry is what makes
    this pass — the same coupling ``tests/test_required_checks.py`` holds
    between ``ci.yml`` and the ruleset. Prose can be reworded freely; if a
    rewording moves a figure out of reach of these patterns,
    ``test_every_figure_was_found`` fails and names the one that went missing
    rather than letting an unparsed claim read as agreement.
    """
    text = _COMPLIANCE.read_text(encoding="utf-8")
    figures: dict[str, int] = {}

    total = re.search(r"(\d+) of them across (\w+) files", text)
    if total:
        figures["total_answers"] = int(total.group(1))
        if total.group(2).lower() in _NUMBER_WORDS:
            figures["total_files"] = _NUMBER_WORDS[total.group(2).lower()]

    checkpoints = re.search(
        r"\*\*(\w+) `run_\\?\*\.json` checkpoints, (\d+) answers\.?\*\*", text
    )
    if checkpoints and checkpoints.group(1).lower() in _NUMBER_WORDS:
        figures["checkpoint_files"] = _NUMBER_WORDS[checkpoints.group(1).lower()]
        figures["checkpoint_answers"] = int(checkpoints.group(2))

    sweeps = re.search(r"\*\*(\w+) sweep files, (\d+) answers\.?\*\*", text)
    if sweeps and sweeps.group(1).lower() in _NUMBER_WORDS:
        figures["sweep_files"] = _NUMBER_WORDS[sweeps.group(1).lower()]
        figures["sweep_answers"] = int(sweeps.group(2))

    return figures


_UPDATE_HINT = (
    "\n\nCommitting an eval artifact is fine; leaving the assessment describing "
    "the old set is not. Re-read the 2026-07-31 entry under 'Recorded "
    "assessments' in docs/COMPLIANCE.md, update its figures, and confirm the "
    "purpose-test conclusion still holds for what was added. If this was an "
    "accidental `git add -A`, docs/EVALUATION.md records the decision not to "
    "commit the untracked runs — unstage them instead."
)


# ── The guard ──────────────────────────────────────────────────────────────────

class TestCensusMatchesWhatIsTracked:
    """The five published figures against the index, one assertion each."""

    def test_checkpoint_files(self):
        actual, recorded = _tracked_census(), _recorded_census()
        assert actual["checkpoint_files"] == recorded["checkpoint_files"], (
            f"docs/COMPLIANCE.md records {recorded['checkpoint_files']} tracked "
            f"`run_*.json` checkpoints; the index holds "
            f"{actual['checkpoint_files']}." + _UPDATE_HINT
        )

    def test_checkpoint_answers(self):
        actual, recorded = _tracked_census(), _recorded_census()
        assert actual["checkpoint_answers"] == recorded["checkpoint_answers"], (
            f"docs/COMPLIANCE.md records {recorded['checkpoint_answers']} "
            f"generated answers across the tracked checkpoints; they hold "
            f"{actual['checkpoint_answers']}." + _UPDATE_HINT
        )

    def test_sweep_files_and_answers(self):
        actual, recorded = _tracked_census(), _recorded_census()
        assert actual["sweep_files"] == recorded["sweep_files"], (
            f"docs/COMPLIANCE.md records {recorded['sweep_files']} tracked sweep "
            f"files; the index holds {actual['sweep_files']}.\n"
            "The census calls the sweeps a closed set predating #117's "
            "`.gitignore` rule — if that is no longer true, the entry's "
            "reasoning about them changes and not only its count." + _UPDATE_HINT
        )
        assert actual["sweep_answers"] == recorded["sweep_answers"], (
            f"docs/COMPLIANCE.md records {recorded['sweep_answers']} generated "
            f"answers across the tracked sweeps; they hold "
            f"{actual['sweep_answers']}." + _UPDATE_HINT
        )

    def test_the_totals_add_up(self):
        actual, recorded = _tracked_census(), _recorded_census()
        total_answers = actual["checkpoint_answers"] + actual["sweep_answers"]
        total_files = actual["checkpoint_files"] + actual["sweep_files"]
        assert total_answers == recorded["total_answers"], (
            f"docs/COMPLIANCE.md scopes the assessment to "
            f"{recorded['total_answers']} generated answers; the tracked files "
            f"hold {total_answers}." + _UPDATE_HINT
        )
        # The headline total and the two-shape breakdown are written separately
        # in that entry and can be updated independently. Checking the sum
        # against the headline catches the half-update.
        assert total_files == recorded["total_files"], (
            f"docs/COMPLIANCE.md scopes the assessment to "
            f"{recorded['total_files']} files but its own breakdown now covers "
            f"{total_files}." + _UPDATE_HINT
        )


class TestNoUnassessedShapes:
    """
    The census reasons about two shapes and concludes both qualify in the same
    sense. A third has not been assessed at all.
    """

    def test_no_unassessed_file_shapes(self):
        known = _SWEEP_FILES | _SCORE_ONLY_FILES
        unknown = sorted(
            name for name in _tracked_files()
            if not name.startswith("run_") and name not in known
        )
        assert not unknown, (
            "Tracked files under eval/results/ that the tripwire 2 census does "
            "not describe:\n" + "\n".join(f"  {name}" for name in unknown)
            + "\n\nThe recorded assessment covers `run_*.json` checkpoints and "
            "three named sweep files, and concludes Art. 50(4) does not attach "
            "on a purpose test applied to those two shapes. A new shape is "
            "outside that reasoning — assess it and extend the entry, or add it "
            "to the score-only exclusion here if it carries no generated text."
        )

    def test_score_only_files_are_still_score_only(self):
        # The entry excludes the two `.csv` files on the grounds that they carry
        # "aggregate scores only and no generated text". Cheap to hold: an
        # `answer` column appearing in one would move it into the census.
        for name in sorted(_SCORE_ONLY_FILES):
            path = _REPO_ROOT / _RESULTS_DIR / name
            if not path.is_file():
                continue
            header = path.read_text(encoding="utf-8").splitlines()[0]
            columns = {column.strip().strip('"').lower() for column in header.split(",")}
            assert "answer" not in columns, (
                f"{name} now has an `answer` column, so it holds generated text "
                "and docs/COMPLIANCE.md excludes it on a premise that no longer "
                "holds." + _UPDATE_HINT
            )


# ── The guard's own seams ──────────────────────────────────────────────────────
#
# Every assertion above compares two numbers that are each read from somewhere.
# If either reader silently returns nothing, they agree at zero.

class TestGuardIsNotVacuous:

    def test_the_index_is_readable(self):
        tracked = _tracked_files()
        assert tracked, (
            "`git ls-files -- eval/results` returned nothing. Either the "
            "artifacts are gone from the index, or this is not a git checkout "
            "— in a source tarball the census cannot be verified at all, and "
            "that is a failure rather than a pass."
        )

    def test_the_compliance_entry_still_exists(self):
        text = _COMPLIANCE.read_text(encoding="utf-8")
        assert "committed eval artifacts (tripwire 2)" in text, (
            "The 2026-07-31 tripwire 2 assessment heading is gone from "
            "docs/COMPLIANCE.md. Recorded assessments are the audit trail; "
            "removing one is not the way to make this test pass."
        )

    def test_every_figure_was_found(self):
        expected = {
            "total_answers", "total_files",
            "checkpoint_files", "checkpoint_answers",
            "sweep_files", "sweep_answers",
        }
        missing = sorted(expected - set(_recorded_census()))
        assert not missing, (
            "Figures this test could not find in docs/COMPLIANCE.md: "
            f"{missing}\n\nThe entry was reworded past the patterns in "
            "`_recorded_census`. Update them — an unparsed claim must not read "
            "as agreement."
        )

    def test_the_walker_counts_every_nesting(self):
        # All four shapes this directory actually uses, plus a non-string
        # `answer` that must not count and a nested list inside a dict value.
        fixture = {
            "results": [{"answer": "a"}, {"answer": "b"}, {"error": "x"}],
            "runs": [{"cases": [{"answer": "c"}]}],
            "combos": {
                "k=8": {"results": [{"answer": "d"}, {"answer": "e"}]},
                "k=12": {"results": [{"answer": "f"}]},
            },
            "summary": {"answer": None, "mean": 4.46},
        }
        assert _count_answers(fixture) == 6
        assert _count_answers({}) == 0
        assert _count_answers([]) == 0

    def test_the_counter_reads_real_artifacts(self):
        # A floor, not an inventory — the equalities above keep it exact. This
        # only proves the walker reaches real files and returns something.
        census = _tracked_census()
        assert census["checkpoint_files"] >= 5
        assert census["checkpoint_answers"] >= 112
        assert census["sweep_files"] == 3
