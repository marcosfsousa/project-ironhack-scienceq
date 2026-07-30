# tests/test_env_defaults.py

"""
``.env.example`` and the ``os.getenv`` fallbacks in code must state the same
values.

Every tunable in this project is read as ``os.getenv(NAME, default)``. That
gives each one two homes: the literal in the code, and the line in
``.env.example`` that documents it. Nothing has ever held the two together, and
they drifted — silently, and in the one direction that is hardest to notice.


Why this is not a style rule
----------------------------

No cloudbuild manifest sets the retrieval variables. The Cloud Run service binds
them out-of-band instead — verified 30 July 2026 against the serving revision,
which carries the sweep winner correctly. So production is *not* what this file
protects.

What it protects is everything that runs unconfigured: local development without
a ``.env``, the image run bare, and — the one that actually bit — the eval
scripts, which import these modules and inherit whatever the literals say. A
value that lives only in ``.env.example`` is documentation of something that is
not happening anywhere the variable is unset.

That is not hypothetical, and the shape of how it happened is the argument for
this file existing.

``1d188bf`` ("feat: Phase 5 — retrieval parameter sweep and tuned defaults") ran
a two-stage sweep over eleven retrieval shapes and five thresholds, declared
``k10_n3_t0.25`` the winner in its own commit message, wrote that winner into
``.env.example`` — and left the code fallbacks at ``top_n=5`` / ``threshold=0.40``,
the pre-sweep values the sweep had just rejected. ``docs/retrieval_sweep_results.md``
called 0.40 "confirmed stale" in the same commit that kept it. It stayed that way
for three months.

It never reached production, which sets the variables explicitly — and that is
precisely why it survived three months. The cost landed on the measurement
instead: ``eval/validate_multilingual.py`` inherits ``SCORE_THRESHOLD`` from
``retriever``, so the Phase 6 multilingual validation ran against 0.40 and
``docs/ARCHITECTURE.md`` recorded 0.40 as the gate — a result describing a
configuration the project had already rejected, produced by the one code path
with no deployment env to correct it. Two docs then quoted ``.env.example`` and
two quoted the code, which is what a contradiction between four files looks like
when nobody is wrong on their own terms.

Two more of the same class were found when this test was first written, both
naming a project that no longer exists: ``PINECONE_INDEX_NAME`` defaulted to
``youtube-qa-bot`` in ``agent/retriever.py`` and ``pipeline/indexer.py`` while
``api/catalog.py`` defaulted to ``scienceq-prod``. Unset, the API would have
served its catalog from one index and its answers from another.


What is compared, and what is not
---------------------------------

A variable is compared when it has **both** a literal default in code and a
concrete value in ``.env.example``.

Secrets are excluded structurally rather than by name. Their code default is
``""`` and their ``.env.example`` value is a ``your_..._here`` placeholder;
neither is a default anybody intends, and comparing them would fail on every key
in the file. ``_is_placeholder`` is the whole rule, and it is asserted directly
in ``TestGuardIsNotVacuous`` so that a real value cannot start being skipped
because it happens to contain the word "key".

Deliberate divergences go in ``_EXPECTED_DIVERGENCE`` with a written reason.
That is the only way to pass while disagreeing, and it costs a sentence — the
same trade ``test_required_checks.py`` makes for an unrequired CI job. An
exception with an empty reason fails.


Read by parsing, not importing
------------------------------

``agent.retriever`` imports ``pinecone`` and ``cohere`` at module scope. Reading
these values by import would make this test a live dependency check and fail it
for reasons that have nothing to do with drift. The defaults are literals in the
source, so they are read from the AST — same approach, and same reason, as
``tests/test_declared_imports.py`` parsing Dockerfile's COPY block rather than
building the image.

The AST walk only accepts ``os.getenv(NAME, default)`` where *both* arguments are
string literals. A computed default is not comparable to a line in a ``.env``
file and is skipped rather than guessed at; ``test_computed_defaults_are_skipped``
pins that.
"""

import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_EXAMPLE = _REPO_ROOT / ".env.example"

# Every directory whose modules are read at runtime with env-driven config.
# eval/ is deliberately absent: its scripts *patch* these variables to sweep
# them (see eval/sweep_retrieval.py), so a literal there is a sweep bound rather
# than a default claiming to match .env.example.
_SOURCE_DIRS = ("agent", "api", "pipeline")


# ── Deliberate divergences ─────────────────────────────────────────────────────
#
# name -> why the two files disagree on purpose. Empty reasons are rejected.

_EXPECTED_DIVERGENCE: dict[str, str] = {
    "LANGCHAIN_TRACING_V2": (
        "Correct as it stands, and load-bearing (issue #17). .env.example sets "
        "true because tracing belongs on in development and evaluation; "
        "agent/rag_chain.py defaults to \"\" because the code must never be the "
        "thing that turns it on. Tracing sends real user questions to "
        "LangSmith, so the fallback is fail-closed by design and the enabling "
        "decision lives in the deployment environment alone — production sets "
        "it false explicitly in cloudbuild-api.yaml. Aligning these would make "
        "an unconfigured deployment start exporting user questions, which is "
        "the failure #17 exists to prevent. LANGSMITH_TRACING is the same "
        "switch read a second way and needs no entry: it is absent from "
        ".env.example, so nothing claims a value for it to disagree with."
    ),
}


# ── Reading .env.example ───────────────────────────────────────────────────────

def _is_placeholder(value: str) -> bool:
    """
    ``.env.example`` fills secrets with ``your_<thing>_here`` rather than a
    value. Matched on that shape alone — not on the variable name — so a
    variable is skipped for how its value reads, not for what it is called.
    """
    return value.startswith("your_") and value.endswith("_here")


def _env_example_values(path: Path) -> dict[str, str]:
    """``{name: value}`` for every concrete assignment in a ``.env``-style file."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        # A commented-out assignment documents a variable without setting it —
        # `# RERANKER_ENABLED=true` in the tuning block is prose, not config.
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if not name.isidentifier() or _is_placeholder(value):
            continue
        values[name] = value
    return values


# ── Reading the code ───────────────────────────────────────────────────────────

def _getenv_defaults(path: Path) -> list[tuple[str, str, int]]:
    """
    ``[(name, default, lineno)]`` for each ``os.getenv(NAME, default)`` in a file
    where both arguments are string literals.

    ``os.environ[...]`` is not collected: it has no default to disagree with,
    and a variable read that way is required rather than defaulted.
    """
    found: list[tuple[str, str, int]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 2:
            continue
        func = node.func
        is_getenv = (
            (isinstance(func, ast.Attribute) and func.attr == "getenv")
            or (isinstance(func, ast.Name) and func.id == "getenv")
        )
        if not is_getenv:
            continue
        name, default = node.args
        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
            continue
        if not (isinstance(default, ast.Constant) and isinstance(default.value, str)):
            continue  # computed default — not comparable to a .env line
        found.append((name.value, default.value, node.lineno))
    return found


def _code_defaults() -> dict[str, set[tuple[str, str, int]]]:
    """``{name: {(relative path, default, lineno)}}`` across the source dirs."""
    defaults: dict[str, set[tuple[str, str, int]]] = {}
    for directory in _SOURCE_DIRS:
        for path in sorted((_REPO_ROOT / directory).rglob("*.py")):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            for name, default, lineno in _getenv_defaults(path):
                defaults.setdefault(name, set()).add((rel, default, lineno))
    return defaults


def _sites(entries: set[tuple[str, str, int]]) -> str:
    return "\n".join(
        f"  {path}:{lineno} -> {default!r}"
        for path, default, lineno in sorted(entries)
    )


def _mismatches(
    env: dict[str, str],
    code: dict[str, set[tuple[str, str, int]]],
    exempt: dict[str, str],
) -> list[str]:
    """
    The comparison itself, taking its inputs as arguments so it can be run
    against a fixture as well as against the repo — see
    ``test_a_real_mismatch_is_caught``. A guard whose only exercise is the
    passing case cannot distinguish "nothing is wrong" from "nothing is
    checked", and this file's whole subject is a check that was not happening.
    """
    out = []
    for name, entries in sorted(code.items()):
        if name not in env or name in exempt:
            continue
        wrong = {e for e in entries if e[1] != env[name]}
        if wrong:
            out.append(f"{name}: .env.example says {env[name]!r}, but\n{_sites(wrong)}")
    return out


# ── The guard ──────────────────────────────────────────────────────────────────

class TestCodeDefaultsMatchEnvExample:
    """Phase 5 (``1d188bf``): the sweep winner reached one file and not the other."""

    def test_every_documented_default_matches_the_code(self):
        mismatches = _mismatches(
            _env_example_values(_ENV_EXAMPLE), _code_defaults(), _EXPECTED_DIVERGENCE
        )
        assert not mismatches, (
            "Code defaults disagree with .env.example:\n\n"
            + "\n\n".join(mismatches)
            + "\n\nThe code literal is what runs when the variable is unset, and "
            "no cloudbuild manifest sets these — so a value that lives only in "
            ".env.example is documentation of something that is not happening. "
            "Change both, or record the divergence in _EXPECTED_DIVERGENCE with "
            "a reason."
        )

    def test_a_variable_has_one_default_everywhere(self):
        # PINECONE_INDEX_NAME was read in three modules and defaulted two ways.
        # Unset, api/catalog.py listed one Pinecone index while agent/retriever.py
        # queried another — a split brain inside one process, and invisible
        # anywhere the variable happens to be set.
        split = []
        for name, entries in sorted(_code_defaults().items()):
            if len({default for _, default, _ in entries}) > 1:
                split.append(f"{name}:\n{_sites(entries)}")

        assert not split, (
            "The same variable is given different fallbacks in different "
            "modules:\n\n" + "\n\n".join(split)
            + "\n\nWhichever is right, the others are a different configuration "
            "reached by whichever module happens to read it first."
        )

    def test_every_divergence_states_a_reason(self):
        for name, reason in _EXPECTED_DIVERGENCE.items():
            assert reason.strip(), (
                f"_EXPECTED_DIVERGENCE[{name!r}] has no reason.\n"
                "An exception without one is indistinguishable from the drift "
                "this file exists to catch."
            )

    def test_divergences_are_still_divergent(self):
        # An entry that has been fixed in code but left here would silently
        # exempt the variable from the check for good.
        env = _env_example_values(_ENV_EXAMPLE)
        code = _code_defaults()
        stale = [
            name
            for name in _EXPECTED_DIVERGENCE
            if name in env
            and name in code
            and all(default == env[name] for _, default, _ in code[name])
        ]
        assert not stale, (
            f"_EXPECTED_DIVERGENCE lists variables that now agree: {stale}\n"
            "Remove the entry so the variable is checked again."
        )


# ── The guard's own seams ──────────────────────────────────────────────────────
#
# Every assertion above passes when the parsers return nothing, so the parsers
# are pinned directly.

class TestGuardIsNotVacuous:

    def test_env_example_exists_and_parses(self):
        assert _ENV_EXAMPLE.is_file(), f"{_ENV_EXAMPLE} is missing"
        env = _env_example_values(_ENV_EXAMPLE)
        # The calibrated retrieval block — the values 1d188bf got wrong.
        assert env["RETRIEVER_FETCH_K"] == "10"
        assert env["RETRIEVER_TOP_N"] == "3"
        assert env["SCORE_THRESHOLD"] == "0.25"

    def test_secret_placeholders_are_skipped(self):
        env = _env_example_values(_ENV_EXAMPLE)
        for name in ("GROQ_API_KEY", "PINECONE_API_KEY", "COHERE_API_KEY"):
            assert name not in env, (
                f"{name} carries a real value in .env.example, or "
                "_is_placeholder stopped matching the your_..._here shape. "
                "The first is a committed secret; the second silently narrows "
                "this whole test."
            )

    def test_commented_assignments_are_not_read(self):
        # The tuning block documents `# RERANKER_ENABLED=true ...` as prose above
        # the live assignment. Reading commented lines would collect explanatory
        # text as configuration.
        text = _ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "# RERANKER_ENABLED=true" in text, (
            "Fixture drift: this asserts a commented assignment is ignored, and "
            ".env.example no longer contains one."
        )
        assert _env_example_values(_ENV_EXAMPLE)["RERANKER_ENABLED"] == "true"

    def test_code_defaults_are_collected(self):
        code = _code_defaults()
        assert code, "No os.getenv defaults found at all — the AST walk is broken."
        by_site = {
            (path, name): default
            for name, entries in code.items()
            for path, default, _ in entries
        }
        assert by_site[("agent/retriever.py", "RETRIEVER_TOP_N")] == "3"
        assert by_site[("agent/retriever.py", "SCORE_THRESHOLD")] == "0.25"
        # Carried an exemption until the production value was read (true). Pinned
        # here so it cannot quietly revert to the false it defaulted to before.
        assert by_site[("agent/retriever.py", "RERANKER_ENABLED")] == "true"
        # tools.py reads the same threshold independently of retriever.py, which
        # is what test_a_variable_has_one_default_everywhere covers.
        assert by_site[("agent/tools.py", "SCORE_THRESHOLD")] == "0.25"

    def test_placeholder_rule(self):
        assert _is_placeholder("your_groq_api_key_here")
        assert not _is_placeholder("scienceq-prod")
        assert not _is_placeholder("0.25")
        # Shape, not substring: a value merely mentioning a key is a real value.
        assert not _is_placeholder("your_key")
        assert not _is_placeholder("api_key_here")

    def test_computed_defaults_are_skipped(self, tmp_path):
        source = tmp_path / "m.py"
        source.write_text(
            "import os\n"
            "A = os.getenv('LITERAL', 'kept')\n"
            "B = os.getenv('COMPUTED', str(1))\n"
            "C = os.getenv('JOINED', 'a' + 'b')\n"
            "D = os.getenv('NO_DEFAULT')\n"
            "E = os.environ['REQUIRED']\n"
            "F = getenv('BARE', 'also-kept')\n",
            encoding="utf-8",
        )
        assert _getenv_defaults(source) == [
            ("LITERAL", "kept", 2),
            ("BARE", "also-kept", 7),
        ]

    def test_a_real_mismatch_is_caught(self):
        # The Phase 5 regression, reconstructed: .env.example carries the sweep
        # winner and the code carries the value it replaced. If this does not
        # report, the passing run above means nothing.
        env = {"RETRIEVER_TOP_N": "3", "SCORE_THRESHOLD": "0.25"}
        code = {
            "RETRIEVER_TOP_N": {("agent/retriever.py", "5", 61)},
            "SCORE_THRESHOLD": {("agent/retriever.py", "0.25", 62)},
        }
        found = _mismatches(env, code, exempt={})
        assert len(found) == 1, found
        assert "RETRIEVER_TOP_N" in found[0]
        assert "'3'" in found[0] and "'5'" in found[0]
        assert "agent/retriever.py:61" in found[0]

        # And an exemption suppresses exactly that one.
        assert _mismatches(env, code, exempt={"RETRIEVER_TOP_N": "because"}) == []

    def test_one_wrong_site_among_several_is_caught(self):
        # A variable read in three modules with one stale fallback is the
        # PINECONE_INDEX_NAME shape. The two correct sites must not mask it.
        env = {"PINECONE_INDEX_NAME": "scienceq-prod"}
        code = {
            "PINECONE_INDEX_NAME": {
                ("agent/retriever.py", "scienceq-prod", 45),
                ("api/catalog.py", "scienceq-prod", 25),
                ("pipeline/indexer.py", "youtube-qa-bot", 82),
            }
        }
        found = _mismatches(env, code, exempt={})
        assert len(found) == 1
        assert "pipeline/indexer.py:82" in found[0]
        assert "agent/retriever.py" not in found[0]

    def test_variables_absent_from_env_example_are_not_compared(self):
        # A code default for something .env.example never mentions is not drift
        # — there is no second statement to disagree with.
        code = {"INTERNAL_ONLY": {("agent/x.py", "whatever", 1)}}
        assert _mismatches({}, code, exempt={}) == []

    def test_env_parser_handles_the_shapes_the_file_uses(self, tmp_path):
        env_file = tmp_path / ".env.example"
        env_file.write_text(
            "# a comment\n"
            "\n"
            "PLAIN=value\n"
            "QUOTED=\"quoted value\"\n"
            "SINGLE='single'\n"
            "SPACED = padded \n"
            "# COMMENTED=ignored\n"
            "SECRET=your_thing_here\n"
            "URL=https://api.example.com/v1\n"
            "NOT AN IDENT=skipped\n",
            encoding="utf-8",
        )
        assert _env_example_values(env_file) == {
            "PLAIN": "value",
            "QUOTED": "quoted value",
            "SINGLE": "single",
            "SPACED": "padded",
            # An `=` in the value must survive: partition, not split.
            "URL": "https://api.example.com/v1",
        }
