# tests/test_declared_imports.py

"""
Every package imported by name must be declared in a requirements file (#39).

#37 pinned four packages that ``api/`` and ``agent/`` imported without
declaring, and #42 pinned a fifth. Both were found by reading, and nothing
stopped a sixth: ``.github/workflows/ci.yml`` installs ``requirements.txt``
plus ``pytest`` and builds no image, so a new direct import resolves in CI —
pulled in transitively — and only becomes visible when a transitive parent
drops it or bumps it across a major. This test is the guard.

The rule it encodes is the one #51 settled and wrote into
``requirements-dev.txt``: placement follows **what ships in an image**, not
which directory holds the importer.

Since #52 that rule is finer than a directory. ``Dockerfile`` COPYs ``api/``
and ``agent/`` wholesale, but only the four ``pipeline/`` modules the API path
actually reaches — ``live_ingest`` and the ``cleaner``, ``chunker`` and
``embedder`` it imports. Those are checked against ``requirements.txt``. The
six batch-only modules ship in ``Dockerfile.pipeline`` alone and are checked,
with ``eval/``, ``tests/`` and ``scripts/``, against ``requirements-dev.txt``.
That file inherits ``requirements.txt`` via ``-r``, so it is the looser of the
two sets and the split can only ever be conservative.

Two other classes exist because splitting a directory across two manifests
introduces failure modes a single list did not have.
``TestShippedSourceMatchesDockerfile`` keeps both halves equal to the COPY
allowlist and insists every ``pipeline/`` module lands in exactly one of them,
so a new module cannot escape both. ``TestShippedSourceIsSelfContained``
asserts that shipped source only bare-imports modules that also ship — the
enumerated-COPY drift #52 flagged, caught here from the AST rather than only by
the image build in ``.github/workflows/ci.yml``.

The check is one-directional — imports ⊆ declared, never the reverse. A
declared package nothing imports is not a finding here: ``uvicorn`` is the
serving entrypoint and is imported by no source file, and ``limits`` is pinned
under the narrow unimported-transitive exception #41 settled. Whether an
unimported package belongs in the file at all is a judgment this test does not
make; see the ``limits`` block at the bottom of ``requirements.txt``.


Resolving an import name to a distribution
------------------------------------------

This is the fiddly part, and the order of the two mechanisms matters.

**Dotted paths first, matched against what is declared.** ``langchain_core``
normalizes to ``langchain-core`` and ``from google.cloud import storage``
yields ``google.cloud.storage`` → ``google-cloud-storage``. This needs no
installed environment, so it gives the same verdict in CI, in the image, and on
a developer machine with a partial env.

**``packages_distributions()`` second, and only when it is unambiguous.** It
maps against the installed set rather than a hand-maintained table, which is
what gets ``dotenv`` → ``python-dotenv`` right. But it keys on the *top-level*
name, so for a namespace package it returns every distribution contributing to
that namespace — ``google`` resolves to ``google-auth``, ``google-api-core``,
``googleapis-common-protos``, ``protobuf`` and more, depending only on what
happens to be installed. Trusting that would let a future ``import
google.protobuf`` pass on ``google-cloud-storage``'s declaration, and on a
partial local env it reports the opposite: a false failure on an import that is
correctly declared. So a mapping is consulted only when it names exactly one
distribution; an ambiguous one is discarded and the dotted path has to carry
the match.

**Each maximal path is resolved on its own**, which is what keeps a namespace
honest. Paths that are a proper prefix of another observed path are dropped
first — ``from google.cloud import storage`` records both ``google.cloud`` and
``google.cloud.storage``, and only the longer one is checked. That is safe in
one direction and strict in the other: a prefix's candidates are always a
subset of the longer path's, so dropping it can never lose a match, while
resolving paths separately means ``google.protobuf`` cannot pass on
``google-cloud-storage``'s declaration the way it would if candidates were
pooled per top-level name. Verified by probe, not by inspection — pooling was
the first implementation and it let exactly that through.


Two other ways this test could quietly stop working
---------------------------------------------------

**The first-party set is derived from the filesystem, never hand-listed, and
scoped to what is actually importable flat.** ``api/__init__.py`` and
``agent/agent.py`` insert ``agent/`` and ``pipeline/`` into ``sys.path``, which
is what makes their top-level modules resolvable as bare names — ``import
live_ingest``, ``from rag_chain import ...``, ``import sponsorblock``. A script
run directly also gets its own directory on ``sys.path``, which is how
``eval/sweep_retrieval.py`` reaches ``from run_evals import ...``. So the set is
the root names, plus the top-level modules of the bridged directories, plus the
top-level modules of the roots being checked — and nothing else.

Scoping it that way is the point, not an optimization. Pooling every ``.py``
basename from every root at any depth was the first implementation, and it
meant an unshipped filename could silence a shipped import: a stray
``tests/httpx.py`` or ``api/helpers/httpx.py`` made an undeclared ``import
httpx`` in ``api/`` disappear. Neither is importable as ``httpx`` from anywhere
in this repo, so neither should have counted. What remains is same-directory
shadowing (``api/httpx.py`` masking ``httpx`` for a directly-run script in
``api/``), which is real Python behaviour rather than an artifact of this test.

If the bridge is ever widened — say ``api/`` is added to it — a bare ``from
schemas import ...`` would resolve at runtime but is not in this set, so it
surfaces here as an undeclared dependency. That is a confusing message rather
than a silent hole, which is the right direction to fail in; widen
``_BRIDGED_DIRS`` to match.

**Names are normalized per PEP 503 on both sides.**
``packages_distributions()`` returns the name as recorded, which is not always
the name the requirements file uses — ``typing_extensions`` maps to
``typing_extensions`` while ``requirements.txt`` says ``typing-extensions``.
"""

import ast
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Source that ships in the serving image, per Dockerfile's COPY allowlist.
#
# api/ and agent/ ship wholesale. pipeline/ no longer does: #52 narrowed that
# COPY to the closure of what the API path imports, so four of its ten modules
# ship here and the other six ship only in the batch image. That is why this is
# written as paths rather than directory names — the allowlist is now finer
# than a directory, and a guard that could only speak in directories would have
# to either over-claim (checking source the image does not carry) or give up.
#
# TestShippedSourceMatchesDockerfile asserts both halves against Dockerfile, so
# a COPY edit not reflected here fails loudly rather than silently drifting.
SHIPPED_DIRS = ("api", "agent")
SHIPPED_MODULES = (
    "pipeline/chunker.py",
    "pipeline/cleaner.py",
    "pipeline/embedder.py",
    "pipeline/live_ingest.py",
)
SHIPPED = SHIPPED_DIRS + SHIPPED_MODULES

# Ships in no image. Note the mechanism, since it is easy to get wrong:
# .dockerignore drops eval/ and tests/ from the build context, but scripts/ is
# not listed there at all and ships nowhere purely because no COPY names it.
# Both Dockerfiles use enumerated allowlists, so the allowlist is what decides.
UNSHIPPED_DIRS = ("eval", "tests", "scripts")

# The pipeline/ modules Dockerfile leaves out. "Unshipped" is relative to the
# *serving* image only — Dockerfile.pipeline still COPYs pipeline/ wholesale,
# so these do ship in the batch image, which installs requirements-dev.txt.
# Checking them against that file is therefore the correct side of the split
# and cannot let an undeclared package through: requirements-dev.txt inherits
# requirements.txt via `-r`, so it is the looser of the two sets.
UNSHIPPED_MODULES = (
    "pipeline/bootstrap_metadata.py",
    "pipeline/enrich_metadata.py",
    "pipeline/indexer.py",
    "pipeline/run.py",
    "pipeline/sponsorblock.py",
    "pipeline/transcript_extractor.py",
)
UNSHIPPED = UNSHIPPED_DIRS + UNSHIPPED_MODULES

# Top-level directories holding first-party Python, independent of the ship
# split above. TestEveryRootIsCovered uses it to prove no source escapes the
# guard; _first_party_names uses it to decide what is not a dependency.
ALL_ROOTS = SHIPPED_DIRS + ("pipeline",) + UNSHIPPED_DIRS

# Directories inserted into sys.path by api/__init__.py and agent/agent.py,
# which is what makes their top-level modules importable as bare names from
# anywhere in the app. TestEveryRootIsCovered keeps ALL_ROOTS honest; this one
# is kept honest by failing loudly — see the module docstring.
_BRIDGED_DIRS = ("agent", "pipeline")

# Directories that hold no first-party Python source. Anything else containing
# a .py file must be covered by ALL_ROOTS.
_NON_SOURCE_DIRS = {".git", ".claude", "__pycache__", "frontend", "node_modules"}


# ── Collecting imports ─────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """PEP 503 normalization, applied to both sides of every comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _python_files(spec: str) -> list[Path]:
    """
    The .py files a spec covers: one module if it names a file, the whole tree
    if it names a directory.
    """
    target = _REPO_ROOT / spec
    if target.is_file():
        return [target]
    return [
        p for p in target.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def _spec_roots(specs: tuple[str, ...]) -> set[str]:
    """Top-level directory of each spec — ``pipeline/chunker.py`` → ``pipeline``."""
    return {spec.split("/")[0] for spec in specs}


def _first_party_names(specs: tuple[str, ...]) -> set[str]:
    """
    Names that resolve to source in this repo rather than a dependency, for an
    import appearing under ``specs``.

    Derived from the tree, not listed, and deliberately narrow: the root names
    (importable as packages from the repo root), the top-level modules of the
    bridged directories, and the top-level modules of the roots ``specs`` live
    under. Only ``glob``, never ``rglob`` — a module nested one level down is
    not importable as a bare name, so counting it would let it mask a real
    dependency. See the module docstring.

    Derived from the *directory on disk*, not from the shipped subset, and that
    distinction survives #52 on purpose. ``pipeline/chunker.py`` ships and
    imports ``sponsorblock``, which does not; resolving first-party names
    against the shipped subset would make ``sponsorblock`` read as an
    undeclared third-party package, and the failure would be unfixable — no
    requirements file can declare a first-party module. Whether shipped source
    may import an unshipped module is a real question, but it is a different
    one, and TestShippedSourceIsSelfContained is where it is asked.
    """
    names = set(ALL_ROOTS)
    for root in _spec_roots(specs) | set(_BRIDGED_DIRS):
        names.update(p.stem for p in (_REPO_ROOT / root).glob("*.py"))
    return names


def _imported_paths(specs: tuple[str, ...]) -> set[str]:
    """
    Dotted module paths imported anywhere under ``specs``.

    Walks the whole AST rather than module-level nodes only: agent/agent.py and
    pipeline/chunker.py:180 both import inside functions, and a function-level
    import is a direct dependency exactly like any other. Relative imports
    (``level > 0``) are first-party by construction and skipped.

    For ``from X import y`` both ``X`` and ``X.y`` are recorded, since ``y`` may
    be a submodule — ``from google.cloud import storage`` is what makes
    ``google-cloud-storage`` resolvable without consulting the environment.
    """
    paths: set[str] = set()
    for spec in specs:
        for path in _python_files(spec):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    paths.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        paths.add(node.module)
                        paths.update(
                            f"{node.module}.{alias.name}"
                            for alias in node.names
                            if alias.name != "*"
                        )
    return paths


def _third_party_paths(specs: tuple[str, ...]) -> set[str]:
    """
    Maximal third-party dotted paths — stdlib and first-party removed.

    Maximal meaning: a path that is a proper prefix of another observed path is
    dropped, so `google.cloud` gives way to `google.cloud.storage`. See the
    module docstring for why this is both safe and stricter than pooling
    candidates per top-level name.
    """
    excluded = set(sys.stdlib_module_names) | _first_party_names(specs)
    paths = {
        path for path in _imported_paths(specs)
        if path.split(".")[0] not in excluded
    }
    return {
        path for path in paths
        if not any(other != path and other.startswith(f"{path}.") for other in paths)
    }


def _top_levels(paths: set[str]) -> set[str]:
    return {path.split(".")[0] for path in paths}


# ── Resolving to declared distributions ────────────────────────────────────────

def _declared_distributions(requirements: Path) -> set[str]:
    """
    Normalized distribution names declared in a requirements file.

    ``-r`` is followed recursively and resolved relative to the referencing
    file's directory, the way pip does — that inheritance is why
    requirements-dev.txt covers requirements.txt's pins without restating them.
    Parsed as text rather than read from the environment on purpose: CI installs
    requirements.txt plus pytest and never requirements-dev.txt, so the declared
    side cannot come from what happens to be importable.
    """
    declared: set[str] = set()
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        # A `#` opens a comment only at the start of a line or after
        # whitespace, per pip. Splitting on every `#` would read the fragment
        # of a URL requirement as a comment and declare a distribution named
        # `git` for `git+https://...#egg=foo`.
        line = re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()
        if not line:
            continue
        # Both spellings of the include flag, with or without `=`.
        include = re.match(r"^(?:-r|--requirement)[=\s]\s*(.+)$", line)
        if include:
            declared |= _declared_distributions(
                requirements.parent / include.group(1).strip()
            )
            continue
        # Name is everything up to the first extras bracket or version
        # specifier — `uvicorn[standard]==0.51.0` declares `uvicorn`. A URL or
        # path requirement has no name in this position and is skipped rather
        # than guessed at; none exist today, and one appearing silently
        # undeclared is safer than one declaring `https`.
        if re.match(r"^[A-Za-z0-9]", line) and "://" not in line:
            match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
            if match:
                declared.add(_normalize(match.group(1)))
    return declared


_INSTALLED_PACKAGES = packages_distributions()


def _candidate_distributions(path: str) -> set[str]:
    """
    Distribution names that could satisfy one dotted import path.

    Every dotted prefix of the path, plus the installed-environment mapping for
    its top-level name when that mapping names exactly one distribution. See the
    module docstring for why an ambiguous mapping is discarded.
    """
    parts = path.split(".")
    candidates = {
        _normalize(".".join(parts[:i])) for i in range(1, len(parts) + 1)
    }
    mapped = _INSTALLED_PACKAGES.get(parts[0]) or []
    if len(mapped) == 1:
        candidates.add(_normalize(mapped[0]))
    return candidates


def _undeclared(specs: tuple[str, ...], requirements: Path) -> set[str]:
    declared = _declared_distributions(requirements)
    return {
        path for path in _third_party_paths(specs)
        if not (_candidate_distributions(path) & declared)
    }


def _report(undeclared: set[str]) -> str:
    lines = []
    for path in sorted(undeclared):
        top = path.split(".")[0]
        mapped = _INSTALLED_PACKAGES.get(top) or []
        if len(mapped) > 1:
            note = (
                f"`{top}` is a namespace package shared by {len(mapped)} "
                "installed distributions, so the import path has to name the "
                "distribution (google.cloud.storage -> google-cloud-storage)"
            )
        elif mapped:
            note = f"provided by {mapped[0]}"
        else:
            note = "not installed here; matched on the import name alone"
        lines.append(f"  {path}\n    {note}")
    return "\n".join(lines)


# ── The guard ──────────────────────────────────────────────────────────────────

class TestShippedImportsAreDeclared:
    """
    Source the serving image carries: api/, agent/, and the four pipeline/
    modules the API path reaches (Dockerfile).
    """

    def test_every_import_is_declared_in_requirements(self):
        undeclared = _undeclared(SHIPPED, _REPO_ROOT / "requirements.txt")
        assert not undeclared, (
            "Imported by shipped source but not declared in requirements.txt:\n"
            + _report(undeclared)
            + "\n\nThese resolve today only because something else pulls them in, "
            "so their versions come from whatever the resolver picks at build "
            "time: two builds of the same commit can ship different versions of "
            "code the app imports, with no diff to review. That is #37 and #42.\n"
            "Placement follows what ships in an image, not which directory "
            "imports it. Since #52 that is finer than a directory: Dockerfile "
            "COPYs api/ and agent/ wholesale but only four named pipeline/ "
            "modules, so a package imported by one of those four belongs in "
            "requirements.txt, while one imported only by the other six belongs "
            "in requirements-dev.txt. A pin there alone never reaches the API "
            "image."
        )


class TestUnshippedImportsAreDeclared:
    """
    Source the serving image does not carry: eval/, tests/, scripts/, and the
    six batch-only pipeline/ modules.
    """

    def test_every_import_is_declared_in_requirements_dev(self):
        undeclared = _undeclared(UNSHIPPED, _REPO_ROOT / "requirements-dev.txt")
        assert not undeclared, (
            "Imported by unshipped source but not declared in "
            "requirements-dev.txt:\n"
            + _report(undeclared)
            + "\n\nrequirements-dev.txt inherits requirements.txt via `-r`, so a "
            "package already declared there satisfies this too. Add it here only "
            "if the importer ships in no image at all."
        )


# ── The guard's own seams ──────────────────────────────────────────────────────
#
# A collector that silently returns nothing passes both assertions above. These
# make the vacuous-green failure mode impossible.

class TestGuardIsNotVacuous:

    @pytest.mark.parametrize("root", ALL_ROOTS)
    def test_every_root_exists(self, root):
        assert (_REPO_ROOT / root).is_dir(), f"{root}/ is missing"

    @pytest.mark.parametrize("root", [r for r in ALL_ROOTS if r != "scripts"])
    def test_every_root_holds_source(self, root):
        # scripts/ holds no third-party imports today and may hold no .py at
        # all; its presence in UNSHIPPED_DIRS states the placement rule rather
        # than a claim about its contents.
        assert _python_files(root), f"{root}/ holds no .py files"

    def test_shipped_imports_are_actually_collected(self):
        # Sanity floor, not an inventory: fastapi and langchain-core are the two
        # ends of the serving image's stack, and #37 pinned the second.
        collected = _top_levels(_third_party_paths(SHIPPED))
        assert "fastapi" in collected
        assert "langchain_core" in collected

    def test_file_specs_collect_only_their_own_module(self):
        # The narrowed half of SHIPPED is file-granular, so a spec naming one
        # module must not drag in its neighbours. cohere is imported by
        # pipeline/embedder.py (shipped) and yt_dlp only by
        # pipeline/transcript_extractor.py (not shipped) — if a file spec
        # silently widened to its directory, the second would appear here.
        collected = _top_levels(_third_party_paths(("pipeline/embedder.py",)))
        assert collected == {"cohere", "dotenv"}, collected

    def test_first_party_flat_imports_are_excluded(self):
        # The sys.path bridge makes these importable as bare top-level names. If
        # the derivation breaks they surface as undeclared dependencies, and the
        # guard fails on its own repo rather than on a real gap.
        shipped = _first_party_names(SHIPPED)
        for name in ("live_ingest", "rag_chain", "retriever", "sponsorblock"):
            assert name in shipped
        # sponsorblock is in that list on purpose: it no longer ships, but it is
        # still first-party source, so it must not read as an undeclared
        # package. See _first_party_names.
        assert "sponsorblock" not in {Path(m).stem for m in SHIPPED_MODULES}
        # eval/ is not bridged; its flat imports work because a directly-run
        # script puts its own directory on sys.path.
        assert "run_evals" in _first_party_names(UNSHIPPED)

    def test_unimportable_names_are_not_treated_as_first_party(self):
        # The masking hole: neither is reachable as a bare name from api/, so
        # neither may silence an undeclared dependency there. Nested modules are
        # excluded by globbing one level, cross-root ones by scoping to the
        # group under test.
        shipped = _first_party_names(SHIPPED)
        assert "conftest" not in shipped     # tests/, a different group
        assert "run_evals" not in shipped    # eval/, a different group
        assert "schemas" in shipped          # api/schemas.py, same group
        assert "limiter" in shipped

    def test_function_level_imports_are_collected(self):
        # pipeline/chunker.py:180 imports sponsorblock inside a branch, and
        # api/catalog.py:48 imports google.cloud inside a function — the whole
        # reason this walks the full AST. Module-level-only collection would
        # miss google entirely.
        assert "google" in _top_levels(_third_party_paths(SHIPPED))

    def test_namespace_package_resolves_through_the_dotted_path(self):
        # The case the installed-environment mapping cannot answer: `google`
        # maps to whatever contributes to the namespace, so the match has to
        # come from google.cloud.storage → google-cloud-storage.
        assert "google-cloud-storage" in _candidate_distributions("google.cloud.storage")

    def test_namespace_siblings_do_not_share_a_declaration(self):
        # The hole that pooling candidates per top-level name left open, and the
        # reason paths are resolved individually: google-cloud-storage being
        # declared must not vouch for an undeclared google.protobuf.
        assert "google-cloud-storage" not in _candidate_distributions(
            "google.protobuf.json_format"
        )

    def test_only_maximal_paths_are_checked(self):
        # `from google.cloud import storage` records google.cloud too; checking
        # that prefix on its own would fail against google-cloud-storage.
        paths = _third_party_paths(SHIPPED)
        assert "google.cloud.storage" in paths
        assert "google.cloud" not in paths

    def test_requirements_dev_inherits_requirements(self):
        # The `-r` is load-bearing: it is what keeps the pipeline image and the
        # API image from drifting onto different versions (#29, #42).
        base = _declared_distributions(_REPO_ROOT / "requirements.txt")
        dev = _declared_distributions(_REPO_ROOT / "requirements-dev.txt")
        assert base and base <= dev

    def test_extras_and_comments_parse(self):
        base = _declared_distributions(_REPO_ROOT / "requirements.txt")
        assert "uvicorn" in base            # declared as uvicorn[standard]==...
        assert "typing-extensions" in base  # hyphenated here, underscored on import
        # requirements.txt is mostly prose; none of it may become a declaration.
        assert "the" not in base and "note" not in base

    @pytest.mark.parametrize("line, expected", [
        ("fastapi==0.140.10", {"fastapi"}),
        ("uvicorn[standard]==0.51.0", {"uvicorn"}),
        ("# a whole-line comment", set()),
        ("requests==2.34.2  # trailing comment", {"requests"}),
        ("git+https://example.invalid/x.git#egg=foo", set()),
        ("https://example.invalid/x-1.0-py3-none-any.whl", set()),
    ])
    def test_requirement_line_forms(self, line, expected, tmp_path):
        path = tmp_path / "r.txt"
        path.write_text(line + "\n", encoding="utf-8")
        assert _declared_distributions(path) == expected

    def test_both_include_spellings_are_followed(self, tmp_path):
        (tmp_path / "base.txt").write_text("fastapi==1.0\n", encoding="utf-8")
        for spelling in ("-r base.txt", "--requirement base.txt",
                         "--requirement=base.txt"):
            path = tmp_path / "top.txt"
            path.write_text(f"{spelling}\npydantic==2.0\n", encoding="utf-8")
            assert _declared_distributions(path) == {"fastapi", "pydantic"}


class TestEveryRootIsCovered:
    """
    No Python source in the repo escapes the guard.

    Without this, ALL_ROOTS is the same hand-maintained list this file argues
    against for first-party names, and the docstring's claim to check *every*
    package imported by name holds only by coincidence of nobody having added a
    directory. That is not hypothetical here: `app/` held the Streamlit
    frontend until f494e16 sunset it, so this tree demonstrably gains and loses
    top-level Python directories. A new `worker.py` at the root, or a new
    `app/` package, would import whatever it liked with nothing declared.

    Directories with no first-party Python are excluded by name; everything
    else containing a .py must be covered.
    """

    def test_no_python_source_lives_outside_the_roots(self):
        uncovered = sorted(
            str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
            for path in _REPO_ROOT.rglob("*.py")
            if not _NON_SOURCE_DIRS & set(path.relative_to(_REPO_ROOT).parts)
            and path.relative_to(_REPO_ROOT).parts[0] not in ALL_ROOTS
        )
        assert not uncovered, (
            "Python source outside the directories this guard checks:\n"
            + "\n".join(f"  {p}" for p in uncovered)
            + "\n\nNothing declares the imports in these files. Add the "
            "directory to SHIPPED_DIRS if a Dockerfile COPYs it into an "
            "image, or to UNSHIPPED_DIRS if it ships nowhere; if it holds no "
            "first-party source at all, add it to _NON_SOURCE_DIRS."
        )


def _dockerfile_logical_lines(text: str) -> list[str]:
    """
    Dockerfile instructions, with backslash continuations joined.

    Needed since #52: the narrowed COPY spans several lines, and a line-based
    parser would read the trailing ``\\`` of ``COPY pipeline/cleaner.py \\`` as
    part of a path and miss every module after the first. Docker also drops a
    comment line appearing *inside* a continuation, so this does too — otherwise
    a commented-out path in the block would be collected as a real source.
    """
    lines: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if buffer and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        lines.append(buffer + stripped)
        buffer = ""
    if buffer:
        lines.append(buffer)
    return lines


class TestShippedSourceMatchesDockerfile:
    """
    SHIPPED_DIRS and SHIPPED_MODULES copy Dockerfile's COPY allowlist, so they
    can drift.

    Before #52 this only had to compare directories. That COPY now names four
    pipeline/ modules individually, so the comparison happens at both
    granularities — a directory source must appear in SHIPPED_DIRS, a .py
    source in SHIPPED_MODULES. A directory-only check would still pass while
    the guard reasoned about six modules the serving image no longer carries,
    which is precisely the drift this class exists to prevent.

    A source counts as a directory because it *is* one on disk, not because it
    was written with a trailing slash. `COPY scripts /app/scripts` is valid
    Docker and ships the directory just as `COPY scripts/ scripts/` does, and
    the slash heuristic this replaced was blind to it — which would have let
    scripts/ into the serving image while it was still checked against
    requirements-dev.txt, reintroducing #42 exactly. Shell and JSON/exec forms
    are both accepted for the same reason.

    Non-.py sources are ignored rather than asserted on: data/metadata.json
    ships too, but it declares no imports, so it is outside what this guard
    reasons about.
    """

    def _copied(self) -> tuple[set[str], set[str]]:
        """(directories, files) named as COPY sources in Dockerfile."""
        dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        dirs: set[str] = set()
        files: set[str] = set()
        for line in _dockerfile_logical_lines(dockerfile):
            match = re.match(r"^\s*COPY\s+(.+)$", line, flags=re.IGNORECASE)
            if not match:
                continue
            # JSON/exec form: COPY ["src/", "dst/"]. Strip the punctuation and
            # both forms tokenize the same way.
            tokens = match.group(1).strip().strip("[]").replace(",", " ").split()
            tokens = [t.strip('"').strip("'") for t in tokens]
            # --chown / --from and friends are flags, not paths.
            tokens = [t for t in tokens if not t.startswith("--")]
            if len(tokens) < 2:
                continue
            # Last token is the destination; everything before it is a source.
            *sources, _dest = tokens
            for source in sources:
                source = source.rstrip("/")
                target = _REPO_ROOT / source
                if target.is_dir():
                    dirs.add(source)
                elif target.is_file():
                    files.add(source)
        return dirs, files

    def test_copied_directories_are_the_shipped_dirs(self):
        dirs, _ = self._copied()
        assert dirs == set(SHIPPED_DIRS), (
            f"Dockerfile COPYs the directories {sorted(dirs)} but SHIPPED_DIRS "
            f"is {sorted(SHIPPED_DIRS)}. The serving image's contents changed: "
            "update SHIPPED_DIRS, and re-check whether any pin in "
            "requirements.txt is now needed only by source that no longer ships "
            "(or vice versa)."
        )

    def test_copied_modules_are_the_shipped_modules(self):
        _, files = self._copied()
        copied_py = {f for f in files if f.endswith(".py")}
        assert copied_py == set(SHIPPED_MODULES), (
            f"Dockerfile COPYs the modules {sorted(copied_py)} but "
            f"SHIPPED_MODULES is {sorted(SHIPPED_MODULES)}. An enumerated COPY "
            "is an implicit module list (#52): if it grew, the new module's "
            "imports are checked against requirements.txt only once it is "
            "listed here; if it shrank, this guard is still vouching for source "
            "the image dropped. Update SHIPPED_MODULES and UNSHIPPED_MODULES "
            "together — every pipeline/ module belongs to exactly one."
        )

    def test_every_pipeline_module_is_classified(self):
        # The drift this catches is the quiet one: a new pipeline/ module is
        # picked up by neither list, so nothing checks its imports against
        # either requirements file. Splitting a directory across two manifests
        # only works if the split is total.
        on_disk = {
            f"pipeline/{path.name}"
            for path in (_REPO_ROOT / "pipeline").glob("*.py")
        }
        classified = set(SHIPPED_MODULES) | set(UNSHIPPED_MODULES)
        assert on_disk == classified, (
            f"pipeline/ holds {sorted(on_disk)} but SHIPPED_MODULES + "
            f"UNSHIPPED_MODULES cover {sorted(classified)}. Add each new module "
            "to SHIPPED_MODULES if Dockerfile COPYs it, otherwise to "
            "UNSHIPPED_MODULES."
        )
        assert not set(SHIPPED_MODULES) & set(UNSHIPPED_MODULES), (
            "A module cannot be both shipped and unshipped."
        )


class TestShippedSourceIsSelfContained:
    """
    Shipped source may only bare-import pipeline/ modules that also ship.

    This is the repo-side half of the gate #52 asked for, and it exists because
    an enumerated COPY is an implicit module list that drifts. The api-image job
    in .github/workflows/ci.yml catches the same class of break by building the
    image and running ``import api.main``, but that needs Docker and a full
    dependency install; this catches it from the AST in the pytest job, in
    milliseconds. Between them, adding ``import transcript_extractor`` to
    live_ingest fails on the PR instead of at deploy — which matters because
    merging to main deploys to Cloud Run.

    Note what this does *not* claim. It reasons about first-party modules, not
    packages: TestShippedImportsAreDeclared already covers distributions, and
    _first_party_names deliberately keeps unshipped modules out of that check so
    they never read as undeclared dependencies. This is the other axis.

    The exception set is the point of the class rather than a wart. #52 chose to
    drop sponsorblock.py and document the seam rather than inject the
    dependency, so exactly one shipped module names an unshipped one and it is
    recorded here with its reachability argument. Equality, not a subset, so
    both directions are deliberate: a new unshipped import fails, and so does
    removing this one without deleting the entry.
    """

    # pipeline/chunker.py:180 imports sponsorblock inside the skip_sponsors
    # branch of chunk_transcript, whose only caller is run() — the batch
    # entrypoint, which does not exist in the serving image. The import is
    # unreachable there, and the ImportError it would raise if that changed is
    # the deliberate trade #52 recorded: a loud failure if the API path ever
    # grows a caller, rather than the silent version drift #42 had to fix.
    DOCUMENTED_SEAMS = {("pipeline/chunker.py", "sponsorblock")}

    def test_shipped_source_only_imports_shipped_modules(self):
        unshipped_stems = {Path(module).stem for module in UNSHIPPED_MODULES}
        found: set[tuple[str, str]] = set()
        for spec in SHIPPED:
            for path in _python_files(spec):
                rel = str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
                tree = ast.parse(
                    path.read_text(encoding="utf-8"), filename=str(path)
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name.split(".")[0] for alias in node.names]
                    elif (
                        isinstance(node, ast.ImportFrom)
                        and node.level == 0
                        and node.module
                    ):
                        names = [node.module.split(".")[0]]
                    else:
                        continue
                    found.update(
                        (rel, name) for name in names if name in unshipped_stems
                    )
        assert found == self.DOCUMENTED_SEAMS, (
            "Shipped source imports pipeline/ modules the serving image does "
            f"not carry.\n  found:    {sorted(found)}\n"
            f"  expected: {sorted(self.DOCUMENTED_SEAMS)}\n\n"
            "An import that appears here but is not documented breaks the image "
            "at import time, and CI would only catch it in the api-image job "
            "(or, before that job existed, at deploy). Either add the module to "
            "Dockerfile's COPY and SHIPPED_MODULES, or restructure so shipped "
            "code does not name it.\n"
            "An entry that is documented but no longer found means the seam was "
            "closed — delete it from DOCUMENTED_SEAMS rather than leaving a "
            "comment describing code that no longer exists."
        )
