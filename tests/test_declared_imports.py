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
which directory holds the importer. ``Dockerfile`` COPYs ``api/``, ``agent/``
*and* ``pipeline/`` into the serving image, so all three are checked against
``requirements.txt``. ``eval/``, ``tests/`` and ``scripts/`` ship in no image
and are the only things ``requirements-dev.txt`` should hold.

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

# Directories that ship in the serving image, per Dockerfile's COPY allowlist.
# TestRootListMatchesDockerfile asserts this stays true, so #52 — which narrows
# that COPY to the modules the API path actually reaches — fails loudly here
# instead of leaving the guard checking source the image no longer carries.
SHIPPED_ROOTS = ("api", "agent", "pipeline")

# Ships in no image. Note the mechanism, since it is easy to get wrong:
# .dockerignore drops eval/ and tests/ from the build context, but scripts/ is
# not listed there at all and ships nowhere purely because no COPY names it.
# Both Dockerfiles use enumerated allowlists, so the allowlist is what decides.
UNSHIPPED_ROOTS = ("eval", "tests", "scripts")

ALL_ROOTS = SHIPPED_ROOTS + UNSHIPPED_ROOTS

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


def _python_files(root: str) -> list[Path]:
    return [
        p for p in (_REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def _first_party_names(roots: tuple[str, ...]) -> set[str]:
    """
    Names that resolve to source in this repo rather than a dependency, for an
    import appearing under ``roots``.

    Derived from the tree, not listed, and deliberately narrow: the root names
    (importable as packages from the repo root), the top-level modules of the
    bridged directories, and the top-level modules of ``roots`` themselves. Only
    ``glob``, never ``rglob`` — a module nested one level down is not importable
    as a bare name, so counting it would let it mask a real dependency. See the
    module docstring.
    """
    names = set(ALL_ROOTS)
    for root in set(roots) | set(_BRIDGED_DIRS):
        names.update(p.stem for p in (_REPO_ROOT / root).glob("*.py"))
    return names


def _imported_paths(roots: tuple[str, ...]) -> set[str]:
    """
    Dotted module paths imported anywhere under ``roots``.

    Walks the whole AST rather than module-level nodes only: agent/agent.py and
    pipeline/chunker.py:180 both import inside functions, and a function-level
    import is a direct dependency exactly like any other. Relative imports
    (``level > 0``) are first-party by construction and skipped.

    For ``from X import y`` both ``X`` and ``X.y`` are recorded, since ``y`` may
    be a submodule — ``from google.cloud import storage`` is what makes
    ``google-cloud-storage`` resolvable without consulting the environment.
    """
    paths: set[str] = set()
    for root in roots:
        for path in _python_files(root):
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


def _third_party_paths(roots: tuple[str, ...]) -> set[str]:
    """
    Maximal third-party dotted paths — stdlib and first-party removed.

    Maximal meaning: a path that is a proper prefix of another observed path is
    dropped, so `google.cloud` gives way to `google.cloud.storage`. See the
    module docstring for why this is both safe and stricter than pooling
    candidates per top-level name.
    """
    excluded = set(sys.stdlib_module_names) | _first_party_names(roots)
    paths = {
        path for path in _imported_paths(roots)
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


def _undeclared(roots: tuple[str, ...], requirements: Path) -> set[str]:
    declared = _declared_distributions(requirements)
    return {
        path for path in _third_party_paths(roots)
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
    """api/, agent/ and pipeline/ ship in the serving image (Dockerfile)."""

    def test_every_import_is_declared_in_requirements(self):
        undeclared = _undeclared(SHIPPED_ROOTS, _REPO_ROOT / "requirements.txt")
        assert not undeclared, (
            "Imported by shipped source but not declared in requirements.txt:\n"
            + _report(undeclared)
            + "\n\nThese resolve today only because something else pulls them in, "
            "so their versions come from whatever the resolver picks at build "
            "time: two builds of the same commit can ship different versions of "
            "code the app imports, with no diff to review. That is #37 and #42.\n"
            "Placement follows what ships in an image, not which directory "
            "imports it: Dockerfile COPYs api/, agent/ AND pipeline/, so a "
            "package imported by name under any of the three belongs in "
            "requirements.txt. A pin in requirements-dev.txt alone never reaches "
            "the API image."
        )


class TestUnshippedImportsAreDeclared:
    """eval/, tests/ and scripts/ ship in no image."""

    def test_every_import_is_declared_in_requirements_dev(self):
        undeclared = _undeclared(UNSHIPPED_ROOTS, _REPO_ROOT / "requirements-dev.txt")
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
        # all; its presence in UNSHIPPED_ROOTS states the placement rule rather
        # than a claim about its contents.
        assert _python_files(root), f"{root}/ holds no .py files"

    def test_shipped_imports_are_actually_collected(self):
        # Sanity floor, not an inventory: fastapi and langchain-core are the two
        # ends of the serving image's stack, and #37 pinned the second.
        collected = _top_levels(_third_party_paths(SHIPPED_ROOTS))
        assert "fastapi" in collected
        assert "langchain_core" in collected

    def test_first_party_flat_imports_are_excluded(self):
        # The sys.path bridge makes these importable as bare top-level names. If
        # the derivation breaks they surface as undeclared dependencies, and the
        # guard fails on its own repo rather than on a real gap.
        shipped = _first_party_names(SHIPPED_ROOTS)
        for name in ("live_ingest", "rag_chain", "retriever", "sponsorblock"):
            assert name in shipped
        # eval/ is not bridged; its flat imports work because a directly-run
        # script puts its own directory on sys.path.
        assert "run_evals" in _first_party_names(UNSHIPPED_ROOTS)

    def test_unimportable_names_are_not_treated_as_first_party(self):
        # The masking hole: neither is reachable as a bare name from api/, so
        # neither may silence an undeclared dependency there. Nested modules are
        # excluded by globbing one level, cross-root ones by scoping to the
        # group under test.
        shipped = _first_party_names(SHIPPED_ROOTS)
        assert "conftest" not in shipped     # tests/, a different group
        assert "run_evals" not in shipped    # eval/, a different group
        assert "schemas" in shipped          # api/schemas.py, same group
        assert "limiter" in shipped

    def test_function_level_imports_are_collected(self):
        # pipeline/chunker.py:180 imports sponsorblock inside a branch, and
        # api/catalog.py:48 imports google.cloud inside a function — the whole
        # reason this walks the full AST. Module-level-only collection would
        # miss google entirely.
        assert "google" in _top_levels(_third_party_paths(SHIPPED_ROOTS))

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
        paths = _third_party_paths(SHIPPED_ROOTS)
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
            "directory to SHIPPED_ROOTS if a Dockerfile COPYs it into an "
            "image, or to UNSHIPPED_ROOTS if it ships nowhere; if it holds no "
            "first-party source at all, add it to _NON_SOURCE_DIRS."
        )


class TestRootListMatchesDockerfile:
    """
    SHIPPED_ROOTS is a copy of Dockerfile's COPY allowlist, so it can drift.

    #52 proposes narrowing that COPY to the four modules the API path actually
    reaches, which would drop sponsorblock.py and enrich_metadata.py — the only
    importers of `requests`. Under the rule this test encodes, `requests` would
    then belong back in requirements-dev.txt. This assertion is what makes that
    change fail here rather than leave the guard checking source the image no
    longer carries.

    Directory granularity only: a narrowed COPY naming individual .py files
    would not satisfy this as written, which is the point — it should be read
    and updated deliberately, not auto-followed.

    A source counts as a directory because it *is* one on disk, not because it
    was written with a trailing slash. `COPY scripts /app/scripts` is valid
    Docker and ships the directory just as `COPY scripts/ scripts/` does, and
    the slash heuristic this replaced was blind to it — which would have let
    scripts/ into the serving image while it was still checked against
    requirements-dev.txt, reintroducing #42 exactly. Shell and JSON/exec forms
    are both accepted for the same reason.
    """

    def test_shipped_roots_are_the_directories_dockerfile_copies(self):
        dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        copied = set()
        for line in dockerfile.splitlines():
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
            copied.update(
                s.rstrip("/") for s in sources
                if (_REPO_ROOT / s.rstrip("/")).is_dir()
            )
        assert copied == set(SHIPPED_ROOTS), (
            f"Dockerfile copies {sorted(copied)} but SHIPPED_ROOTS is "
            f"{sorted(SHIPPED_ROOTS)}. The serving image's contents changed: "
            "update SHIPPED_ROOTS, and re-check whether any pin in "
            "requirements.txt is now needed only by source that no longer ships "
            "(or vice versa)."
        )
