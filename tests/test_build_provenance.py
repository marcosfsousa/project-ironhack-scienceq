# tests/test_build_provenance.py

"""
Every image build must run with provenance off (#150).

``keep-5-most-recent`` keeps the 5 most recent **versions** per ``scienceq-*``
package, and rollback needs 5 **deploys**. The two are the same only while each
push stores one version. On the ``regular`` Cloud Build worker release, the
default from 2027-03-28, BuildKit adds provenance by default: each push stores
an image index and two children, and #149 measured the result — the rule kept
3 deploys, not 5. Nothing errors. The rollback window just shrinks.

``BUILDX_NO_DEFAULT_ATTESTATIONS=1`` on the build step turns provenance off.
#151 tested it on both workers: on ``regular`` each push is one OCI manifest
again, and on ``legacy`` Docker 20.10 ignores the variable and builds exactly
as before. ``--provenance=false`` was rejected because ``legacy`` fails on the
unknown flag.


What is checked
---------------

``TestEveryBuildIsProvenanceOff``
    Every ``docker build`` step in a Cloud Build config sets
    ``BUILDX_NO_DEFAULT_ATTESTATIONS=1`` in its own ``env``. Build-wide
    ``options.env`` does not count: #151 tested only the step-level form.

``TestEveryBuildIsRead``
    A step that mentions both ``docker`` and ``build`` but is not a shape this
    file reads (a ``bash`` entrypoint running ``docker build``, say) is
    reported, not skipped. Otherwise a build written that way would pass
    without being checked.

The broken-input cases strip, change or move the variable in a copy of each
config and assert the check reports it. A check that has never failed has not
been shown to check anything.


What this test does not check
-----------------------------

It reads the configs, not GCP: a build trigger edited in the console to use an
inline config is outside it. It sees only files matching ``cloudbuild*.yaml``
or ``cloudbuild*.yml`` at the repo root. A build that pushes ``scienceq-*``
from anywhere else, such as a GitHub Actions job, is outside it too. The
``docker build`` in ``ci.yml`` is never pushed, so it does not matter here.

It also does not check that the variable still works. A future Docker could
rename or ignore it. #150 verified it end to end on ``regular``, and the check
to repeat after a worker change is the one recorded there: one version per
push, and the revision records that digest.


Parsing without PyYAML
----------------------

For the reason ``test_required_checks.py`` gives: the backend job installs only
the runtime requirements. Cloud Build steps have a small fixed shape — a list
of mappings whose values are scalars or lists, written block or flow style —
and that is all the parser below reads.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_GLOBS = ("cloudbuild*.yaml", "cloudbuild*.yml")

_VARIABLE = "BUILDX_NO_DEFAULT_ATTESTATIONS"
_REQUIRED = f"{_VARIABLE}=1"

# The Docker builder, by any registry path and tag: gcr.io/cloud-builders/docker,
# docker:24, and so on.
_DOCKER_IMAGE = re.compile(r"(?:^|/)docker(?::[\w.-]+)?$")


# ── Reading the configs ────────────────────────────────────────────────────────

@dataclass
class Step:
    file: str
    line: int
    keys: dict[str, str | list[str]] = field(default_factory=dict)
    text: str = ""

    def list(self, key: str) -> list[str]:
        value = self.keys.get(key)
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    @property
    def is_build(self) -> bool:
        """A Docker builder step whose args start ``build`` or ``buildx build``."""
        name = self.keys.get("name")
        if not isinstance(name, str) or not _DOCKER_IMAGE.search(name):
            return False
        if self.keys.get("entrypoint") not in (None, "docker"):
            return False
        args = self.list("args")
        return args[:1] == ["build"] or args[:2] == ["buildx", "build"]

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"


def _strip_comment(line: str) -> str:
    """YAML comment rule: `#` opens a comment at line start or after whitespace."""
    return re.split(r"(?:^|\s)#", line, maxsplit=1)[0].rstrip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _scalar_or_flow(value: str) -> str | list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_unquote(v) for v in inner.split(",")] if inner else []
    return _unquote(value)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _mapping(lines: list[tuple[int, str]], key_indent: int) -> dict[str, str | list[str]]:
    """Keys at ``key_indent``, each a scalar, a flow list, or a block list below it."""
    keys: dict[str, str | list[str]] = {}
    current = None
    for _, line in lines:
        indent, body = _indent(line), line.strip()
        if indent == key_indent and ":" in body and not body.startswith("- "):
            name, _, value = body.partition(":")
            current = name.strip()
            keys[current] = _scalar_or_flow(value) if value.strip() else []
        # YAML lets a block list sit at its key's own indent.
        elif indent >= key_indent and body.startswith("- ") and isinstance(keys.get(current), list):
            keys[current].append(_unquote(body[2:]))
    return keys


def _section(lines: list[tuple[int, str]], name: str) -> list[tuple[int, str]]:
    """The lines under a top-level key, up to the next top-level key."""
    out, inside = [], False
    for number, line in lines:
        # A `- ` at column 0 is a list item YAML allows unindented, not a key.
        if _indent(line) == 0 and not line.startswith("- "):
            inside = line.split(":", 1)[0].strip() == name
            continue
        if inside:
            out.append((number, line))
    return out


def _lines(path: Path) -> list[tuple[int, str]]:
    return [
        (number, stripped)
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if (stripped := _strip_comment(raw)).strip()
    ]


def _steps(path: Path) -> list[Step]:
    """The items of ``steps:``, each read as a mapping."""
    body = _section(_lines(path), "steps")
    if not body:
        return []
    item_indent = _indent(body[0][1])
    items: list[list[tuple[int, str]]] = []
    for number, line in body:
        if _indent(line) == item_indent and line.strip().startswith("- "):
            # Re-indent the item's first key so it lines up with the rest.
            items.append([(number, " " * (item_indent + 2) + line.strip()[2:])])
        elif items:
            items[-1].append((number, line))
    return [
        Step(
            file=path.name,
            line=item[0][0],
            keys=_mapping(item, item_indent + 2),
            text="\n".join(line for _, line in item),
        )
        for item in items
    ]


def _configs(root: Path) -> list[Path]:
    return sorted({p for glob in _BUILD_GLOBS for p in root.glob(glob)})


def _all_steps(root: Path) -> list[Step]:
    return [step for path in _configs(root) for step in _steps(path)]


# ── The checks ─────────────────────────────────────────────────────────────────
#
# Each returns a list of violations, empty when the check passes.

def _provenance_on(root: Path) -> list[str]:
    """Build steps whose own env does not turn provenance off."""
    problems = []
    for step in _all_steps(root):
        if not step.is_build:
            continue
        env = step.list("env")
        if _REQUIRED in env:
            continue
        found = [e for e in env if e.split("=", 1)[0] == _VARIABLE]
        problems.append(
            f"{step}: env has {found[0]!r}" if found
            else f"{step}: no {_REQUIRED} in the step's env"
        )
    return problems


def _unread_builds(root: Path) -> list[str]:
    """Steps that look like a Docker build in a shape this file does not read."""
    return [
        f"{step}: {step.keys.get('name')!r}"
        for step in _all_steps(root)
        if not step.is_build
        and re.search(r"\bdocker\b", step.text)
        and re.search(r"\bbuild\b", step.text)
    ]


# ── The guard ──────────────────────────────────────────────────────────────────

class TestEveryBuildIsProvenanceOff:

    def test_every_build_step_sets_the_variable(self):
        problems = _provenance_on(_REPO_ROOT)
        assert not problems, (
            "Cloud Build docker build steps with provenance on:\n"
            + "\n".join(f"  {p}" for p in problems)
            + f"\n\nOn the regular worker each push then stores 3 versions, and "
            "keep-5-most-recent protects 3 deploys instead of 5 (#149). Add "
            f"`env: ['{_REQUIRED}']` to the build step itself, not to "
            "options.env (#151)."
        )


class TestEveryBuildIsRead:

    def test_no_build_step_is_unreadable(self):
        unread = _unread_builds(_REPO_ROOT)
        assert not unread, (
            "Steps that look like a docker build but are not a shape this file "
            "reads:\n" + "\n".join(f"  {u}" for u in unread)
            + "\n\nThe provenance check skips these. Use the Docker builder with "
            "args starting `build`, or teach `Step.is_build` the new shape and "
            "add it to the parser-shapes test."
        )


# ── The checks fail on broken input ────────────────────────────────────────────

_ENV_BLOCK = "    env:\n      - 'BUILDX_NO_DEFAULT_ATTESTATIONS=1'\n"
_CONFIGS = ["cloudbuild-api.yaml", "cloudbuild-web.yaml", "cloudbuild-pipeline.yaml"]


def _config_copy(tmp_path: Path, file: str | None = None, edit=None) -> Path:
    """Copy the build configs into tmp_path, applying ``edit`` to one of them."""
    for path in _configs(_REPO_ROOT):
        text = path.read_text(encoding="utf-8")
        if path.name == file:
            changed = edit(text)
            assert changed != text, f"the edit changed nothing in {file}"
            text = changed
        (tmp_path / path.name).write_text(text, encoding="utf-8")
    return tmp_path


def _replace(old: str, new: str):
    def edit(text: str) -> str:
        assert old in text, f"{old!r} is gone; the case no longer breaks anything"
        return text.replace(old, new)
    return edit


def _moved_to_options(text: str) -> str:
    text = _replace(_ENV_BLOCK, "")(text)
    if "\noptions:\n" in text:
        return text.replace("\noptions:\n", f"\noptions:\n  env:\n    - '{_REQUIRED}'\n")
    return text + f"options:\n  env:\n    - '{_REQUIRED}'\n"


def _moved_to_push(text: str) -> str:
    text = _replace(_ENV_BLOCK, "")(text)
    if "      - push\n" not in text:
        # The pipeline config pushes through `images:`, so it has no push step.
        # Put the env on a new step that is not the build instead.
        step = "  - name: 'gcr.io/cloud-builders/docker'\n" + _ENV_BLOCK + "    args: [version]\n"
        return _replace("\nimages:\n", "\n" + step + "images:\n")(text)
    return _replace("    args:\n      - push\n", _ENV_BLOCK + "    args:\n      - push\n")(text)


_BROKEN = [
    # (id, edit, expected substring in a violation)
    ("removed", _replace(_ENV_BLOCK, ""), "no BUILDX_NO_DEFAULT_ATTESTATIONS=1"),
    ("set-to-0", _replace(f"{_REQUIRED}'", f"{_VARIABLE}=0'"), f"{_VARIABLE}=0"),
    ("commented-out", _replace(f"      - '{_REQUIRED}'", f"      # - '{_REQUIRED}'"),
     "no BUILDX_NO_DEFAULT_ATTESTATIONS=1"),
    ("moved-to-options-env", _moved_to_options, "no BUILDX_NO_DEFAULT_ATTESTATIONS=1"),
    ("moved-to-another-step", _moved_to_push, "no BUILDX_NO_DEFAULT_ATTESTATIONS=1"),
]


class TestChecksFailOnBrokenInput:

    @pytest.mark.parametrize("file", _CONFIGS)
    @pytest.mark.parametrize(
        "edit, expected",
        [case[1:] for case in _BROKEN],
        ids=[case[0] for case in _BROKEN],
    )
    def test_broken_config(self, tmp_path, file, edit, expected):
        real, broken = tmp_path / "real", tmp_path / "broken"
        real.mkdir()
        broken.mkdir()
        # The unmodified copy must pass, or the case proves nothing about the
        # edit — only that copying broke something.
        assert not _provenance_on(_config_copy(real)), "the real configs must pass first"
        problems = _provenance_on(_config_copy(broken, file, edit))
        assert any(p.startswith(file) and expected in p for p in problems), problems

    def test_a_bash_build_is_reported_unread(self, tmp_path):
        bash_step = (
            "  - name: 'gcr.io/cloud-builders/docker'\n"
            "    entrypoint: bash\n"
            "    args: ['-c', 'docker build -t x .']\n"
        )
        broken = _config_copy(
            tmp_path, "cloudbuild-web.yaml",
            # Inside `steps:`, which ends where `options:` starts.
            _replace("\noptions:\n", "\n" + bash_step + "options:\n"),
        )
        unread = _unread_builds(broken)
        assert any(u.startswith("cloudbuild-web.yaml") for u in unread), unread


# ── The guard's own seams ──────────────────────────────────────────────────────
#
# Every check above passes vacuously if the parser finds no build steps.

class TestGuardIsNotVacuous:

    def test_one_build_step_per_config(self):
        builds = {}
        for step in _all_steps(_REPO_ROOT):
            if step.is_build:
                builds[step.file] = builds.get(step.file, 0) + 1
        assert builds == {name: 1 for name in _CONFIGS}

    def test_the_other_steps_are_read_too(self):
        # api: build, 2 pushes, deploy, IAM. web: build, 2 pushes, deploy.
        # pipeline: build only. A parser that merged steps would miss this.
        counts = {path.name: len(_steps(path)) for path in _configs(_REPO_ROOT)}
        assert counts == {
            "cloudbuild-api.yaml": 5,
            "cloudbuild-pipeline.yaml": 1,
            "cloudbuild-web.yaml": 4,
        }

    def test_parser_handles_the_shapes_yaml_allows(self, tmp_path):
        (tmp_path / "cloudbuild-x.yaml").write_text(
            "steps:\n"
            "- name: gcr.io/cloud-builders/docker\n"
            "  env: ['BUILDX_NO_DEFAULT_ATTESTATIONS=1', \"OTHER=x\"]\n"
            "  args: [build, -t, img, .]\n"
            "- name: docker:24\n"
            "  args:\n"
            "  - buildx\n"
            "  - build\n"
            "  - .\n"
            "- name: 'gcr.io/cloud-builders/docker'  # a comment\n"
            "  args: [push, img]\n"
            "- name: gcr.io/cloud-builders/docker\n"
            "  entrypoint: bash\n"
            "  args: ['-c', 'echo hi']\n"
            "options:\n"
            "  env: ['BUILDX_NO_DEFAULT_ATTESTATIONS=1']\n",
            encoding="utf-8",
        )
        steps = _steps(tmp_path / "cloudbuild-x.yaml")
        assert [s.is_build for s in steps] == [True, True, False, False]
        assert steps[0].list("env") == [_REQUIRED, "OTHER=x"]
        assert steps[1].list("args") == ["buildx", "build", "."]
        assert steps[2].keys["name"] == "gcr.io/cloud-builders/docker"
        # Only the second build lacks a step-level env; options.env does not count.
        assert _provenance_on(tmp_path) == [
            "cloudbuild-x.yaml:5: no BUILDX_NO_DEFAULT_ATTESTATIONS=1 in the step's env",
        ]
        assert not _unread_builds(tmp_path)
