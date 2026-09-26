# tests/test_cleanup_policy.py

"""
The Artifact Registry cleanup policy must still protect what the build configs
push and what Cloud Run pulls (#145).

``cloud-run-source-deploy`` has deleted on a schedule since 2026-09-25 (#144):
every version older than 7 days goes, unless a keep rule holds it. The keep
rules hold images by **package-name prefix** and **tag prefix**, and Artifact
Registry does not check those strings against anything. A keep rule that stops
matching does not error. It stops keeping, and the next run deletes.

``infra/artifact-registry/cloud-run-source-deploy.json`` is the record of the
live policy, and this file is what holds it against the configs that name
images.


The case that matters most
--------------------------

The pipeline job has exactly one image and pulls it by tag,
``scienceq-pipeline:latest``. Each execution resolves the tag to a digest when
it starts (checked 2026-09-26: execution ``scienceq-pipeline-ztrdp`` recorded
``@sha256:866ef8a3…``, the digest ``latest`` pointed at), so what the next run
needs is whatever ``latest`` points at. ``keep-latest-tag`` is what keeps it.
Rename the tag in the build config, narrow or drop that rule, and the job's only
image becomes a deletion candidate with no error anywhere — until the next
scheduled run fails to pull.

The API and web services are deployed by ``$COMMIT_SHA``, a tag no two builds
share, so tag rules cannot protect them. ``keep-5-most-recent`` does: it keeps
the current image and four rollback targets per package.


What is checked
---------------

``TestEveryPushIsKept``
    Every package a Cloud Build config pushes is covered by a
    most-recent-versions keep rule. Tighter than "some keep rule": the
    ``latest`` rule has no package filter, so it covers every package and that
    check could not fail. Recency is what rollback depends on.

``TestEveryKeepPrefixMatches``
    Every package prefix on a keep rule matches at least one pushed package. A
    prefix that matches nothing has stopped applying: the other direction of the
    same coupling.

``TestEveryTagPullIsKept``
    Every image Cloud Run pulls by a fixed tag is held by a keep rule on that
    tag, and that tag is one a build config actually pushes.

``TestRollbackFloor``
    ``keepCount`` is at least 2, so an edit cannot remove every rollback target.

``TestEveryReferenceIsRead``
    Every mention of ``docker.pkg.dev`` in the configs is one the parser reads.
    The checks above see only what the parser collects, so a reference it
    skips would have none of them.

``TestOneRepository``
    Every Artifact Registry reference in the configs points at the repository
    the policy belongs to. A reference to another repository is outside this
    policy, and nothing here would say so.

Each check is a function that returns its violations, so the same function runs
against the real files and, in ``TestChecksFailOnBrokenInput``, against broken
copies of them. A check that has never failed has not been shown to check
anything.


Fail closed on what this file does not understand
-------------------------------------------------

A keep rule counts as protecting something only when every key on it is one this
file knows the meaning of. ``versionNamePrefixes``, ``olderThan`` or
``newerThan`` on a keep rule narrow it; so, likely, would a key the API adds
later. ``scripts/export_cleanup_policy.py`` copies ``condition`` wholesale, so
such a key reaches the record. Here it makes the rule stop counting, and the
test fails until someone teaches it the key.

The same rule applies to tags. A tag that is a Cloud Build per-build
substitution (``$COMMIT_SHA`` and the like) is unique per build, so recency
covers it. Any other ``$`` substitution could expand to anything, so it is
treated as a fixed tag, and it will not match a tag prefix.

And to references. The parser needs the region, project, repository and
package written literally, so ``$PROJECT_ID`` or ``${_REGION}`` in the path,
the usual Cloud Build idiom, is a shape it cannot read. Such a reference is
reported, not skipped: a package pushed that way would otherwise pass every
check without being checked.


What this test does not check
-----------------------------

It reads the committed JSON, not GCP. Someone editing the policy in the console
and not re-exporting leaves this file green and wrong. Re-export after any
change:

    python scripts/export_cleanup_policy.py

It also does not check the operating hazard of **pinned traffic**. A revision
serving pinned traffic is deployed by digest, and ``keep-5-most-recent`` keeps
it only while it is among the 5 newest images of its package. On 2026-07-25
traffic was pinned (see ``cloudbuild-api.yaml``). Five deploys after a pin, the
policy deletes the image production serves. Traffic lives in Cloud Run, not in
this repo, so the check is a manual one; it is written down in ``CLAUDE.md``.

It sees only references that spell out ``docker.pkg.dev``. A host that is
itself a substitution (``${_REGISTRY}/scienceq-x``) never does, so it passes
unread. An image moved to ``gcr.io`` or Docker Hub leaves this policy's scope
without this file noticing, and so does a config file whose name does not match
``cloudbuild-*.yaml`` or ``cloudrun-*.yaml``.


Parsing without PyYAML
----------------------

For the reason ``test_required_checks.py`` gives: the backend job installs only
the runtime requirements. Image references have a fixed textual shape, so they
are found by pattern after YAML comments are stripped. Whether a reference is a
pull depends on where it sits: a ``--image=`` flag or an ``image:`` key is a
pull, and anything else in a Cloud Build config (``-t``, ``push``, ``images:``)
is a push.
"""

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_POLICY = _REPO_ROOT / "infra" / "artifact-registry" / "cloud-run-source-deploy.json"

_BUILD_GLOB = "cloudbuild-*.yaml"
_RUN_GLOB = "cloudrun-*.yaml"

# Cloud Build substitutions that differ on every build. A tag built from one of
# these is never re-pointed, so recency, not a tag rule, is what keeps it.
_PER_BUILD = {"COMMIT_SHA", "SHORT_SHA", "REVISION_ID", "BUILD_ID"}

_KEEP_FLOOR = 2

# The keys whose meaning the checks below implement. See "Fail closed" above.
_KNOWN_CONDITION = {"tagState", "tagPrefixes", "packageNamePrefixes"}
_KNOWN_MOST_RECENT = {"keepCount", "packageNamePrefixes"}

_REF = re.compile(
    r"(?P<registry>[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9-]+/[a-z0-9-]+)"
    r"/(?P<package>[a-z0-9._-]+(?:/[a-z0-9._-]+)*)"
    r"(?::(?P<tag>[A-Za-z0-9_.${}-]+))?"
    r"(?:@(?P<digest>sha256:[0-9a-f]+))?"
    # Where a YAML scalar can end: whitespace, a quote, or a flow-list `,`/`]`.
    r"(?=$|[\s'\",\]])"
)
_PULL_CONTEXT = re.compile(r"--image=|(?:^|\s|-)image:\s")

# Any scalar that names the registry host, readable or not. See "Fail closed".
_REGISTRY_HOST = "docker.pkg.dev"
_REGISTRY_TOKEN = re.compile(r"[^\s'\",\[\]]*docker\.pkg\.dev[^\s'\",\[\]]*")


# ── Reading the configs ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ref:
    file: str
    line: int
    registry: str
    package: str
    tag: str | None
    digest: str | None
    pulled: bool

    @property
    def per_build(self) -> bool:
        """Pinned by digest or by a substitution unique to each build."""
        if self.digest:
            return True
        name = re.fullmatch(r"\$\{?([A-Z_]+)\}?", self.tag or "")
        return bool(name and name.group(1) in _PER_BUILD)

    def __str__(self) -> str:
        where = f":{self.tag}" if self.tag else f"@{self.digest}"
        return f"{self.package}{where}  ({self.file}:{self.line})"


def _strip_comment(line: str) -> str:
    """YAML comment rule: `#` opens a comment at line start or after whitespace."""
    return re.split(r"(?:^|\s)#", line, maxsplit=1)[0]


def _config_lines(root: Path):
    """(path, line number, line without its comment) for every build and run config."""
    for path in sorted(root.glob(_BUILD_GLOB)) + sorted(root.glob(_RUN_GLOB)):
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            yield path, number, _strip_comment(raw)


def _refs(root: Path) -> list[Ref]:
    """Every Artifact Registry image reference in the build and run configs."""
    refs: list[Ref] = []
    for path, number, line in _config_lines(root):
        for match in _REF.finditer(line):
            tag, digest = match.group("tag"), match.group("digest")
            refs.append(Ref(
                file=path.name,
                line=number,
                registry=match.group("registry"),
                package=match.group("package"),
                # No tag and no digest means Docker's implicit `latest`.
                tag=tag if tag or digest else "latest",
                digest=digest,
                # A run config names images only to pull them.
                pulled=path.match(_RUN_GLOB)
                or bool(_PULL_CONTEXT.search(line[: match.start()])),
            ))
    return refs


def _unparsed(root: Path) -> list[str]:
    """Every ``docker.pkg.dev`` occurrence that no ``_REF`` match covers.

    Checked per occurrence, not per line, so a readable reference cannot hide an
    unreadable one beside it in a flow list.
    """
    unparsed = []
    for path, number, line in _config_lines(root):
        spans = [match.span() for match in _REF.finditer(line)]
        for token in _REGISTRY_TOKEN.finditer(line):
            host = token.start() + token.group().index(_REGISTRY_HOST)
            if not any(start <= host < end for start, end in spans):
                unparsed.append(f"{path.name}:{number}: {token.group()}")
    return unparsed


def _pushed(refs: list[Ref]) -> set[str]:
    return {ref.package for ref in refs if not ref.pulled}


def _load_policy() -> dict:
    return json.loads(_POLICY.read_text(encoding="utf-8"))


# ── Reading the policy ─────────────────────────────────────────────────────────

def _keep_rules(policy: dict) -> list[dict]:
    return [
        rule for rule in policy.get("cleanupPolicies", [])
        if str(rule.get("action", "")).upper() == "KEEP"
    ]


def _matches_package(prefixes: list[str] | None, package: str) -> bool:
    # A rule without a package filter applies to every package.
    return prefixes is None or any(package.startswith(p) for p in prefixes)


def _recency_rules(policy: dict) -> list[dict]:
    """Keep rules that keep the N most recent versions, and nothing narrower."""
    return [
        rule for rule in _keep_rules(policy)
        if "mostRecentVersions" in rule
        and set(rule["mostRecentVersions"]) <= _KNOWN_MOST_RECENT
    ]


def _keeps_tag(rule: dict, package: str, tag: str) -> bool:
    """Whether a keep rule holds every version of ``package`` tagged ``tag``."""
    condition = rule.get("condition")
    if condition is None or not set(condition) <= _KNOWN_CONDITION:
        return False
    if str(condition.get("tagState", "")).upper() not in {"TAGGED", "ANY"}:
        return False
    prefixes = condition.get("tagPrefixes")
    if prefixes is not None and not any(tag.startswith(p) for p in prefixes):
        return False
    return _matches_package(condition.get("packageNamePrefixes"), package)


# ── The checks ─────────────────────────────────────────────────────────────────
#
# Each returns a list of violations, empty when the check passes.

def _unkept_pushes(policy: dict, refs: list[Ref]) -> list[str]:
    rules = _recency_rules(policy)
    return sorted(
        package for package in _pushed(refs)
        if not any(
            _matches_package(r["mostRecentVersions"].get("packageNamePrefixes"), package)
            for r in rules
        )
    )


def _dead_prefixes(policy: dict, refs: list[Ref]) -> list[str]:
    pushed = _pushed(refs)
    dead = []
    for rule in _keep_rules(policy):
        for block in ("condition", "mostRecentVersions"):
            for prefix in rule.get(block, {}).get("packageNamePrefixes") or []:
                if not any(package.startswith(prefix) for package in pushed):
                    dead.append(f"{rule.get('id')}: {prefix!r}")
    return dead


def _unkept_tag_pulls(policy: dict, refs: list[Ref]) -> list[str]:
    pushed_tags = {(r.package, r.tag) for r in refs if not r.pulled}
    problems = []
    for ref in refs:
        if not ref.pulled or ref.per_build:
            continue
        if (ref.package, ref.tag) not in pushed_tags:
            problems.append(f"{ref}: no build config pushes this tag")
        elif not any(_keeps_tag(rule, ref.package, ref.tag) for rule in _keep_rules(policy)):
            problems.append(f"{ref}: no keep rule holds this tag")
    return problems


def _unbuilt_pulls(refs: list[Ref]) -> list[str]:
    pushed = _pushed(refs)
    return [str(ref) for ref in refs if ref.pulled and ref.package not in pushed]


def _low_keep_counts(policy: dict) -> list[str]:
    return [
        f"{rule.get('id')}: keepCount {rule['mostRecentVersions'].get('keepCount')}"
        for rule in _keep_rules(policy)
        if "mostRecentVersions" in rule
        and (rule["mostRecentVersions"].get("keepCount") or 0) < _KEEP_FLOOR
    ]


def _foreign_refs(policy: dict, refs: list[Ref]) -> list[str]:
    return [
        f"{ref}: in {ref.registry}"
        for ref in refs
        if ref.registry != policy.get("registryUri")
    ]


# ── The guard ──────────────────────────────────────────────────────────────────

_REEXPORT = (
    "If the policy was changed in the console, re-export it with "
    "`python scripts/export_cleanup_policy.py`; if a config changed, change "
    "the live policy to match and then re-export. The file is the record, not "
    "the enforcement."
)


class TestEveryPushIsKept:

    def test_every_pushed_package_has_a_recency_rule(self):
        unkept = _unkept_pushes(_load_policy(), _refs(_REPO_ROOT))
        assert not unkept, (
            "Packages pushed by a Cloud Build config that no most-recent-versions "
            "keep rule covers:\n" + "\n".join(f"  {p}" for p in unkept)
            + "\n\nEvery version of these older than 7 days is deleted, the one "
            "serving included, and there is nothing to roll back to. "
            + _REEXPORT
        )


class TestEveryKeepPrefixMatches:

    def test_every_keep_prefix_matches_a_pushed_package(self):
        dead = _dead_prefixes(_load_policy(), _refs(_REPO_ROOT))
        assert not dead, (
            "Keep-rule package prefixes that match no pushed package:\n"
            + "\n".join(f"  {d}" for d in dead)
            + "\n\nA prefix matching nothing keeps nothing. It is usually the "
            "trace of a rename on one side only. " + _REEXPORT
        )


class TestEveryTagPullIsKept:

    def test_every_image_pulled_by_tag_is_held_by_a_tag_rule(self):
        refs = _refs(_REPO_ROOT)
        problems = _unkept_tag_pulls(_load_policy(), refs) + [
            f"{p}: pulled, but no build config pushes this package"
            for p in _unbuilt_pulls(refs)
        ]
        assert not problems, (
            "Images Cloud Run pulls by tag that the cleanup policy does not "
            "hold:\n" + "\n".join(f"  {p}" for p in problems)
            + "\n\nThe pipeline job has one image and pulls it by tag. Once no "
            "keep rule holds that tag, the image is deleted 7 days after it was "
            "pushed and the next job run fails to pull. " + _REEXPORT
        )


class TestRollbackFloor:

    def test_keep_count_is_at_least_the_floor(self):
        low = _low_keep_counts(_load_policy())
        assert not low, (
            f"Most-recent-versions rules below the floor of {_KEEP_FLOOR}:\n"
            + "\n".join(f"  {entry}" for entry in low)
            + "\n\nWith keepCount 1 the serving image is the only one kept and "
            "a bad deploy has no previous image to roll back to; with 0 the rule "
            "keeps nothing."
        )


class TestEveryReferenceIsRead:

    def test_every_registry_reference_is_parsed(self):
        unparsed = _unparsed(_REPO_ROOT)
        assert not unparsed, (
            "Artifact Registry references this file cannot parse:\n"
            + "\n".join(f"  {u}" for u in unparsed)
            + "\n\nEvery check above skips these, so a package pushed this way "
            "has no coverage check at all. Write the registry path literally, or "
            "teach `_REF` the new shape and add it to the parser-shapes test."
        )


class TestOneRepository:

    def test_every_reference_is_in_the_policed_repository(self):
        policy = _load_policy()
        foreign = _foreign_refs(policy, _refs(_REPO_ROOT))
        assert not foreign, (
            f"Image references outside {policy.get('registryUri')}:\n"
            + "\n".join(f"  {f}" for f in foreign)
            + "\n\nThis policy does not apply there, and none of the checks "
            "above can say what does."
        )


# ── The checks fail on broken input ────────────────────────────────────────────
#
# Each case breaks one thing, in the policy or in a copy of the configs, and
# asserts the check meant to catch it reports it. The real files are the
# starting point, so the parser is exercised on the shapes the repo uses.

def _config_copy(tmp_path: Path, replace: dict[str, tuple[str, str]] | None = None) -> Path:
    """Copy the build and run configs into tmp_path, applying text replacements."""
    replace = replace or {}
    for path in list(_REPO_ROOT.glob(_BUILD_GLOB)) + list(_REPO_ROOT.glob(_RUN_GLOB)):
        text = path.read_text(encoding="utf-8")
        if path.name in replace:
            old, new = replace[path.name]
            assert old in text, f"{old!r} not in {path.name}; the case no longer breaks anything"
            text = text.replace(old, new)
        (tmp_path / path.name).write_text(text, encoding="utf-8")
    return tmp_path


def _rule(policy: dict, rule_id: str) -> dict:
    return next(r for r in policy["cleanupPolicies"] if r["id"] == rule_id)


def _without_latest_rule(policy):
    policy["cleanupPolicies"] = [
        r for r in policy["cleanupPolicies"] if r["id"] != "keep-latest-tag"
    ]


def _latest_rule_renamed(policy):
    _rule(policy, "keep-latest-tag")["condition"]["tagPrefixes"] = ["stable"]


def _latest_rule_untagged(policy):
    _rule(policy, "keep-latest-tag")["condition"]["tagState"] = "UNTAGGED"


def _latest_rule_narrowed(policy):
    # A key this file does not implement must make the rule stop counting.
    _rule(policy, "keep-latest-tag")["condition"]["versionNamePrefixes"] = ["sha256:0"]


def _latest_rule_scoped_away(policy):
    _rule(policy, "keep-latest-tag")["condition"]["packageNamePrefixes"] = ["scienceq-api"]


def _recency_prefix_changed(policy):
    _rule(policy, "keep-5-most-recent")["mostRecentVersions"]["packageNamePrefixes"] = ["sciq-"]


def _recency_rule_dropped(policy):
    policy["cleanupPolicies"] = [
        r for r in policy["cleanupPolicies"] if r["id"] != "keep-5-most-recent"
    ]


def _keep_count(n):
    def mutate(policy):
        _rule(policy, "keep-5-most-recent")["mostRecentVersions"]["keepCount"] = n
    return mutate


_POLICY_CASES = [
    # (id, mutation, check, expected substring in a violation)
    ("latest-rule-dropped", _without_latest_rule, "tag", "scienceq-pipeline:latest"),
    ("latest-rule-renamed", _latest_rule_renamed, "tag", "scienceq-pipeline:latest"),
    ("latest-rule-untagged", _latest_rule_untagged, "tag", "scienceq-pipeline:latest"),
    ("latest-rule-narrowed", _latest_rule_narrowed, "tag", "scienceq-pipeline:latest"),
    ("latest-rule-scoped-away", _latest_rule_scoped_away, "tag", "scienceq-pipeline:latest"),
    ("recency-prefix-changed", _recency_prefix_changed, "push", "scienceq-api"),
    ("recency-prefix-changed", _recency_prefix_changed, "prefix", "'sciq-'"),
    ("recency-rule-dropped", _recency_rule_dropped, "push", "scienceq-web"),
    ("keep-count-0", _keep_count(0), "floor", "keepCount 0"),
    ("keep-count-1", _keep_count(1), "floor", "keepCount 1"),
]

_CONFIG_CASES = [
    # (id, {file: (old, new)}, check, expected substring)
    (
        "api-package-renamed",
        {"cloudbuild-api.yaml": ("/scienceq-api:", "/sq-api:")},
        "push", "sq-api",
    ),
    (
        "pipeline-tag-renamed-in-build",
        {"cloudbuild-pipeline.yaml": ("scienceq-pipeline:latest", "scienceq-pipeline:prod")},
        "tag", "no build config pushes this tag",
    ),
    (
        "pipeline-job-pulls-new-tag",
        {"cloudrun-pipeline-job.yaml": ("scienceq-pipeline:latest", "scienceq-pipeline:prod")},
        "tag", "scienceq-pipeline:prod",
    ),
    (
        "pipeline-renamed-on-both-sides",
        # Build and job agree, so the tag is pushed. Only the policy is wrong.
        {
            "cloudbuild-pipeline.yaml": ("scienceq-pipeline:latest", "scienceq-pipeline:prod"),
            "cloudrun-pipeline-job.yaml": ("scienceq-pipeline:latest", "scienceq-pipeline:prod"),
        },
        "tag", "no keep rule holds this tag",
    ),
    (
        "repository-moved",
        {"cloudbuild-web.yaml": ("/cloud-run-source-deploy/scienceq-web:$COMMIT_SHA\n",
                                 "/other-repo/scienceq-web:$COMMIT_SHA\n")},
        "repo", "other-repo",
    ),
]


_UNREADABLE_CASES = [
    # (id, {file: (old, new)}, expected substring). A substitution in each
    # segment of the path, each of which `_REF` requires to be literal.
    (
        "region-substituted",
        {"cloudbuild-api.yaml": ("europe-west1-docker.pkg.dev/", "${_REGION}-docker.pkg.dev/")},
        "${_REGION}-docker.pkg.dev",
    ),
    (
        "project-substituted",
        {"cloudbuild-web.yaml": ("/scienceq-prod/", "/$PROJECT_ID/")},
        "$PROJECT_ID",
    ),
    (
        "repository-substituted",
        {"cloudrun-pipeline-job.yaml": ("/cloud-run-source-deploy/", "/${_REPO}/")},
        "${_REPO}",
    ),
    (
        "package-substituted",
        {"cloudbuild-pipeline.yaml": ("/scienceq-pipeline:", "/${_SERVICE}:")},
        "${_SERVICE}",
    ),
]


def _run(check: str, policy: dict, refs: list[Ref]) -> list[str]:
    return {
        "push": lambda: _unkept_pushes(policy, refs),
        "prefix": lambda: _dead_prefixes(policy, refs),
        "tag": lambda: _unkept_tag_pulls(policy, refs),
        "floor": lambda: _low_keep_counts(policy),
        "repo": lambda: _foreign_refs(policy, refs),
    }[check]()


class TestChecksFailOnBrokenInput:

    @pytest.mark.parametrize(
        "mutate, check, expected",
        [case[1:] for case in _POLICY_CASES],
        ids=[f"{case[0]}->{case[2]}" for case in _POLICY_CASES],
    )
    def test_broken_policy(self, mutate, check, expected):
        policy = copy.deepcopy(_load_policy())
        refs = _refs(_REPO_ROOT)
        assert not _run(check, policy, refs), "the real policy must pass first"
        mutate(policy)
        violations = _run(check, policy, refs)
        assert any(expected in v for v in violations), violations

    @pytest.mark.parametrize(
        "replace, check, expected",
        [case[1:] for case in _CONFIG_CASES],
        ids=[f"{case[0]}->{case[2]}" for case in _CONFIG_CASES],
    )
    def test_broken_configs(self, tmp_path, replace, check, expected):
        policy = _load_policy()
        real, broken = tmp_path / "real", tmp_path / "broken"
        real.mkdir()
        broken.mkdir()
        # The unmodified copy must pass, or the case proves nothing about the
        # replacement — only that copying broke something.
        assert not _run(check, policy, _refs(_config_copy(real))), "the real configs must pass first"
        violations = _run(check, policy, _refs(_config_copy(broken, replace)))
        assert any(expected in v for v in violations), violations

    @pytest.mark.parametrize(
        "replace, expected",
        [case[1:] for case in _UNREADABLE_CASES],
        ids=[case[0] for case in _UNREADABLE_CASES],
    )
    def test_unreadable_references(self, tmp_path, replace, expected):
        real, broken = tmp_path / "real", tmp_path / "broken"
        real.mkdir()
        broken.mkdir()
        assert not _unparsed(_config_copy(real)), "the real configs must pass first"
        unparsed = _unparsed(_config_copy(broken, replace))
        assert any(expected in u for u in unparsed), unparsed


# ── The guard's own seams ──────────────────────────────────────────────────────
#
# Every check above passes vacuously if the parser returns nothing. These make
# that impossible.

class TestGuardIsNotVacuous:

    def test_the_record_exists_and_holds_three_rules(self):
        policy = _load_policy()
        assert [r["id"] for r in policy["cleanupPolicies"]] == [
            "delete-older-than-7d", "keep-5-most-recent", "keep-latest-tag",
        ]

    def test_the_policy_is_recorded_as_live(self):
        # Written explicitly by the export script; the API omits it when false.
        assert _load_policy()["cleanupPolicyDryRun"] is False

    def test_pushed_packages_are_collected(self):
        # A floor, not an inventory: a new package is fine, so long as a
        # recency rule covers it, which is the check above.
        assert _pushed(_refs(_REPO_ROOT)) >= {
            "scienceq-api", "scienceq-web", "scienceq-pipeline",
        }

    def test_the_pipeline_tag_pull_is_collected(self):
        tag_pulls = {
            (r.package, r.tag) for r in _refs(_REPO_ROOT) if r.pulled and not r.per_build
        }
        assert ("scienceq-pipeline", "latest") in tag_pulls

    def test_service_deploys_are_collected_as_per_build_pulls(self):
        per_build = {r.package for r in _refs(_REPO_ROOT) if r.pulled and r.per_build}
        assert per_build >= {"scienceq-api", "scienceq-web"}

    def test_a_recency_rule_is_recognised(self):
        assert len(_recency_rules(_load_policy())) == 1

    def test_parser_handles_the_shapes_yaml_allows(self, tmp_path):
        registry = "europe-west1-docker.pkg.dev/p/r"
        (tmp_path / "cloudbuild-x.yaml").write_text(
            "steps:\n"
            f"  - args: [build, -t, {registry}/built:$COMMIT_SHA, .]\n"
            f"  - args: [push, '{registry}/quoted:${{SHORT_SHA}}']\n"
            f"      - --image={registry}/built:$COMMIT_SHA\n"
            f"  # - --image={registry}/commented:latest\n"
            f"      - --image={registry}/built  # no tag: Docker's latest\n"
            "images:\n"
            f"  - \"{registry}/built@sha256:abc123\"\n",
            encoding="utf-8",
        )
        (tmp_path / "cloudrun-x.yaml").write_text(
            f"          - image: {registry}/job:$_CHANNEL\n",
            encoding="utf-8",
        )
        got = {(r.package, r.tag, r.digest, r.pulled, r.per_build) for r in _refs(tmp_path)}
        assert got == {
            ("built", "$COMMIT_SHA", None, False, True),
            ("quoted", "${SHORT_SHA}", None, False, True),
            ("built", "$COMMIT_SHA", None, True, True),
            ("built", "latest", None, True, False),
            ("built", None, "sha256:abc123", False, True),
            # A user substitution could expand to anything, so it is a fixed
            # tag, and one no tag prefix will match.
            ("job", "$_CHANNEL", None, True, False),
        }
        # Every shape above is readable, so none of it may be reported.
        assert not _unparsed(tmp_path)

    def test_an_unreadable_reference_is_reported_per_occurrence(self, tmp_path):
        literal = "europe-west1-docker.pkg.dev/p/r"
        (tmp_path / "cloudbuild-x.yaml").write_text(
            "steps:\n"
            # One readable and one unreadable reference in the same flow list:
            # the readable one must not hide its neighbour.
            f"  - args: [build, -t, {literal}/a:$COMMIT_SHA,"
            " -t, europe-west1-docker.pkg.dev/$PROJECT_ID/r/b:latest, .]\n"
            "  # - --image=${_REGION}-docker.pkg.dev/p/r/commented:latest\n"
            f"      - --image={literal}/a  # ${{_REGION}}-docker.pkg.dev/p/r/c\n",
            encoding="utf-8",
        )
        assert _unparsed(tmp_path) == [
            "cloudbuild-x.yaml:2: europe-west1-docker.pkg.dev/$PROJECT_ID/r/b:latest",
        ]
