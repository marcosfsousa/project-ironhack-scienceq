# tests/test_required_checks.py

"""
The required status checks on ``main`` must name jobs that exist (#90).

``main`` had no protection of any kind until this issue: no protected-branch
config, no rulesets. Merging to it fires the ``deploy-scienceq-api`` and
``deploy-scienceq-web`` Cloud Build triggers, so anything reaching ``main``
reaches production, and nothing required a check to pass, a review to happen,
or a PR to exist. The practice was already right — the last twenty first-parent
commits are all ``Merge pull request #NN`` — but convention was all it was.

``.github/rulesets/main.json`` is what holds that in place, and this file is
what stops it from silently ceasing to.


Why a test and not a comment
----------------------------

Required checks are matched by **string**. GitHub does not verify that a
required context corresponds to anything: a context naming a job that no longer
exists simply never reports, and a rule waiting on a check that never arrives is
indistinguishable from one that has not run yet. The PR sits pending rather than
failing, which reads as "still working" instead of "misconfigured".

``.github/workflows/ci.yml`` has warned about exactly this in its own header
since it was written — job ``name:`` values are the status-check identifiers,
treat them as a stable contract. The warning was moot while nothing pointed at
those names, and it stopped being moot the moment the ruleset did.

It is also not hypothetical. #86 renamed ``Frontend build (tsc + vite)`` to
``Frontend typecheck + build``; both spellings are visible in this repo's Actions
history, either side of ``f7377f9``. That rename was harmless because nothing
referenced the old name yet. The next one would not be, and the failure would be
invisible at exactly the moment it mattered — at merge, on the PR that renamed
it, with a green tick and a rule that no longer applies.

So the coupling is checked from both ends:

``TestEveryRequiredCheckExists``
    A context in the ruleset with no matching job. This is the rename, and the
    typo.

``TestEveryJobIsRequired``
    A job in ``ci.yml`` that no context names. This is #87/#88: ``API image
    (build + import)`` was added by #87 to protect the narrowed ``pipeline/``
    COPY that #88 then made, and for one PR it existed as a job while nothing
    required it. A new seam that is not required is a seam that does not gate.

Together they are an equality, deliberately. Adding a job to ``ci.yml`` without
requiring it is a decision, and it should cost a line in the ruleset rather than
happening by omission.


What this test does not check
-----------------------------

It reads the committed JSON, not GitHub. Nothing here can prove the ruleset is
actually applied, or applied with this content — the API is the authority and it
needs a token this suite does not have. Someone editing the ruleset in the web
UI and not re-exporting it leaves this file green and wrong.

That is a smaller hole than the one it closes, and it points the right way: the
committed file is the reviewable record, so a change made only in the UI is a
change made outside review. Re-export after any UI edit:

    python scripts/export_ruleset.py

That script drops the fields GitHub assigns rather than accepts, so the file it
writes is both an accurate record and a body you can POST to recreate the
ruleset from scratch.


Parsing ci.yml without PyYAML
-----------------------------

The backend job installs ``requirements.txt`` plus ``pytest`` and never
``requirements-dev.txt``, so a test that imports ``yaml`` fails in CI unless
PyYAML lands in the deploy manifest — a dependency in the serving image for the
benefit of a test. Not worth it, and ``tests/test_declared_imports.py`` would
correctly fail it. The job header is a fixed, shallow shape; it is parsed by
indentation here the same way that file parses Dockerfile's COPY block.

Two details the shape actually depends on:

**A job's check context is its ``name:``, or its job id when it has none.**
``backend`` would report as ``backend`` if line 38 were deleted. The parser
falls back the same way, so a job that loses its ``name:`` fails as a rename
rather than vanishing from the comparison.

**``#`` opens a comment only at the start of a line or after whitespace**, per
YAML, so a value is truncated on that pattern and not on every ``#``. Same rule
``_declared_distributions`` applies to requirements files, for the same reason:
splitting on every ``#`` corrupts values that legitimately contain one.
"""

import json
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent

_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_RULESET = _REPO_ROOT / ".github" / "rulesets" / "main.json"

# The branch Cloud Build's deploy triggers watch. Written as a literal ref
# rather than GitHub's `~DEFAULT_BRANCH` alias on purpose: the triggers match a
# branch pattern in GCP, not "whatever this repo calls its default branch", so
# following the default would silently decouple protection from deployment if
# the default were ever moved.
#
# The live trigger config is in the GCP project and cannot be read from here.
# scripts/create-triggers.sh is the next best thing — it is what creates both
# triggers, so it states the branch this repo intends to deploy from, and
# TestRulesetPinsItsDecisions checks the two against each other. It is a
# statement of intent rather than the live value: a trigger edited in the
# console diverges from the script with nothing to notice. Worth having anyway,
# since the pair moving apart in the repo is the likelier mistake.
_PROTECTED_REF = "refs/heads/main"
_TRIGGERS = _REPO_ROOT / "scripts" / "create-triggers.sh"


# ── Reading the workflow ───────────────────────────────────────────────────────

def _strip_comment(value: str) -> str:
    """YAML comment rule: `#` opens a comment at line start or after whitespace."""
    return re.split(r"(?:^|\s)#", value, maxsplit=1)[0].strip()


def _job_contexts(workflow: Path) -> dict[str, str]:
    """
    ``{job id: status-check context}`` for every job in a workflow.

    The context is the job's ``name:`` where it has one and its id otherwise,
    which is what GitHub reports. Keyed by id so a failure can name the job
    whose ``name:`` moved rather than only the string that disappeared.

    Indentation-based: job ids sit at two spaces under ``jobs:`` and a job-level
    ``name:`` at four. Step names are ``- name:`` at six and are therefore
    excluded structurally, not by pattern — see
    ``test_step_names_are_not_collected``.
    """
    contexts: dict[str, str] = {}
    lines = workflow.read_text(encoding="utf-8").splitlines()

    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == "jobs:")
    except StopIteration:  # pragma: no cover - asserted directly below
        return contexts

    current: str | None = None
    for line in lines[start + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # A non-indented key ends the jobs block.
        if not line.startswith(" "):
            break
        job = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if job:
            current = job.group(1)
            contexts[current] = current  # id is the context until a name says otherwise
            continue
        name = re.match(r"^    name:\s*(.+)$", line)
        if name and current:
            contexts[current] = _strip_comment(name.group(1)).strip("'\"")
    return contexts


# ── Reading the ruleset ────────────────────────────────────────────────────────

def _rule(ruleset: dict, rule_type: str) -> dict | None:
    return next((r for r in ruleset.get("rules", []) if r.get("type") == rule_type), None)


def _required_contexts(ruleset: dict) -> set[str]:
    rule = _rule(ruleset, "required_status_checks") or {}
    checks = rule.get("parameters", {}).get("required_status_checks", [])
    return {check["context"] for check in checks}


def _load_ruleset() -> dict:
    return json.loads(_RULESET.read_text(encoding="utf-8"))


# ── The guard ──────────────────────────────────────────────────────────────────

class TestEveryRequiredCheckExists:
    """The rename (#86): a required context naming a job that is not there."""

    def test_every_required_context_is_a_job(self):
        contexts = set(_job_contexts(_WORKFLOW).values())
        missing = sorted(_required_contexts(_load_ruleset()) - contexts)
        assert not missing, (
            "Required status checks that no job in .github/workflows/ci.yml "
            "reports:\n" + "\n".join(f"  {name}" for name in missing)
            + f"\n\nJobs currently report: {sorted(contexts)}\n\n"
            "A required check that never arrives leaves the PR pending, not "
            "red — the rule reads as protection while applying to nothing. If "
            "a job was renamed, rename the context in "
            ".github/rulesets/main.json to match and re-apply the ruleset; the "
            "committed file is not what GitHub enforces until it is POSTed."
        )


class TestEveryJobIsRequired:
    """The gap #87/#88 opened: a seam that exists as a job but gates nothing."""

    def test_every_job_is_a_required_context(self):
        contexts = _job_contexts(_WORKFLOW)
        required = _required_contexts(_load_ruleset())
        unrequired = sorted(
            f"{name}  (job: {job})"
            for job, name in contexts.items()
            if name not in required
        )
        assert not unrequired, (
            "Jobs in .github/workflows/ci.yml that no ruleset context "
            "requires:\n" + "\n".join(f"  {entry}" for entry in unrequired)
            + "\n\nThe job runs and can go red without blocking a merge, which "
            "is the state #87's api-image job was in on #88 — the gate built to "
            "protect the narrowed COPY did not gate the PR that narrowed it. "
            "Add the context to .github/rulesets/main.json and re-apply, or, if "
            "the job is deliberately advisory, say so here and in that file "
            "rather than leaving it to omission."
        )


class TestRulesetPinsItsDecisions:
    """
    The ruleset file is re-exported from GitHub after any UI edit, so the
    decisions #90 settled can be dropped by a click and arrive here as a diff
    nobody reads closely. These are the ones worth failing on.
    """

    def test_it_targets_the_branch_that_deploys(self):
        ruleset = _load_ruleset()
        include = ruleset.get("conditions", {}).get("ref_name", {}).get("include", [])
        assert include == [_PROTECTED_REF], (
            f"The ruleset targets {include}, not [{_PROTECTED_REF!r}].\n"
            "main is the branch the Cloud Build triggers watch, so it is the "
            "branch where a merge is a deploy. Widening this is fine; narrowing "
            "or moving it leaves production unprotected."
        )

    def test_the_deploy_trigger_watches_the_branch_it_protects(self):
        # The other end of the same coupling. If a trigger is ever pointed at a
        # different branch, the ruleset does not follow it — production would
        # deploy from a branch anyone can push to, and every check here would
        # still be green.
        script = _TRIGGERS.read_text(encoding="utf-8")
        branch = re.search(r'^BRANCH="\^([A-Za-z0-9._/-]+)\$"', script, re.MULTILINE)
        assert branch, (
            f"No BRANCH=\"^...$\" assignment in {_TRIGGERS.name}. The deploy "
            "branch can no longer be read from the repo, so nothing checks that "
            "the protected branch is the one that ships."
        )
        assert f"refs/heads/{branch.group(1)}" == _PROTECTED_REF, (
            f"{_TRIGGERS.name} deploys from {branch.group(1)!r} but the ruleset "
            f"protects {_PROTECTED_REF!r}.\n"
            "Whichever moved, the other has to follow: a deploy branch without "
            "the ruleset is an unprotected production branch."
        )

    def test_the_admin_bypass_is_recorded(self):
        actors = _load_ruleset().get("bypass_actors", [])
        assert len(actors) == 1, (
            f"Expected one bypass actor, found {len(actors)}: {actors}\n"
            "With none, the only way past a stuck required check is to delete "
            "or disable the ruleset — a change that leaves protection off after "
            "the emergency instead of leaving a bypass in the log. With more "
            "than one, say here who else may skip the checks and why."
        )
        actor = actors[0]
        # 5 is GitHub's built-in Repository admin role. Read back from the API
        # after adding the bypass through the UI rather than assumed: the role
        # ids are not documented, and a wrong one here is a silent widening —
        # 2 is Write, which on this repo would be the same person and would look
        # identical in a diff.
        assert (actor.get("actor_type"), actor.get("actor_id")) == ("RepositoryRole", 5), (
            f"The bypass actor is {actor}, not the Repository admin role.\n"
            "Re-run scripts/export_ruleset.py if this changed in the UI, and "
            "check what role was actually granted before committing it."
        )
        assert actor.get("bypass_mode") == "always"

    def test_branches_must_be_up_to_date(self):
        rule = _rule(_load_ruleset(), "required_status_checks") or {}
        assert rule.get("parameters", {}).get("strict_required_status_checks_policy") is True, (
            "strict_required_status_checks_policy is not true.\n"
            "Without it a PR merges on checks that ran against an older main. "
            "That is #88 exactly: branched before #87 added the api-image job, "
            "so its first CI run had three jobs and the missing one could not "
            "fail. It was caught by hand. Strict mode is what makes the catch "
            "mechanical."
        )

    def test_force_push_and_deletion_are_blocked(self):
        ruleset = _load_ruleset()
        for rule_type in ("non_fast_forward", "deletion"):
            assert _rule(ruleset, rule_type), (
                f"The {rule_type} rule is absent.\n"
                "PR branches here are rebased and force-pushed routinely and "
                "this does not affect them — the ruleset applies to main alone. "
                "Loosening this is not the fix for friction on a feature branch."
            )

    def test_a_pull_request_is_required(self):
        rule = _rule(_load_ruleset(), "pull_request")
        assert rule, (
            "The pull_request rule is absent, so a direct push to main is "
            "accepted — and deploys."
        )
        # Zero approvals is not an oversight. This repo has one collaborator and
        # GitHub does not let an author approve their own PR, so any positive
        # count means no human PR can ever merge and every Dependabot PR needs a
        # bypass. The rule is here for the PR requirement itself: the diff, the
        # checks, and the record.
        assert rule.get("parameters", {}).get("required_approving_review_count") == 0


# ── The guard's own seams ──────────────────────────────────────────────────────
#
# Both assertions above pass vacuously if the parser returns nothing, and the
# equality passes if it returns nothing on both sides. These make that
# impossible.

class TestGuardIsNotVacuous:

    def test_both_files_exist(self):
        assert _WORKFLOW.is_file(), f"{_WORKFLOW} is missing"
        assert _RULESET.is_file(), f"{_RULESET} is missing"

    def test_jobs_are_collected(self):
        contexts = _job_contexts(_WORKFLOW)
        # The four seams the workflow header names. A floor, not an inventory —
        # the equality above is what keeps the set exact.
        assert len(contexts) >= 4, contexts
        assert "backend" in contexts
        assert contexts["backend"] == "Backend tests (pytest)"

    def test_required_contexts_are_collected(self):
        assert len(_required_contexts(_load_ruleset())) >= 4

    def test_step_names_are_not_collected(self):
        # `Install dependencies` appears as a step name in three jobs. If the
        # parser ever matched `- name:` the equality above would fail on it, but
        # only by luck of that string differing from a job name — assert the
        # structural exclusion directly.
        collected = set(_job_contexts(_WORKFLOW).values())
        assert "Install dependencies" not in collected
        assert "Run pytest" not in collected
        # The workflow's own top-level `name: CI` sits at indent 0 and is not a
        # job either.
        assert "CI" not in collected

    def test_parser_handles_the_shapes_yaml_allows(self, tmp_path):
        workflow = tmp_path / "w.yml"
        workflow.write_text(
            "name: CI\n"
            "on:\n"
            "  pull_request:\n"
            "    branches: [main]\n"
            "jobs:\n"
            "  named:\n"
            "    name: Backend tests (pytest)\n"
            "    steps:\n"
            "      - name: Install dependencies\n"
            "        run: pip install -r requirements.txt\n"
            "  quoted:\n"
            '    name: "Frontend typecheck + build"\n'
            "  commented:\n"
            "    name: UI tests (Playwright)  # the slow one\n"
            "  unnamed:\n"
            "    runs-on: ubuntu-latest\n"
            "\n"
            "permissions:\n"
            "  contents: read\n",
            encoding="utf-8",
        )
        assert _job_contexts(workflow) == {
            "named": "Backend tests (pytest)",
            "quoted": "Frontend typecheck + build",
            "commented": "UI tests (Playwright)",
            # No `name:`, so GitHub reports the job id — and so does this.
            "unnamed": "unnamed",
        }

    def test_keys_after_the_jobs_block_are_not_jobs(self, tmp_path):
        # `permissions:` above sits after `jobs:` in the fixture and must not be
        # collected. Asserted separately because a parser that walked to EOF
        # would still produce the four correct entries in that test.
        workflow = tmp_path / "w.yml"
        workflow.write_text(
            "jobs:\n"
            "  only:\n"
            "    name: The one job\n"
            "permissions:\n"
            "  contents: read\n",
            encoding="utf-8",
        )
        assert _job_contexts(workflow) == {"only": "The one job"}
