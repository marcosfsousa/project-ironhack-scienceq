#!/usr/bin/env python3
# export_cleanup_policy.py
# ------------------------
"""
Writes the live Artifact Registry cleanup policy to
infra/artifact-registry/cloud-run-source-deploy.json (#145).

Artifact Registry does not read that path. The policy lives in the repository's
settings in GCP, applied with ``gcloud artifacts repositories
set-cleanup-policies`` (#143, #144), and the file is the reviewable record of
it. Records drift from what they record, so this is the one command that
refreshes it. Run it after any change made in the console or with gcloud:

    python scripts/export_cleanup_policy.py

Prerequisites: ``gcloud`` authenticated against an account that can read the
repository in ``scienceq-prod``. That is also why
``tests/test_cleanup_policy.py`` cannot check the live config and reads this
file instead.

What is written is the API's response, **not converted**. The policy JSON that
#143 applied spells things the way gcloud accepts them (``"Keep"``, ``"7d"``);
the API hands back its own canonical forms (``"KEEP"``, ``"604800s"``). This
file records the second. Translating it back into gcloud's input dialect would
be a rebuild — the thing a record must not be — and a rebuild is where a field
goes missing. So this file is not promised to be a valid ``--policy`` input.
Re-applying from it means writing the policy file by hand and reading this one
back afterwards.

Of the repository's fields, only these are kept:

  identity        name, registryUri — the test holds every image reference in
                  the build configs against registryUri, so a reference to a
                  different repository fails instead of passing unprotected
  policy          cleanupPolicies, cleanupPolicyDryRun

Everything else (size, timestamps, scanning config) describes the repository,
not the policy. It is ignored rather than classified, since this file is not a
record of the repository.

``cleanupPolicyDryRun`` is written explicitly, false included. The API leaves
out a boolean that is false (#144: after the switch the key was simply gone), so
copying the response as-is would record live mode as an absence. Here it is a
value, and a switch back to dry run shows up in the diff as ``false`` → ``true``.

**Within the policy, this script refuses to run on a key it does not
recognise**, at each point where it picks keys by name — the same rule
``export_ruleset.py`` follows and for the same reason. An unclassified key on a
policy entry is either new config that belongs in the record or new server
state that does not, and both are decisions for a person. ``condition`` and
``mostRecentVersions`` are copied wholesale instead: copying is lossless, and a
new key there lands in the record and shows up in the diff. The test treats a
Keep rule carrying a condition key it does not understand as protecting
nothing, so a key copied through here cannot quietly count as coverage.

Line endings are pinned to LF and the idempotency check compares bytes, as in
``export_ruleset.py`` — on Windows, text mode rewrites line endings while
hiding the rewrite from the comparison.
"""

import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


PROJECT = "scienceq-prod"
LOCATION = "europe-west1"
REPOSITORY = "cloud-run-source-deploy"
REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "infra" / "artifact-registry" / f"{REPOSITORY}.json"
# Every path this script prints is repo-root-relative and posix-separated, so a
# command it suggests can be pasted from the repo root on any platform.
RELATIVE = TARGET.relative_to(REPO_ROOT).as_posix()

POLICY_KEYS = {"id", "action", "condition", "mostRecentVersions"}


def _describe() -> dict:
    # Resolved through PATH here rather than by the OS. On Windows gcloud is
    # gcloud.CMD, which CreateProcess will not find from a bare "gcloud" but
    # will run when handed the full path.
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        sys.exit("gcloud is not on PATH. Install the Cloud SDK and authenticate.")
    command = [
        gcloud, "artifacts", "repositories", "describe", REPOSITORY,
        f"--location={LOCATION}", f"--project={PROJECT}", "--format=json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"{' '.join(command)} failed:\n{result.stderr.strip()}")
    # stdout only: gcloud writes "Encryption:" and "Repository Size:" lines to
    # stderr ahead of the JSON.
    return json.loads(result.stdout)


def _policy(entry: dict) -> OrderedDict:
    unknown = set(entry) - POLICY_KEYS
    if unknown:
        sys.exit(
            f"The cleanup policy {entry.get('id')!r} carries fields this script "
            "does not classify:\n"
            + "\n".join(f"  {name}" for name in sorted(unknown))
            + "\n\nAdd each to POLICY_KEYS if it is configuration that belongs "
            "in the record, and teach tests/test_cleanup_policy.py what it "
            "means. Until then this export would drop it, and a record that "
            "drops a field of a deletion rule misstates what gets deleted."
        )
    # Ordered the way a rule is read: which one, what it does, what it matches.
    out = OrderedDict(id=entry["id"], action=entry["action"])
    for key in ("condition", "mostRecentVersions"):
        if key in entry:
            out[key] = OrderedDict(sorted(entry[key].items()))
    return out


def _normalize(live: dict) -> OrderedDict:
    policies = live.get("cleanupPolicies", {})
    return OrderedDict(
        name=live["name"],
        registryUri=live["registryUri"],
        cleanupPolicyDryRun=live.get("cleanupPolicyDryRun", False),
        # The API returns a map keyed by id. A list sorted by id reads the same
        # on every export, so the diff shows changes and nothing else.
        cleanupPolicies=[_policy(policies[key]) for key in sorted(policies)],
    )


if __name__ == "__main__":
    exported = json.dumps(_normalize(_describe()), indent=2) + "\n"

    payload = exported.encode("utf-8")
    before = TARGET.read_bytes() if TARGET.is_file() else None
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(exported)

    if before == payload:
        print(f"{RELATIVE} already matches {REPOSITORY}.")
    else:
        print(
            f"{RELATIVE} updated from {REPOSITORY}.\n"
            "Review the diff: what changed here changed in the repository's "
            "settings, and this is the only place it gets read. Then run "
            "pytest tests/test_cleanup_policy.py."
        )
