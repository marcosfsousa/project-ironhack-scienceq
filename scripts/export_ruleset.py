#!/usr/bin/env python3
# export_ruleset.py
# -----------------
"""
Writes the live ``main`` ruleset to .github/rulesets/main.json (#90).

GitHub does not read that path — it is a directory in the repo like any other.
The ruleset lives in repository settings, applied through the API, and the file
is the reviewable record of it. Records drift from what they record, so this is
the one command that refreshes it. Run it after any change made in the web UI:

    python scripts/export_ruleset.py

Prerequisites: ``gh`` authenticated against a token with admin rights on the
repository. Reading a ruleset needs them, which is also why
``tests/test_required_checks.py`` cannot check the live config and reads this
file instead.

What is written is the API's response minus two classes of field, so the export
stays POSTable as well as accurate:

  server-assigned identity   id, node_id, source, source_type,
                             created_at, updated_at, _links
  viewer-relative state      current_user_can_bypass

The second is the one worth naming. It answers "can *the caller* bypass this",
so it is a property of the token, not of the ruleset — committing it would
record an admin's view as if it were config. A field the API grows that is in
neither class stops this script rather than being dropped silently: an
unclassified field is either new config that belongs in the record, or a new
piece of server state that belongs in the list above, and both are decisions
for a person.

Keys are ordered rather than sorted, so a diff of this file reads in the order
the ruleset is reasoned about — what it is, then who skips it, then where it
applies, then what it does. Parameters within a rule are sorted, since they have
no such order.
"""

import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


REPO = "marcosfsousa/project-ironhack-scienceq"
RULESET_NAME = "main"
TARGET = Path(__file__).resolve().parent.parent / ".github" / "rulesets" / "main.json"

# Order is the argument, not alphabetics. See the module docstring.
SETTABLE = ["name", "target", "enforcement", "bypass_actors", "conditions", "rules"]
SERVER_STATE = {
    "id", "node_id", "source", "source_type", "created_at", "updated_at",
    "_links", "current_user_can_bypass",
}


def _gh(path: str) -> object:
    result = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(f"gh api {path} failed:\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def _find_ruleset_id() -> int:
    """
    Looked up by name, never hardcoded. The id is assigned at creation, so a
    ruleset deleted and recreated — which is how a bad one gets rolled back —
    comes back with a different one, and a hardcoded id would then export a
    ruleset that no longer exists or, worse, a different one that does.
    """
    rulesets = _gh(f"repos/{REPO}/rulesets")
    matches = [r for r in rulesets if r["name"] == RULESET_NAME]
    if not matches:
        sys.exit(
            f"No ruleset named {RULESET_NAME!r} on {REPO}.\n"
            "If it has not been created yet:\n"
            f"  gh api -X POST repos/{REPO}/rulesets --input {TARGET.name}"
        )
    if len(matches) > 1:
        sys.exit(
            f"{len(matches)} rulesets are named {RULESET_NAME!r}; this file can "
            "only be the record of one. Delete or rename the duplicates."
        )
    return matches[0]["id"]


def _normalize(live: dict) -> OrderedDict:
    unclassified = set(live) - set(SETTABLE) - SERVER_STATE
    if unclassified:
        sys.exit(
            "The rulesets API returned fields this script does not classify:\n"
            + "\n".join(f"  {name}" for name in sorted(unclassified))
            + "\n\nAdd each to SETTABLE if it is configuration that belongs in "
            "the committed record, or to SERVER_STATE if it is assigned by "
            "GitHub. Guessing is what this check exists to prevent."
        )

    def rule(entry: dict) -> OrderedDict:
        out = OrderedDict(type=entry["type"])
        if "parameters" in entry:
            out["parameters"] = OrderedDict(sorted(entry["parameters"].items()))
        return out

    ref_name = live["conditions"]["ref_name"]
    return OrderedDict(
        name=live["name"],
        target=live["target"],
        enforcement=live["enforcement"],
        bypass_actors=[OrderedDict(sorted(a.items())) for a in live["bypass_actors"]],
        conditions={
            "ref_name": OrderedDict(
                include=ref_name["include"], exclude=ref_name["exclude"]
            )
        },
        rules=[rule(entry) for entry in live["rules"]],
    )


if __name__ == "__main__":
    ruleset_id = _find_ruleset_id()
    exported = json.dumps(_normalize(_gh(f"repos/{REPO}/rulesets/{ruleset_id}")), indent=2)
    exported += "\n"

    before = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else None
    TARGET.write_text(exported, encoding="utf-8")

    relative = TARGET.relative_to(TARGET.parent.parent.parent)
    if before == exported:
        print(f"{relative} already matches ruleset {ruleset_id}.")
    else:
        print(
            f"{relative} updated from ruleset {ruleset_id}.\n"
            "Review the diff: what changed here changed in repository settings, "
            "and this is the only place it gets read."
        )
