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
record an admin's view as if it were config.

**Anywhere this script enumerates keys, it refuses to run on one it does not
recognise.** An unclassified field is either new config that belongs in the
record or new server state that does not, and both are decisions for a person.
That check used to be top-level only, which made the guarantee above false in
the place it mattered most: ``conditions`` was rebuilt as a hardcoded
``{"ref_name": {include, exclude}}``, so a condition narrowing *where* the
ruleset applies would have been dropped without a word — and this file is
advertised as a POST body, so re-creating from a record that lost a condition
recreates the ruleset at a different scope. Every enumeration point is now
guarded: ``conditions``, ``ref_name``, each ``rules[]`` entry, and each
``bypass_actors[]`` entry.

Two places pass unknown keys through rather than rejecting them, because they
are copied wholesale and copying is lossless: ``rules[].parameters`` and the
values inside a bypass actor. A new parameter lands in the record and shows up
in the diff, which is the outcome wanted.

Keys are ordered rather than sorted, so a diff of this file reads in the order
the ruleset is reasoned about — what it is, then who skips it, then where it
applies, then what it does. Parameters within a rule are sorted, since they have
no such order.

Line endings are pinned to LF and the idempotency check compares **bytes**. Both
matter on Windows: ``write_text`` translates to ``os.linesep`` and ``read_text``
normalises on the way back, so a CRLF rewrite of all 65 lines reported itself as
"already matches". It is the only file in this repo that has ever been CRLF, and
that is how it got there.
"""

import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


REPO = "marcosfsousa/project-ironhack-scienceq"
RULESET_NAME = "main"
REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / ".github" / "rulesets" / "main.json"
# Every path this script prints is repo-root-relative and posix-separated, so a
# command it suggests can be pasted from the repo root on any platform.
RELATIVE = TARGET.relative_to(REPO_ROOT).as_posix()

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
            f"  gh api -X POST repos/{REPO}/rulesets --input {RELATIVE}"
        )
    if len(matches) > 1:
        sys.exit(
            f"{len(matches)} rulesets are named {RULESET_NAME!r}; this file can "
            "only be the record of one. Delete or rename the duplicates."
        )
    return matches[0]["id"]


def _known(mapping: dict, allowed: set, where: str, remedy: str) -> dict:
    """
    ``mapping``, or exit naming what is unrecognised.

    Called at every point where this script picks keys out by name, because
    picking by name is exactly where a field goes missing.
    """
    unknown = set(mapping) - allowed
    if unknown:
        sys.exit(
            f"The rulesets API returned fields this script does not classify, "
            f"in {where}:\n"
            + "\n".join(f"  {name}" for name in sorted(unknown))
            + f"\n\n{remedy}\n"
            "Until then this export would drop them, and the file is a POST "
            "body as well as a record — a dropped field recreates a different "
            "ruleset. Guessing is what this check exists to prevent."
        )
    return mapping


def _normalize(live: dict) -> OrderedDict:
    _known(
        live, set(SETTABLE) | SERVER_STATE, "the ruleset",
        "Add each to SETTABLE if it is configuration that belongs in the "
        "committed record, or to SERVER_STATE if it is assigned by GitHub.",
    )

    def rule(entry: dict) -> OrderedDict:
        _known(
            entry, {"type", "parameters"}, f"the {entry.get('type')!r} rule",
            "A rule entry carrying something other than a type and its "
            "parameters is a shape this script has not seen; extend rule().",
        )
        out = OrderedDict(type=entry["type"])
        if "parameters" in entry:
            # Copied wholesale, so a new parameter is kept rather than
            # rejected — it lands in the record and shows up in the diff.
            out["parameters"] = OrderedDict(sorted(entry["parameters"].items()))
        return out

    def actor(entry: dict) -> OrderedDict:
        _known(
            entry, {"actor_id", "actor_type", "bypass_mode"}, "a bypass actor",
            "A new field on a bypass actor changes who may skip the rules or "
            "how; decide what it means before recording it.",
        )
        return OrderedDict(sorted(entry.items()))

    conditions = _known(
        live["conditions"], {"ref_name"}, "conditions",
        "A condition other than ref_name narrows *where* the ruleset applies. "
        "Dropping one silently is the failure this guard exists for: extend "
        "_normalize to carry it.",
    )
    ref_name = _known(
        conditions["ref_name"], {"include", "exclude"}, "conditions.ref_name",
        "A new key inside ref_name changes which refs match; carry it through "
        "rather than letting the export decide it does not exist.",
    )

    return OrderedDict(
        name=live["name"],
        target=live["target"],
        enforcement=live["enforcement"],
        bypass_actors=[actor(entry) for entry in live["bypass_actors"]],
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

    # Bytes on both sides, LF on the way out. Text mode hides a line-ending
    # rewrite from the comparison while performing one — see the docstring.
    payload = exported.encode("utf-8")
    before = TARGET.read_bytes() if TARGET.is_file() else None
    with open(TARGET, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(exported)

    if before == payload:
        print(f"{RELATIVE} already matches ruleset {ruleset_id}.")
    else:
        print(
            f"{RELATIVE} updated from ruleset {ruleset_id}.\n"
            "Review the diff: what changed here changed in repository settings, "
            "and this is the only place it gets read."
        )
