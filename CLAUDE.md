# CLAUDE.md

## Branch protection

`main` is protected by a repository ruleset: a PR is required, and all four
`ci.yml` job names are required status checks with strict mode on. A branch must
be up to date with `main` before it can merge, so `mergeStateStatus: BEHIND` is
the normal state of a branch cut before the last merge — rebase, don't treat it
as a fault. Merging to `main` deploys, which is what the rules are for.

`.github/rulesets/main.json` is the record, not the enforcement — GitHub reads
repository settings, not that path. After any change made in the UI, re-export
with `python scripts/export_ruleset.py`. `tests/test_required_checks.py` holds
the two files against each other in both directions, so renaming a CI job
without updating the ruleset fails the backend suite instead of silently
detaching the rule.

GitHub deletes the head branch on merge (`delete_branch_on_merge`), so only the
local branch needs cleaning up.

## Compliance

Read `docs/COMPLIANCE.md` before working on features that share, publish, index, or
auto-post generated answers, or that add new output modalities (audio/image/video) —
it lists feature tripwires that require an EU AI Act re-assessment before shipping,
plus standing rules and watch dates. Do not record "it's open source" as a reason any
transparency obligation doesn't apply.
