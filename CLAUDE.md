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

## Artifact Registry cleanup policy

The `cloud-run-source-deploy` repository deletes images on a schedule (live since
2026-09-25, #144). Every version older than 7 days is deleted, except the 5 most
recent per `scienceq-*` package and anything tagged `latest*`.

`infra/artifact-registry/cloud-run-source-deploy.json` is the record, not the
enforcement — Artifact Registry reads the repository's settings, not that file.
After any change made in the console or with gcloud, re-export with
`python scripts/export_cleanup_policy.py`. The file is the API's response
as-is, so it is not a valid `set-cleanup-policies --policy` input.
`tests/test_cleanup_policy.py` holds it against the Cloud Build and Cloud Run
configs. Renaming a package or the pipeline's `latest` tag without updating the
policy fails the backend suite. Without that test, the rule would stop matching
and the image would be deleted.

**Before pinning traffic to a revision, confirm its image is among the 5 most
recent for its package, or add a keep rule for its digest.** A pinned revision is
kept only by recency, so five deploys after a pin the policy deletes the image
production is serving. Traffic was pinned once, on 2026-07-25 (see
`cloudbuild-api.yaml`). The test cannot see traffic, so this check is manual.

## Compliance

Read `docs/COMPLIANCE.md` before working on features that share, publish, index, or
auto-post generated answers, or that add new output modalities (audio/image/video) —
it lists feature tripwires that require an EU AI Act re-assessment before shipping,
plus standing rules and watch dates. Do not record "it's open source" as a reason any
transparency obligation doesn't apply.
