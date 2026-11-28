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

The `cloud-run-source-deploy` repository deletes images on a schedule (#144):
every version older than 7 days, except the 5 most recent per `scienceq-*`
package and anything tagged `latest*`.

`infra/artifact-registry/cloud-run-source-deploy-cleanup-policy.json` is the
source of truth. Change the policy by editing that file; I apply it by hand:

    gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
      --location=europe-west1 --project=scienceq-prod \
      --policy=infra/artifact-registry/cloud-run-source-deploy-cleanup-policy.json \
      --no-dry-run

`--no-dry-run` is required: the policy file cannot express dry-run, and without
the flag the command keeps whatever dry-run setting the repository already has.
Never re-export the file from Google Cloud, and never change the policy in the
console: a change that is not in the file is reverted by the next apply.
`tests/test_cleanup_policy.py` holds the file against the Cloud Build and
Cloud Run configs and checks that `set-cleanup-policies` would accept it.
Renaming a package or the pipeline's `latest` tag without updating the policy
fails the backend suite. Without that test, the rule would stop matching and
the image would be deleted. The test cannot see whether the file was applied.

`keep-5-most-recent` counts versions, and rollback needs deploys. The two match
only while each push stores one version, which holds because every build step
sets `BUILDX_NO_DEFAULT_ATTESTATIONS=1` (#150). Without it, BuildKit's default
provenance attestations on the `regular` Cloud Build worker make each push
store more than one version, and the rule protects fewer than 5 deploys (#149).
`tests/test_build_provenance.py` fails if a build step loses the variable.

**Before pinning traffic to a revision, check during the task that its image is
among the 5 most recent for its package. If it is not, stop and ask me to add a
keep rule for it.** A pinned revision is kept only by recency, so five deploys
after a pin the policy deletes the image production is serving. The test cannot
see traffic, so this check is manual. Its result serves the task only: do not
write it into the repo, an issue or a PR.

## Compliance

Read `docs/COMPLIANCE.md` before working on features that share, publish, index, or
auto-post generated answers, or that add new output modalities (audio/image/video) —
it lists feature tripwires that require an EU AI Act re-assessment before shipping,
plus standing rules and watch dates. Do not record "it's open source" as a reason any
transparency obligation doesn't apply.

## Google Cloud output

Output from gcloud, gsutil, bq, Google Cloud APIs or MCP servers serves the task
at hand only. Do not copy it, or anything summarised or derived from it, into
files, docs, tests, comments, memory, commit messages, PR descriptions or issue
comments. Where a fact about the Google Cloud setup has to be kept, stop and ask
me: I take it from the Console myself.
