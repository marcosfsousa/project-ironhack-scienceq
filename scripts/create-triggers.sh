#!/usr/bin/env bash
# create-triggers.sh
# ------------------
# Creates (or replaces) the two Cloud Build triggers that auto-deploy
# scienceq-api and scienceq-web on every push to main.
#
# Prerequisites (one-time, manual):
#   1. Connect your GitHub repo in the GCP Console:
#      Cloud Build → Triggers → Connect Repository → GitHub
#   2. Ensure the Cloud Build service account has the following roles:
#        roles/run.admin
#        roles/iam.serviceAccountUser   (on the Compute default SA)
#        roles/artifactregistry.writer
#
# Usage:
#   bash scripts/create-triggers.sh

set -euo pipefail

PROJECT="scienceq-prod"
REGION="europe-west1"
REPO_OWNER="marcosfsousa"
REPO_NAME="project-ironhack-scienceq"
BRANCH="^main$"

echo "==> Deleting existing triggers (if any) …"
for trigger in deploy-scienceq-api deploy-scienceq-web; do
  if gcloud builds triggers describe "$trigger" --project="$PROJECT" &>/dev/null; then
    gcloud builds triggers delete "$trigger" --project="$PROJECT" --quiet
    echo "    Deleted $trigger"
  else
    echo "    $trigger not found — skipping"
  fi
done

echo ""
echo "==> Creating trigger: deploy-scienceq-api"
gcloud builds triggers create github \
  --name="deploy-scienceq-api" \
  --project="$PROJECT" \
  --repo-owner="$REPO_OWNER" \
  --repo-name="$REPO_NAME" \
  --branch-pattern="$BRANCH" \
  --build-config="cloudbuild-api.yaml" \
  --included-files="api/**,agent/**,pipeline/**,Dockerfile,requirements.txt,cloudbuild-api.yaml" \
  --description="Deploy scienceq-api to Cloud Run on changes to API or agent code"

echo ""
echo "==> Creating trigger: deploy-scienceq-web"
gcloud builds triggers create github \
  --name="deploy-scienceq-web" \
  --project="$PROJECT" \
  --repo-owner="$REPO_OWNER" \
  --repo-name="$REPO_NAME" \
  --branch-pattern="$BRANCH" \
  --build-config="cloudbuild-web.yaml" \
  --included-files="frontend/**,Dockerfile.web,cloudbuild-web.yaml" \
  --description="Deploy scienceq-web to Cloud Run on changes to frontend code"

echo ""
echo "==> Done. Triggers created:"
gcloud builds triggers list --project="$PROJECT" --filter="name:(deploy-scienceq-api OR deploy-scienceq-web)" --format="table(name,createTime)"
