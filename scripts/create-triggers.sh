#!/usr/bin/env bash
# create-triggers.sh
# ------------------
# Creates (or replaces) the two Cloud Build triggers that auto-deploy
# scienceq-api and scienceq-web on every push to main.
#
# Prerequisites (one-time, manual):
#   Connect your GitHub repo in the GCP Console:
#   Cloud Build → Triggers → Connect Repository → GitHub
#
# Usage:
#   bash scripts/create-triggers.sh

set -euo pipefail

PROJECT="scienceq-prod"
REGION="global"
REPO_OWNER="marcosfsousa"
REPO_NAME="project-ironhack-scienceq"
BRANCH="^main$"
SA_NAME="cloudbuild-deployer"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
SA_RESOURCE="projects/${PROJECT}/serviceAccounts/${SA_EMAIL}"

# ── Service account ────────────────────────────────────────────────────────────

echo "==> Ensuring service account ${SA_EMAIL} exists …"
if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT" &>/dev/null; then
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="Cloud Build Deployer" \
    --project="$PROJECT"
  echo "    Created ${SA_EMAIL}"
else
  echo "    Already exists — skipping"
fi

echo ""
echo "==> Granting roles to ${SA_EMAIL} …"
for role in \
  roles/cloudbuild.builds.builder \
  roles/run.admin \
  roles/artifactregistry.writer \
  roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$role" \
    --condition=None \
    --quiet
  echo "    Granted $role"
done

# ── Triggers ───────────────────────────────────────────────────────────────────

echo ""
echo "==> Deleting existing triggers (if any) …"
for trigger in deploy-scienceq-api deploy-scienceq-web; do
  if gcloud builds triggers describe "$trigger" --project="$PROJECT" --region="$REGION" &>/dev/null; then
    gcloud builds triggers delete "$trigger" --project="$PROJECT" --region="$REGION" --quiet
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
  --region="$REGION" \
  --repo-owner="$REPO_OWNER" \
  --repo-name="$REPO_NAME" \
  --branch-pattern="$BRANCH" \
  --build-config="cloudbuild-api.yaml" \
  --service-account="$SA_RESOURCE" \
  --included-files="api/**,agent/**,pipeline/**,Dockerfile,requirements.txt,cloudbuild-api.yaml" \
  --description="Deploy scienceq-api to Cloud Run on changes to API or agent code"

echo ""
echo "==> Creating trigger: deploy-scienceq-web"
gcloud builds triggers create github \
  --name="deploy-scienceq-web" \
  --project="$PROJECT" \
  --region="$REGION" \
  --repo-owner="$REPO_OWNER" \
  --repo-name="$REPO_NAME" \
  --branch-pattern="$BRANCH" \
  --build-config="cloudbuild-web.yaml" \
  --service-account="$SA_RESOURCE" \
  --included-files="frontend/**,Dockerfile.web,cloudbuild-web.yaml" \
  --description="Deploy scienceq-web to Cloud Run on changes to frontend code"

echo ""
echo "==> Done. Active triggers:"
gcloud builds triggers list --project="$PROJECT" --region="$REGION" \
  --filter="name:(deploy-scienceq-api OR deploy-scienceq-web)" \
  --format="table(name,createTime)"
