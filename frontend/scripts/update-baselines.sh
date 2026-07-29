#!/usr/bin/env bash
# Regenerate the committed Linux pixel baselines (issue #89, option 1).
#
# Baselines must match what CI renders, and CI is ubuntu-latest. Generating them
# on Windows or macOS produces font rasterisation that will not match, so this
# runs the suite inside the Playwright image whose browser build CI installs.
#
# Usage, from frontend/:   npm run test:e2e:baselines
# Then review the diff — every changed PNG is a UI change you are approving.
set -euo pipefail

cd "$(dirname "$0")/.."
FRONTEND="$(pwd)"

# Windows/Git Bash needs a native path for the bind mount; `pwd -W` supplies it
# and is a no-op elsewhere. MSYS_NO_PATHCONV stops MSYS rewriting /w.
ROOT="$(cd .. && { pwd -W 2>/dev/null || pwd; })"

# Pinned to the installed @playwright/test so the image and the runner can never
# drift apart — a mismatched image would quietly produce baselines that fail CI.
VERSION="$(node -p "require('@playwright/test/package.json').version")"
IMAGE="mcr.microsoft.com/playwright:v${VERSION}-noble"

echo "frontend : ${FRONTEND}"
echo "image    : ${IMAGE}"

# The anonymous volume on node_modules keeps the container's Linux binaries from
# overwriting the host's (esbuild/rollup ship platform-specific binaries, so a
# shared node_modules would break the host toolchain).
MSYS_NO_PATHCONV=1 exec docker run --rm \
  -v "${ROOT}:/w" \
  -v /w/frontend/node_modules \
  -w /w/frontend \
  "${IMAGE}" \
  bash -lc "npm ci && npx playwright test --update-snapshots --project=desktop --project=mobile"
