#!/bin/bash

# ----------- CONFIG ------------
REPO="luandnh/ltek_learn_mlops"         # GitHub repo name
ARTIFACT_NAME="mlruns-logs"             # Artifact name from upload-artifact
TARGET_DIR="./mlruns"                   # Where to extract mlruns to
RUN_ID=${1:-${GITHUB_RUN_ID}}           # GitHub Actions Run ID (auto from env if available)

# Optional: GitHub token if required for private repo access
GITHUB_TOKEN=${GITHUB_TOKEN:-""}

# ----------- CHECK REQUIREMENTS ------------
if ! command -v gh &> /dev/null; then
  echo "❌ 'gh' CLI is not installed. Install: https://cli.github.com"
  exit 1
fi

if [ -z "$RUN_ID" ]; then
  echo "⚠️  Usage: $0 <run-id>"
  echo "ℹ️  You can find run-id from GitHub Actions URL or 'gh run list'"
  exit 1
fi

# ----------- LOGIN IF PRIVATE -----------
if [ -n "$GITHUB_TOKEN" ]; then
  echo "$GITHUB_TOKEN" | gh auth login --with-token
fi

# ----------- DOWNLOAD ARTIFACT -----------
echo "📥 Downloading artifact '$ARTIFACT_NAME' from run $RUN_ID..."
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

gh run download "$RUN_ID" -n "$ARTIFACT_NAME"

cd ..
echo "✅ Sync completed. mlruns is available at: $TARGET_DIR"

# Optional: re-deploy service
if [ -f docker-compose.yml ]; then
  echo "🔄 Restarting mlflow-ui with docker-compose..."
  docker compose up -d mlflow-ui
fi
