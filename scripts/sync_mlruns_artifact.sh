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

if ! command -v unzip &> /dev/null; then
  echo "❌ 'unzip' is required. Install it with 'sudo apt install unzip -y'"
  exit 1
fi

if ! command -v rsync &> /dev/null; then
  echo "❌ 'rsync' is required. Install it with 'sudo apt install rsync -y'"
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

# ----------- CREATE PROJECT-LOCAL TEMP DIR -----------
mkdir -p "$TARGET_DIR"

TEMP_DIR="./tmp_mlruns_sync_$(date +%s)"
mkdir -p "$TEMP_DIR"
echo "📥 Downloading artifact '$ARTIFACT_NAME' from run $RUN_ID into temp dir: $TEMP_DIR"

cd "$TEMP_DIR" || exit 1
gh run download "$RUN_ID" -n "$ARTIFACT_NAME"

cd ..
# ----------- EXTRACT & SYNC -----------
echo "📂 Extracting and syncing to '$TARGET_DIR'..."
rsync -a "$TEMP_DIR/" "$TARGET_DIR/"

# ----------- CLEANUP -----------
cd ..
rm -rf "$TEMP_DIR"

echo "✅ Sync completed. mlruns is available at: $TARGET_DIR"

# Optional: re-deploy service
if [ -f docker-compose.yml ]; then
  echo "🔄 Restarting mlflow-ui with docker-compose..."
  docker compose up -d mlflow-ui
fi