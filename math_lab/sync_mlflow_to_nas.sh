#!/usr/bin/env bash
# sync_mlflow_to_nas.sh — push the local mlruns/ to the NAS landing dir.
# Idempotent: re-runs only copy new/changed files.
#
# Run from autoresearch root:
#   ./math_lab/sync_mlflow_to_nas.sh
# Or from anywhere with explicit path:
#   ./math_lab/sync_mlflow_to_nas.sh /path/to/mlruns
set -euo pipefail

LOCAL=${1:-"$HOME/Documents/autoresearch/mlruns"}
NAS_USER=claudeli
NAS_HOST=192.168.1.171
NAS_PATH="/home/claudeli/autoresearch/mlflow"
SSH_KEY="$HOME/.ssh/autoresearch_macmini"

if [ ! -d "$LOCAL" ]; then
  echo "no local mlruns at $LOCAL — nothing to sync"
  exit 0
fi

# UGOS rsync is blocked (cannot set euid as root); fall back to scp -O -r.
# tar-piped scp avoids the n²-file overhead UGOS has on directory walks.
echo "  syncing $LOCAL  ->  $NAS_USER@$NAS_HOST:$NAS_PATH/"
ssh -i "$SSH_KEY" "$NAS_USER@$NAS_HOST" "mkdir -p $NAS_PATH"
tar -czf - -C "$LOCAL" . | \
  ssh -i "$SSH_KEY" "$NAS_USER@$NAS_HOST" "tar -xzf - -C $NAS_PATH"
echo "  done.  size:"
ssh -i "$SSH_KEY" "$NAS_USER@$NAS_HOST" "du -sh $NAS_PATH"
