#!/usr/bin/env bash
# Push inference-path files from this Mac to the remote boxes for the
# currently checked-out branch. Run after `git checkout <branch>` and
# before launching serve.py on the remotes.
#
# Usage: sync_to_boxes.sh [mini|windows|all]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET="${1:-all}"

FILES=(
    "math_lab/serve.py"
    "math_lab/finetune.py"
    "math_lab/eval_canonical.py"
    "math_lab/chat.html"
)

push_mini() {
    echo ">> mini (~/Documents/autoresearch/math_lab/)"
    for f in "${FILES[@]}"; do
        rsync -az "$REPO/$f" "mini:Documents/autoresearch/$f"
    done
}

push_windows() {
    echo ">> windows (C:\\Users\\seanj\\Documents\\autoresearch\\math_lab\\)"
    for f in "${FILES[@]}"; do
        scp -q "$REPO/$f" "windows:Documents/autoresearch/$f"
    done
}

case "$TARGET" in
    mini)    push_mini ;;
    windows) push_windows ;;
    all)     push_mini && push_windows ;;
    *)       echo "usage: $0 [mini|windows|all]" >&2; exit 1 ;;
esac

echo
echo "branch:  $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
echo "commit:  $(git -C "$REPO" rev-parse --short HEAD)"
