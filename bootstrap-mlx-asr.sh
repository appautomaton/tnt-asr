#!/usr/bin/env bash
# Point TNT at a Qwen3-ASR-1.7B MLX checkpoint by symlinking it into
# bin/qwen3-asr-mlx (and the per-user location a pip/uv-tool install reads).
#
# TNT's runtime is local-path only (no network at runtime). This setup-time
# script can fetch the published checkpoint for you, or link one you already
# have:
#
#   ./bootstrap-mlx-asr.sh                       # download the default int8 build from Hugging Face
#   ./bootstrap-mlx-asr.sh <hf-repo-id>          # download a specific Hugging Face repo
#   ./bootstrap-mlx-asr.sh /path/to/checkpoint   # link a local checkpoint dir (no download)
#
# Downloads go through huggingface_hub (already installed via mlx-speech) and
# are cached/managed by Hugging Face under ~/.cache/huggingface; we symlink to
# the resolved snapshot. No new runtime dependency is added — TNT still loads a
# local path at runtime.
set -euo pipefail

cd "$(dirname "$0")"

DEFAULT_REPO="appautomaton/qwen3-asr-1.7b-int8-mlx"
DEST="bin/qwen3-asr-mlx"
USER_DEST="$HOME/.local/share/tnt/qwen3-asr-mlx"
ARG="${1:-}"

# Resolve ARG to a local checkpoint directory (SRC_ABS): an existing dir is
# used as-is; anything else (empty or a repo id) is downloaded from the Hub.
if [[ -n "$ARG" && -d "$ARG" ]]; then
    SRC_ABS="$(cd "$ARG" && pwd)"
    echo "Using local checkpoint: $SRC_ABS"
else
    REPO="${ARG:-$DEFAULT_REPO}"
    echo "Downloading $REPO from Hugging Face (managed cache, ~2.5 GB for int8)..."
    SRC_ABS="$(uv run python - "$REPO" <<'PY' | tail -n1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY
)"
    echo "Downloaded to: $SRC_ABS"
fi

for f in config.json model.safetensors preprocessor_config.json vocab.json merges.txt; do
    if [[ ! -f "$SRC_ABS/$f" ]]; then
        echo "error: $SRC_ABS is missing $f (not a converted MLX checkpoint?)" >&2
        exit 1
    fi
done

ln -sfn "$SRC_ABS" "$DEST"
echo "Linked $DEST -> $SRC_ABS"

# Also link the per-user location so a pip/uv-tool-installed tnt finds the
# model when run outside this checkout.
mkdir -p "$(dirname "$USER_DEST")"
ln -sfn "$SRC_ABS" "$USER_DEST"
echo "Linked $USER_DEST -> $SRC_ABS"

echo "Done. Run: uv run tnt"
