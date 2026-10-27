#!/usr/bin/env bash
# Point TNT at an MLX checkpoint by symlinking it into bin/ (and the per-user
# location a pip/uv-tool install reads). Confucius4-R2T2 is the default model;
# Qwen3-ASR is optional, and TNT offers every linked model on `m`.
#
# TNT's runtime is local-path only (no network at runtime). This setup-time
# script can fetch the published checkpoint for you, or link one you already
# have:
#
#   ./bootstrap-mlx-asr.sh                          # download R2T2 from Hugging Face
#   ./bootstrap-mlx-asr.sh qwen3                    # download Qwen3-ASR (int8)
#   ./bootstrap-mlx-asr.sh r2t2 <hf-repo-id>        # download a specific repo
#   ./bootstrap-mlx-asr.sh qwen3 /path/to/checkpoint  # link a local dir (no download)
#
# Downloads go through huggingface_hub (already installed via mlx-speech) and
# are cached/managed by Hugging Face under ~/.cache/huggingface; we symlink to
# the resolved snapshot. No new runtime dependency is added — TNT still loads a
# local path at runtime.
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${1:-r2t2}"
ARG="${2:-}"
case "$MODEL" in
    r2t2)
        DEFAULT_REPO="appautomaton/confucius4-r2t2-bf16-mlx"
        SIZE="~3.8 GB, bf16"
        NAME="r2t2-mlx"
        ;;
    qwen3)
        DEFAULT_REPO="appautomaton/qwen3-asr-1.7b-int8-mlx"
        SIZE="~2.5 GB, int8"
        NAME="qwen3-asr-mlx"
        ;;
    *)
        echo "usage: $0 [r2t2|qwen3] [hf-repo-id | /path/to/checkpoint]" >&2
        exit 2
        ;;
esac
DEST="bin/$NAME"
USER_DEST="$HOME/.local/share/tnt/$NAME"

# Resolve ARG to a local checkpoint directory (SRC_ABS): an existing dir is
# used as-is; anything else (empty or a repo id) is downloaded from the Hub.
if [[ -n "$ARG" && -d "$ARG" ]]; then
    SRC_ABS="$(cd "$ARG" && pwd)"
    echo "Using local checkpoint: $SRC_ABS"
else
    REPO="${ARG:-$DEFAULT_REPO}"
    echo "Downloading $REPO from Hugging Face (managed cache, $SIZE)..."
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
