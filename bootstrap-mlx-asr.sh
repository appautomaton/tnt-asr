#!/usr/bin/env bash
# Link a local Qwen3-ASR-1.7B MLX checkpoint into bin/qwen3-asr-mlx.
#
# mlx-speech's Qwen3-ASR runtime is local-path only (no auto-download).
# Convert upstream weights with mlx-voice's scripts/convert/qwen3_asr.py,
# then point this script at the converted directory:
#
#   ./bootstrap-mlx-asr.sh /path/to/Qwen3-ASR-1.7B-MLX-BF16
set -euo pipefail

cd "$(dirname "$0")"

SRC="${1:-}"
DEST="bin/qwen3-asr-mlx"

if [[ -z "$SRC" ]]; then
    echo "usage: $0 /path/to/Qwen3-ASR-1.7B-MLX-BF16" >&2
    exit 1
fi

if [[ ! -d "$SRC" ]]; then
    echo "error: $SRC is not a directory" >&2
    exit 1
fi

for f in config.json model.safetensors preprocessor_config.json vocab.json merges.txt; do
    if [[ ! -f "$SRC/$f" ]]; then
        echo "error: $SRC is missing $f (not a converted MLX checkpoint?)" >&2
        exit 1
    fi
done

SRC_ABS="$(cd "$SRC" && pwd)"
ln -sfn "$SRC_ABS" "$DEST"
echo "Linked $DEST -> $SRC_ABS"
