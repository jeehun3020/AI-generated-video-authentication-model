#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${PYTHON_BIN:-}" ]]; then
  RUN_PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1 && python -c "import torch, timm" >/dev/null 2>&1; then
  RUN_PYTHON="python"
elif [[ -x ".venv/bin/python" ]]; then
  RUN_PYTHON=".venv/bin/python"
else
  RUN_PYTHON="python3"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

CONFIG_PATH="${CONFIG_PATH:-configs/baseline_binary_allvideo_textmask.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/checkpoints_allvideo_binary_textmask/best.pt}"

"$RUN_PYTHON" inference.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  "$@"
