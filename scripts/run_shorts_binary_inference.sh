#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${PYTHON_BIN:-}" ]]; then
  RUN_PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1 && python -c "import torch, timm, cv2" >/dev/null 2>&1; then
  RUN_PYTHON="python"
elif [[ -x ".venv/bin/python" ]]; then
  RUN_PYTHON=".venv/bin/python"
else
  RUN_PYTHON="python3"
fi

CONFIG_PATH="${CONFIG_PATH:-configs/temporal_binary_shorts_holdout.yaml}"
FRAME_CHECKPOINT="${FRAME_CHECKPOINT:-outputs/checkpoints_shorts_binary_holdout/best.pt}"
TEMPORAL_CHECKPOINT="${TEMPORAL_CHECKPOINT:-outputs/checkpoints_temporal_shorts_binary_holdout/best.pt}"
FREQ_CHECKPOINT="${FREQ_CHECKPOINT:-outputs/checkpoints_shorts_binary_holdout_fft/best.pt}"

"$RUN_PYTHON" inference_ensemble.py \
  --config "$CONFIG_PATH" \
  --frame-checkpoint "$FRAME_CHECKPOINT" \
  --temporal-checkpoint "$TEMPORAL_CHECKPOINT" \
  --freq-checkpoint "$FREQ_CHECKPOINT" \
  "$@"
