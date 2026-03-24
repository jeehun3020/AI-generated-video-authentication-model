#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${PYTHON_BIN:-}" ]]; then
  RUN_PYTHON="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  RUN_PYTHON=".venv/bin/python"
else
  RUN_PYTHON="python3"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

FRAME_CONFIG="${FRAME_CONFIG:-configs/baseline_binary_allvideo.yaml}"
TEMPORAL_CONFIG="${TEMPORAL_CONFIG:-configs/temporal_binary_allvideo.yaml}"

"$RUN_PYTHON" prepare_data.py --config "$FRAME_CONFIG"
"$RUN_PYTHON" train.py --config "$FRAME_CONFIG"
"$RUN_PYTHON" eval.py --config "$FRAME_CONFIG" --split test --checkpoint outputs/checkpoints_allvideo_binary/best.pt

"$RUN_PYTHON" train_temporal.py --config "$TEMPORAL_CONFIG"
"$RUN_PYTHON" eval_temporal.py --config "$TEMPORAL_CONFIG" --split test --checkpoint outputs/checkpoints_temporal_allvideo_binary/best.pt
