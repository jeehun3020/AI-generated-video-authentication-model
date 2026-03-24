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

FRAME_CONFIG="${FRAME_CONFIG:-configs/baseline_binary_allvideo.yaml}"
TEMPORAL_CONFIG="${TEMPORAL_CONFIG:-configs/temporal_binary_allvideo.yaml}"
FFT_CONFIG="${FFT_CONFIG:-configs/baseline_binary_allvideo_fft.yaml}"

FRAME_CKPT="${FRAME_CKPT:-outputs/checkpoints_allvideo_binary/best.pt}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-outputs/checkpoints_temporal_allvideo_binary/best.pt}"
FFT_CKPT="${FFT_CKPT:-outputs/checkpoints_allvideo_binary_fft/best.pt}"

"$RUN_PYTHON" scripts/tune_multicue_ensemble.py \
  --component "rgb:frame:${FRAME_CONFIG}:${FRAME_CKPT}" \
  --component "temporal:temporal:${TEMPORAL_CONFIG}:${TEMPORAL_CKPT}" \
  --component "fft:frame:${FFT_CONFIG}:${FFT_CKPT}" \
  --samples "${SAMPLES:-600}" \
  --output-json "${OUTPUT_JSON:-outputs/eval/multicue_allvideo_tuning.json}"
