#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

echo "[INFO] face preprocessing"
"$PYTHON_BIN" prepare_data.py --config configs/baseline_binary_shorts_face.yaml

echo "[INFO] face frame training"
"$PYTHON_BIN" train.py --config configs/baseline_binary_shorts_face.yaml

echo "[INFO] face frame evaluation"
"$PYTHON_BIN" eval.py --config configs/baseline_binary_shorts_face.yaml --split test --checkpoint outputs/checkpoints_shorts_binary_face/best.pt

echo "[INFO] background preprocessing"
"$PYTHON_BIN" prepare_data.py --config configs/baseline_binary_shorts_background.yaml

echo "[INFO] background frame training"
"$PYTHON_BIN" train.py --config configs/baseline_binary_shorts_background.yaml

echo "[INFO] background frame evaluation"
"$PYTHON_BIN" eval.py --config configs/baseline_binary_shorts_background.yaml --split test --checkpoint outputs/checkpoints_shorts_binary_background/best.pt

echo "[INFO] background temporal training"
"$PYTHON_BIN" train_temporal.py --config configs/temporal_binary_shorts_background.yaml

echo "[INFO] background temporal evaluation"
"$PYTHON_BIN" eval_temporal.py --config configs/temporal_binary_shorts_background.yaml --split test --checkpoint outputs/checkpoints_temporal_shorts_binary_background/best.pt

echo "[INFO] face temporal training"
"$PYTHON_BIN" train_temporal.py --config configs/temporal_binary_shorts_face.yaml

echo "[INFO] face temporal evaluation"
"$PYTHON_BIN" eval_temporal.py --config configs/temporal_binary_shorts_face.yaml --split test --checkpoint outputs/checkpoints_temporal_shorts_binary_face/best.pt

echo "[INFO] full-frame fft training"
"$PYTHON_BIN" train.py --config configs/baseline_binary_shorts_holdout_fft.yaml

echo "[INFO] full-frame fft evaluation"
"$PYTHON_BIN" eval.py --config configs/baseline_binary_shorts_holdout_fft.yaml --split test --checkpoint outputs/checkpoints_shorts_binary_holdout_fft/best.pt
