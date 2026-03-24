#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

RGB_CONFIG="configs/baseline_ultraquick.yaml"
FFT_CONFIG="configs/baseline_ultraquick_fft.yaml"
TEMP_CONFIG="configs/temporal_ultraquick_rgb.yaml"

RGB_CKPT="outputs/checkpoints_ultraquick/best.pt"
FFT_CKPT="outputs/checkpoints_ultraquick_fft/best.pt"
TEMP_CKPT="outputs/checkpoints_temporal_ultraquick_rgb/best.pt"

python prepare_data.py --config "$RGB_CONFIG"

python train.py --config "$RGB_CONFIG"
python eval.py --config "$RGB_CONFIG" --split test --checkpoint "$RGB_CKPT"

python train.py --config "$FFT_CONFIG"
python eval.py --config "$FFT_CONFIG" --split test --checkpoint "$FFT_CKPT"

python train_temporal.py --config "$TEMP_CONFIG"
python eval_temporal.py --config "$TEMP_CONFIG" --split test --checkpoint "$TEMP_CKPT"

echo "[DONE] Multi-cue ultraquick pipeline finished."
