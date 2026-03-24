#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

CONFIG="configs/baseline_ultraquick.yaml"
CKPT="outputs/checkpoints_ultraquick/best.pt"

python prepare_data.py --config "$CONFIG"
python train.py --config "$CONFIG"
python eval.py --config "$CONFIG" --split val --checkpoint "$CKPT"
python eval.py --config "$CONFIG" --split test --checkpoint "$CKPT"

echo "[DONE] Ultraquick baseline pipeline finished."
