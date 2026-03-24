#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

REAL_CSV="data/youtube/shorts_urls_channels_real.csv"
GEN_CSV="data/youtube/shorts_urls_channels_generated.csv"
DOWNLOAD_ROOT="data/raw/youtube_shorts"
RGB_CONFIG="configs/baseline_ultraquick_shortsmix.yaml"
FFT_CONFIG="configs/baseline_ultraquick_shortsmix_fft.yaml"
TEMP_CONFIG="configs/temporal_ultraquick_shortsmix.yaml"

# Keep pilot runs bounded. Increase or remove the limit after the first pass.
LIMIT_PER_LABEL="${LIMIT_PER_LABEL:-200}"

python scripts/download_shorts_dataset.py \
  --csv "$REAL_CSV" \
  --csv "$GEN_CSV" \
  --output-root "$DOWNLOAD_ROOT" \
  --limit-per-label "$LIMIT_PER_LABEL" \
  --report-path outputs/inference/shorts_download_report.json

python prepare_data.py --config "$RGB_CONFIG"

python train.py --config "$RGB_CONFIG"
python eval.py --config "$RGB_CONFIG" --split test --checkpoint outputs/checkpoints_shortsmix/best.pt

python train.py --config "$FFT_CONFIG"
python eval.py --config "$FFT_CONFIG" --split test --checkpoint outputs/checkpoints_shortsmix_fft/best.pt

python train_temporal.py --config "$TEMP_CONFIG"
python eval_temporal.py --config "$TEMP_CONFIG" --split test --checkpoint outputs/checkpoints_temporal_shortsmix/best.pt

echo "[DONE] Shortsmix experiment finished."
