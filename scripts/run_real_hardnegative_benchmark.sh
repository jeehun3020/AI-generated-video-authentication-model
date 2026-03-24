#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_ROOT="${INPUT_ROOT:-data/raw/youtube_shorts_real_hardnegative/real}"
ARGMAX_CONFIG="${ARGMAX_CONFIG:-configs/temporal_binary_shorts_holdout_argmax.yaml}"
AUTO_CONFIG="${AUTO_CONFIG:-configs/temporal_binary_shorts_holdout.yaml}"
FRAME_CKPT="${FRAME_CKPT:-outputs/checkpoints_shorts_binary_holdout/best.pt}"
TEMPORAL_CKPT="${TEMPORAL_CKPT:-outputs/checkpoints_temporal_shorts_binary_holdout/best.pt}"
ARGMAX_JSON="${ARGMAX_JSON:-outputs/eval/real_hardnegative_argmax.json}"
ARGMAX_CSV="${ARGMAX_CSV:-outputs/eval/real_hardnegative_argmax.csv}"
AUTO_JSON="${AUTO_JSON:-outputs/eval/real_hardnegative_auto.json}"
AUTO_CSV="${AUTO_CSV:-outputs/eval/real_hardnegative_auto.csv}"
COMPARE_JSON="${COMPARE_JSON:-outputs/eval/real_hardnegative_policy_compare.json}"
COMPARE_MD="${COMPARE_MD:-outputs/eval/real_hardnegative_policy_compare.md}"

source .venv/bin/activate

python scripts/run_hardcase_batch_inference.py \
  --input-root "$INPUT_ROOT" \
  --config "$ARGMAX_CONFIG" \
  --frame-checkpoint "$FRAME_CKPT" \
  --temporal-checkpoint "$TEMPORAL_CKPT" \
  --expected-label real \
  --output-json "$ARGMAX_JSON" \
  --output-csv "$ARGMAX_CSV"

python scripts/run_hardcase_batch_inference.py \
  --input-root "$INPUT_ROOT" \
  --config "$AUTO_CONFIG" \
  --frame-checkpoint "$FRAME_CKPT" \
  --temporal-checkpoint "$TEMPORAL_CKPT" \
  --expected-label real \
  --output-json "$AUTO_JSON" \
  --output-csv "$AUTO_CSV"

python scripts/compare_batch_predictions.py \
  --left-json "$ARGMAX_JSON" \
  --right-json "$AUTO_JSON" \
  --left-name argmax \
  --right-name adaptive_auto \
  --expected-label real \
  --output-json "$COMPARE_JSON" \
  --output-md "$COMPARE_MD"

echo "[INFO] real hard-negative benchmark finished"
echo "[INFO] comparison markdown: $COMPARE_MD"
