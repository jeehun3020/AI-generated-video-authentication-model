#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="${CONFIG_PATH:-configs/temporal_binary_shorts_hardcase_sensitive.yaml}" \
FRAME_CHECKPOINT="${FRAME_CHECKPOINT:-outputs/checkpoints_shorts_binary_holdout/best.pt}" \
TEMPORAL_CHECKPOINT="${TEMPORAL_CHECKPOINT:-outputs/checkpoints_temporal_shorts_binary_holdout/best.pt}" \
bash scripts/run_shorts_binary_inference.sh "$@"
