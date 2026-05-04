# TTA hflip evaluation — robustaug single model (Phase A)

## Setup
- Checkpoint: `outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt`
- Config: `configs/protocol_youtube_dataset_plus_local_videoonly_clean_robustaug.yaml`
- TTA: average softmax over original + horizontal flip (1 pass + 1 pass = 2x inference cost)
- Same eval pipeline (`eval.py`, `scripts/eval_lowfpr_calibration.py`) with new `--tta {none,hflip}` flag.

## Results

### Test split (779 videos)
| metric            | no-TTA  | TTA hflip | Δ        | rel    |
|-------------------|---------|-----------|----------|--------|
| F1                | 0.7547  | 0.7560    | +0.0013  | +0.2%  |
| AUC               | 0.8299  | 0.8324    | +0.0025  | +0.3%  |
| TPR@FPR=1%        | 0.2112  | 0.2214    | +0.0102  | +4.8%  |
| **TPR@FPR=0.1%**  | 0.0865  | **0.1018**| +0.0153  | **+17.7%** |
| ECE               | 0.0993  | 0.0931    | -0.0062  | -6.2%  |
| Brier             | 0.1791  | 0.1766    | -0.0025  | -1.4%  |

### Val split (782 videos)
| metric            | no-TTA  | TTA hflip | Δ        |
|-------------------|---------|-----------|----------|
| TPR@FPR=1%        | 0.2270  | 0.2551    | +0.0281  |
| TPR@FPR=0.1%      | 0.0077  | 0.0102    | +0.0026  |
| ECE               | 0.0819  | 0.0827    | +0.0008  |
| Brier             | 0.1570  | 0.1544    | -0.0026  |

## Read
- All low-FPR metrics improve on both splits — strongest single gain is **TPR@FPR=0.1% on test +17.7% relative**, the strictest operating point that matters most for a production filter.
- F1/AUC see modest bumps; the headline robustness metric (TPR@FPR) gets the real benefit. Consistent with what TTA usually buys for tightly-cropped natural-image classifiers when training already includes hflip aug.
- ECE essentially unchanged on val and slightly improved on test; Brier improves on both. No calibration regression.
- Val and test both show the same direction → not a test-set lucky run.

## Cost
- 2x inference time (45s → 90s on test). Negligible at deploy time relative to a video being uploaded.

## Recommendation
Adopt `--tta hflip` as the default eval/inference setting. Re-tune the final pipeline policy (`outputs/eval/final_video_pipeline_policy*.json`) on top of TTA-enabled probabilities so the new low-FPR headroom feeds the safety/coverage modes too.

## Files
- `outputs/eval/operating_test_20260428_154858.json` (no-TTA test)
- `outputs/eval/operating_test_tta-hflip_20260428_155931.json` (TTA test)
- `outputs/eval/operating_val_20260428_160040.json` (no-TTA val)
- `outputs/eval/operating_val_tta-hflip_20260428_160125.json` (TTA val)
- `outputs/eval/eval_test_20260428_154658.json` (no-TTA F1/AUC test)
- `outputs/eval/eval_test_tta-hflip_20260428_154754.json` (TTA F1/AUC test)
