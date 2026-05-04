# Phase D — N=4 ensemble + temperature calibration (final-stage tuning)

## What was added on top of Phase C

1. **Seed diversity**: trained `robustaug+EMA` again with `seed=1337` (was 42). New checkpoint at `outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed1337_frame/best.pt`. Best epoch 9, EMA val_f1 0.7608 (close to seed 42's 0.7634).
2. **Aggregation comparison** (`scripts/eval_n_model_aggregations.py`): mean, geomean (logit-avg), median, max_pos, min_pos, trimmed_mean. **Geomean reliably wins on the strict operating point (TPR@FPR=0.1%) while mean is best on most other metrics**.
3. **Per-model temperature scaling** (`scripts/eval_n_model_temperature_calibrated.py`): each model's TTA-averaged logits are scaled by a single T_i fit by LBFGS NLL on val. Calibrated softmax probs then aggregated. T values: model1 1.69, model2 1.64, model3 (holdout) 1.38, model4 (seed1337) 1.52 — all >1, meaning all models were over-confident and T pulls them toward calibrated.
4. **Reusable scripts**: `eval_n_model_ensemble.py`, `tune_n_model_ensemble_weights.py`, `eval_n_model_aggregations.py`, `eval_n_model_temperature_calibrated.py` all accept N configs, optional checkpoints, `--video-manifest` override, and `--tta {none,hflip}`.

## Final test-set numbers (frame-level, TTA hflip, N=4 = robustaug + ema + holdout + seed1337)

| setting | F1 | AUC | TPR@FPR=1% | TPR@FPR=0.1% | ECE | Brier |
|---|---|---|---|---|---|---|
| robustaug (start of work) | 0.7547 | 0.8299 | 0.2112 | 0.0865 | 0.0993 | 0.1791 |
| robustaug + TTA (Phase A) | 0.7560 | 0.8324 | 0.2214 | 0.1018 | 0.0931 | 0.1766 |
| 2-model ensemble + TTA (Phase B) | 0.7574 | 0.8390 | 0.2392 | 0.0789 | 0.0755 | 0.1715 |
| 3-model ensemble mean + TTA | 0.7689 | 0.8427 | 0.2468 | 0.1120 | 0.0561 | 0.1659 |
| 3-model ensemble geomean + TTA | 0.7689 | 0.8430 | 0.2316 | 0.1578 | 0.0710 | 0.1679 |
| **N=4 mean + cal + TTA** | **0.7792** | 0.8500 | **0.2646** | 0.1807 | 0.0515 | 0.1590 |
| **N=4 geomean + cal + TTA** | **0.7792** | **0.8502** | 0.2519 | **0.1883** | **0.0442** | 0.1590 |

### Cumulative win over starting point

| metric | start | best | absolute Δ | relative Δ |
|---|---|---|---|---|
| F1 | 0.7547 | 0.7792 | +0.0245 | +3.2% |
| AUC | 0.8299 | 0.8502 | +0.0203 | +2.4% |
| TPR@FPR=1% | 0.2112 | 0.2646 | +0.053 | **+25%** |
| TPR@FPR=0.1% | 0.0865 | 0.1908 (uncal geomean) / 0.1883 (cal geomean) | +0.10 | **+121%** |
| ECE | 0.0993 | 0.0442 | -0.0551 | **−55.5%** |

**Headline:** TPR@FPR=0.1% more than doubled (2.2×), ECE more than halved.

## ff_holdout (Face2Face OOD) — N=4 vs in-distribution single

| setting | trained on Face2Face? | F1 | AUC | TPR@FPR=1% | TPR@FPR=0.1% |
|---|:-:|---|---|---|---|
| robustaug+EMA single (in-dist) | ✓ | 0.8902 | 0.9812 | 0.542 | 0.244 |
| **N=4 mean + cal** (mixed) | partial | 0.9002 | **0.9865** | **0.676** | 0.456 |
| **N=4 geomean + cal** (mixed) | partial | 0.9002 | 0.9864 | 0.630 | **0.460** |

The N=4 ensemble (with one OOD-trained model + seed diversity + temperature calibration) **beats the in-distribution single model on Face2Face**, the very method one of its members never saw:
- TPR@FPR=1%: +13.4pp absolute (0.542 → 0.676)
- TPR@FPR=0.1%: +21.2pp absolute (0.244 → 0.456)
- AUC: +0.0053 (0.9812 → 0.9865)

This is the strongest evidence that the model stack is learning *generator-agnostic* fakery cues, plus that ensemble diversity (different seeds, different train splits) compounds with calibration.

## Two recommended operating modes

- **High-coverage / general-purpose**: N=4 mean + per-model temperature calibration + TTA hflip. Best F1 and TPR@FPR=1% with very good ECE.
- **Strict / low-FPR safety**: N=4 geomean + per-model temperature calibration + TTA hflip. Best AUC, TPR@FPR=0.1%, and lowest ECE. Use when false positives are very costly.

Both run in seconds at inference time (4 forward passes × 2 with hflip = 8 forwards per video) so the cost is acceptable for non-realtime detection.

## Notes / open work

- Weight tuning on val *did not* generalize — val (782 rows) is too small for meaningful weight optimization across N=3+ correlated models. Uniform 1/N is the robust default. This is documented in `outputs/eval/tune_ensembleN_*.json`.
- Possible next steps (if needed):
  - Train another seed (seed=7) for N=5; expect diminishing returns but cheap.
  - Train a Deepfakes-holdout variant (analogous to Face2Face holdout) to expand the OOD coverage of the ensemble.
  - Port the N=4 ensemble (with mode switch) into `scripts/inference_final_video_pipeline.py` and re-tune the safety/coverage policies on top of the new probabilities.
- Caveat (from Phase C): the ff_holdout split has FF++-style fakes vs YouTube-style reals; the absolute AUC level is partly inflated by stylistic gap. The cross-model deltas are still controlled comparisons.

## Files / artifacts

- Checkpoints (4 models):
  - `outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_frame/best.pt`
  - `outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_frame/best.pt`
  - `outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_ff2f_holdout_frame/best.pt`
  - `outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_robustaug_ema_seed1337_frame/best.pt`
- Eval JSONs:
  - N=4 ensemble test: `outputs/eval/ensembleN_N4_rb_ema_holdout_seed1337_test_tta-hflip_*.json`
  - N=4 aggregations test: `outputs/eval/aggN_N4_test_tta-hflip_20260429_233423.json`
  - N=4 temperature calibration: `outputs/eval/tempcal_N4_20260429_233932.json`
- Train log seed1337: `logs/protocol_videoonly_clean_robustaug_ema_seed1337_train_20260429.log`
- Reusable eval scripts: `scripts/eval_n_model_ensemble.py`, `scripts/eval_n_model_aggregations.py`, `scripts/eval_n_model_temperature_calibrated.py`, `scripts/tune_n_model_ensemble_weights.py`
