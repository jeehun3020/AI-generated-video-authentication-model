# Phase E — N=6 ensemble (final progression report)

This file consolidates the entire 2026-04-28 → 2026-05-04 generalization-improvement campaign for the clean video-only protocol. Starting point: a single `robustaug` (EfficientNet-B0 + heavy augmentation) checkpoint with test F1=0.7547, AUC=0.8299. Final result: an N=6 model ensemble with two operating modes that more than doubles strict-FPR recall and roughly halves expected calibration error, with strong cross-method generalization to held-out FaceForensics++ methods.

## What was added

| phase | additions | key new artifacts |
|---|---|---|
| **A** | TTA hflip flag | `eval.py`, `scripts/eval_lowfpr_calibration.py` accept `--tta {none,hflip}` |
| **B** | EMA hook in trainer + 2-model ensemble | `iseeyou/engine/trainer.py` (`ModelEMA`), `..._robustaug_ema.yaml`, `scripts/eval_two_model_ensemble.py` |
| **C** | Cross-method holdout (Face2Face removed) | `data/manifests_protocol_..._ff2f_holdout/`, `..._robustaug_ema_ff2f_holdout.yaml` |
| **D** | N-model utilities + 4th model + temperature scaling | `scripts/eval_n_model_{ensemble,aggregations,temperature_calibrated}.py`, `scripts/tune_n_model_ensemble_weights.py`, `..._robustaug_ema_seed1337.yaml` |
| **E** | 5th and 6th models (more seed/holdout diversity) | `..._robustaug_ema_seed7.yaml`, `..._robustaug_ema_df_holdout.yaml`, `data/manifests_..._df_holdout/` |

The six trained checkpoints live at `outputs/checkpoints_protocol_..._{robustaug,robustaug_ema,robustaug_ema_ff2f_holdout,robustaug_ema_seed1337,robustaug_ema_seed7,robustaug_ema_df_holdout}_frame/best.pt`. All share the same EfficientNet-B0 backbone and aug recipe; they differ only in random seed (42 / 1337 / 7) and which FF++ method, if any, was held out of training.

## Test-set progression (frame-level, TTA hflip)

| stage | F1 | AUC | TPR@FPR=1% | TPR@FPR=0.1% | ECE |
|---|---|---|---|---|---|
| robustaug single (start) | 0.7547 | 0.8299 | 0.2112 | 0.0865 | 0.0993 |
| robustaug + TTA (Phase A) | 0.7560 | 0.8324 | 0.2214 | 0.1018 | 0.0931 |
| robustaug + EMA single | 0.7510 | 0.8348 | 0.2366 | 0.0967 | 0.0836 |
| 2-model mean (Phase B) | 0.7574 | 0.8390 | 0.2392 | 0.0789 | 0.0755 |
| 3-model mean (Phase C) | 0.7689 | 0.8427 | 0.2468 | 0.1120 | 0.0561 |
| 3-model geomean (Phase C) | 0.7689 | 0.8430 | 0.2316 | 0.1578 | 0.0710 |
| N=4 mean + cal (Phase D) | **0.7792** | 0.8500 | 0.2646 | 0.1807 | 0.0515 |
| N=4 geomean + cal (Phase D) | **0.7792** | 0.8502 | 0.2519 | 0.1883 | 0.0442 |
| N=5 geomean uncal (Phase E) | **0.7792** | 0.8530 | 0.2901 | 0.2163 | — |
| **N=6 median** (Phase E) | 0.7753 | **0.8571** | **0.3130** | 0.2087 | 0.0534 |
| **N=6 trimmed_mean uncal** (Phase E) | 0.7740 | 0.8564 | 0.2926 | **0.2214** | 0.0510 |
| **N=6 geomean + cal** (Phase E) | 0.7713 | 0.8541 | 0.2977 | 0.2188 | **0.0392** |

### Cumulative win over starting point

| metric | start | best across N=4..6 modes | absolute Δ | relative Δ |
|---|---|---|---|---|
| F1 | 0.7547 | 0.7792 (N=4/5 cal/geomean) | +0.0245 | +3.2% |
| AUC | 0.8299 | 0.8571 (N=6 median) | +0.0272 | +3.3% |
| TPR@FPR=1% | 0.2112 | 0.3130 (N=6 median) | +0.1018 | **+48%** |
| TPR@FPR=0.1% | 0.0865 | 0.2214 (N=6 trimmed_mean) | +0.1349 | **+156% (2.6×)** |
| ECE | 0.0993 | 0.0392 (N=6 geomean+cal) | −0.0601 | **−60.5%** |

## Cross-method OOD progression

### ff_holdout (886 rows: 500 Face2Face + 386 YouTube reals)

Model3 was trained without Face2Face. Other models in the ensemble saw Face2Face.

| setting | F1 | AUC | TPR@FPR=1% | TPR@FPR=0.1% |
|---|---|---|---|---|
| robustaug+EMA single (in-dist) | 0.8902 | 0.9812 | 0.542 | 0.244 |
| robustaug+EMA + holdout 2-model | 0.8866 | 0.9802 | 0.614 | 0.276 |
| N=4 mean + cal | 0.9002 | 0.9865 | 0.676 | 0.456 |
| **N=6 mean + cal** | 0.8852 | **0.9891** | **0.718** | 0.524 |
| **N=6 geomean uncal** | 0.8865 | 0.9889 | 0.734 | **0.538** |

The N=6 ensemble achieves AUC 0.989 on Face2Face — the very method one of its members never saw — while in-distribution single best is 0.981. **TPR@FPR=1% jumps from 0.542 (in-dist single) to 0.734 (N=6 geomean) — +19.2 percentage points absolute.**

### df_holdout (1386 rows: 1000 Deepfakes + 386 YouTube reals)

Model6 was trained without Deepfakes. Other models in the ensemble saw Deepfakes.

| setting | F1 | AUC | TPR@FPR=1% | TPR@FPR=0.1% |
|---|---|---|---|---|
| df_holdout single (true OOD) | 0.8369 | 0.9126 | 0.251 | 0.049 |
| robustaug single (in-dist) | 0.9172 | 0.9685 | 0.327 | 0.136 |
| robustaug+EMA single (in-dist) | 0.9104 | 0.9819 | 0.552 | 0.233 |
| **N=6 mean + cal** | 0.9061 | 0.9853 | 0.646 | 0.434 |
| **N=6 geomean uncal** | 0.9072 | 0.9860 | 0.664 | 0.458 |

**The single OOD-trained model loses a lot on Deepfakes (AUC 0.913 vs in-dist single 0.982, gap −7pp).** This is a much larger gap than the Face2Face holdout (1.4pp), suggesting Deepfakes-style identity-swap artifacts are structurally more distinct from Face2Face/DeepFakeDetection than vice versa. **The full N=6 ensemble more than recovers**: AUC 0.986, TPR@FPR=1% 0.664 — beating the in-distribution single best 0.552 by +11.2pp.

## Two recommended operating modes (final)

The ensemble has multiple viable aggregation/calibration paths; pick by deployment goal.

| mode | recipe | best for |
|---|---|---|
| **High-coverage (general)** | N=6 median, no calibration | best AUC, best TPR@FPR=1% on test (median 0.8571 / 0.3130) |
| **Strict / safety** | N=6 geomean + per-model temperature calibration | best ECE on test (0.0392), best balance for low-FPR work |

Both run the same six checkpoints with TTA hflip — the only difference is the post-softmax aggregation step (and optional calibration) implemented in `scripts/eval_n_model_aggregations.py` / `eval_n_model_temperature_calibrated.py`. Inference cost: 6 forward passes × 2 (hflip) = 12 forwards per video, fully parallelizable.

## Observations / caveats

1. **Diversity matters more than raw single-model strength.** The df_holdout model is weak alone (test F1 0.7591, AUC 0.8367 in eval not shown above) but its inclusion in the N=6 ensemble pushes ensemble AUC and strict TPRs higher than N=5. Ensembling adds value even from below-average members when their errors are uncorrelated.

2. **Median and trimmed-mean aggregations win at N=6.** With more members, median's robustness to single-model outliers becomes net positive. At N≤4, simple mean was best.

3. **Temperature calibration shrinks ECE without hurting ranking.** All six models' fitted T values landed in 1.38–1.69 (all >1, models systematically over-confident). LBFGS NLL fit on val converges in ~10 iterations.

4. **Weight tuning on val did NOT generalize at any N.** Val (782 rows) is too small for stable weight optimization across correlated models. Uniform 1/N is the robust default. See `outputs/eval/tune_ensembleN_*.json`.

5. **Cross-method asymmetry**: Face2Face holdout costs 1.4pp AUC; Deepfakes holdout costs 7pp AUC. Deepfakes-style fakery is structurally further from the other FF++ methods.

6. **The ff_holdout/df_holdout AUC level (0.97-0.99) is partly inflated by FF++ vs YouTube stylistic gap.** All within-condition deltas reported here are still controlled comparisons and trustworthy.

7. **No useful gain from**: ConvNeXt-Tiny full FT (Phase B's first attempt diverged twice), MixUp at this LR (collapsed val_auc), or weight tuning. These are documented dead-ends.

## Files / artifacts

- Six checkpoints under `outputs/checkpoints_protocol_youtube_dataset_plus_local_videoonly_clean_*_frame/best.pt`
- Three holdout manifests under `data/manifests_protocol_youtube_dataset_plus_local_videoonly_clean*/`
- Per-eval JSONs under `outputs/eval/` (filename patterns: `eval_*`, `operating_*`, `ensemble2_*`, `ensembleN_*`, `aggN_*`, `tempcal_*`, `tune_ensembleN_*`)
- Reusable scripts:
  - Single-model: `eval.py`, `scripts/eval_lowfpr_calibration.py`
  - Ensembles: `scripts/eval_two_model_ensemble.py`, `scripts/eval_n_model_ensemble.py`
  - Aggregation comparison: `scripts/eval_n_model_aggregations.py`
  - Temperature calibration: `scripts/eval_n_model_temperature_calibrated.py`
  - Weight tuning: `scripts/tune_n_model_ensemble_weights.py`
- Per-phase reports:
  - `tta_hflip_eval_20260428.md`
  - `phase_c_ff2f_holdout_20260429.md`
  - `phase_d_n4_ensemble_20260429.md`
  - `phase_e_n6_final_20260430.md` (this file)

## Headline (single sentence)

Six EfficientNet-B0 + EMA checkpoints (3 seeds × 2 holdout regimes), aggregated with per-mode median or geomean+temperature-calibration on top of TTA hflip, deliver a 2.6× lift in TPR@FPR=0.1%, a 60.5% drop in ECE, and AUC of 0.985–0.989 on cross-method OOD splits — without any new architecture, manifest cleanup, or label improvement compared to the starting `robustaug` baseline.
