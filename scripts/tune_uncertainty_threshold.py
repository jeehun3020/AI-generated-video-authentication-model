#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import load_config
from iseeyou.constants import build_task_spec
from scripts.tune_ensemble_weights import (
    align_modalities,
    build_frame_video_probs,
    collect_frame_predictions,
    collect_temporal_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune decision threshold and uncertainty band for binary inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--hardneg-config", action="append", default=[])
    parser.add_argument("--frame-weight", type=float, default=0.95)
    parser.add_argument("--temporal-weight", type=float, default=0.05)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def load_main_probs(config_path: str, frame_ckpt: str, temporal_ckpt: str, split: str, fw: float, tw: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = load_config(config_path)
    task_spec = build_task_spec(cfg["task"])
    device = torch.device("cpu")

    y_true_f, y_prob_f, frame_video_ids = collect_frame_predictions(
        config=cfg,
        task_spec=task_spec,
        split=split,
        checkpoint_path=Path(frame_ckpt),
        device=device,
    )
    y_true_frame_video, y_prob_frame_video, frame_video_ids_sorted = build_frame_video_probs(
        config=cfg,
        y_true=y_true_f,
        y_prob=y_prob_f,
        video_ids=frame_video_ids,
    )
    temporal_cfg = load_config("configs/temporal_binary_shorts_holdout.yaml")
    temporal_cfg["paths"]["manifests_dir"] = cfg["paths"]["manifests_dir"]
    y_true_t, y_prob_t, temporal_video_ids = collect_temporal_predictions(
        config=temporal_cfg,
        task_spec=task_spec,
        split=split,
        checkpoint_path=Path(temporal_ckpt),
        device=device,
    )
    y_true, frame_aligned, temporal_aligned, _ = align_modalities(
        frame_true=y_true_frame_video,
        frame_prob=y_prob_frame_video,
        frame_video_ids=frame_video_ids_sorted,
        temporal_true=y_true_t,
        temporal_prob=y_prob_t,
        temporal_video_ids=temporal_video_ids,
    )
    prob = fw * frame_aligned + tw * temporal_aligned
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)
    return y_true.astype(int), prob[:, 1]


def load_hardneg_probs(config_path: str, frame_ckpt: str, temporal_ckpt: str, fw: float, tw: float) -> np.ndarray:
    cfg = load_config(config_path)
    task_spec = build_task_spec(cfg["task"])
    device = torch.device("cpu")
    splits = ["train", "test"]
    true_list = []
    prob_list = []
    for split in splits:
        manifest = Path(cfg["paths"]["manifests_dir"]) / f"{split}.csv"
        if not manifest.exists():
            continue
        y_true_f, y_prob_f, frame_video_ids = collect_frame_predictions(
            config=cfg,
            task_spec=task_spec,
            split=split,
            checkpoint_path=Path(frame_ckpt),
            device=device,
        )
        y_true_frame_video, y_prob_frame_video, frame_video_ids_sorted = build_frame_video_probs(
            config=cfg,
            y_true=y_true_f,
            y_prob=y_prob_f,
            video_ids=frame_video_ids,
        )
        temporal_cfg = load_config("configs/temporal_binary_shorts_holdout.yaml")
        temporal_cfg["paths"]["manifests_dir"] = cfg["paths"]["manifests_dir"]
        y_true_t, y_prob_t, temporal_video_ids = collect_temporal_predictions(
            config=temporal_cfg,
            task_spec=task_spec,
            split=split,
            checkpoint_path=Path(temporal_ckpt),
            device=device,
        )
        y_true, frame_aligned, temporal_aligned, _ = align_modalities(
            frame_true=y_true_frame_video,
            frame_prob=y_prob_frame_video,
            frame_video_ids=frame_video_ids_sorted,
            temporal_true=y_true_t,
            temporal_prob=y_prob_t,
            temporal_video_ids=temporal_video_ids,
        )
        prob = fw * frame_aligned + tw * temporal_aligned
        prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)
        true_list.append(y_true.astype(int))
        prob_list.append(prob[:, 1])
    if not prob_list:
        return np.array([], dtype=np.float64)
    return np.concatenate(prob_list, axis=0)


def main() -> None:
    args = parse_args()
    y_true, fake_prob = load_main_probs(
        config_path=args.config,
        frame_ckpt=args.frame_checkpoint,
        temporal_ckpt=args.temporal_checkpoint,
        split=args.val_split,
        fw=args.frame_weight,
        tw=args.temporal_weight,
    )
    hardneg_probs = [load_hardneg_probs(path, args.frame_checkpoint, args.temporal_checkpoint, args.frame_weight, args.temporal_weight) for path in args.hardneg_config]
    hardneg_probs = [p for p in hardneg_probs if len(p) > 0]
    merged_hardneg = np.concatenate(hardneg_probs, axis=0) if hardneg_probs else np.array([], dtype=np.float64)

    best = None
    trials = []
    for threshold in np.arange(0.45, 0.81, 0.02):
        for band in np.arange(0.0, 0.21, 0.02):
            low = max(0.0, threshold - band)
            high = min(1.0, threshold + band)
            pred = np.full_like(y_true, fill_value=-1)
            pred[fake_prob <= low] = 0
            pred[fake_prob >= high] = 1
            confident_mask = pred >= 0
            coverage = float(np.mean(confident_mask))
            if not np.any(confident_mask):
                continue
            conf_true = y_true[confident_mask]
            conf_pred = pred[confident_mask]
            f1 = f1_binary(conf_true, conf_pred)
            acc = accuracy(conf_true, conf_pred)
            hardneg_fp = float(np.mean(merged_hardneg >= high)) if len(merged_hardneg) else 0.0
            score = (0.55 * f1) + (0.15 * acc) + (0.15 * coverage) - (0.35 * hardneg_fp)
            row = {
                "threshold": round(float(threshold), 4),
                "band": round(float(band), 4),
                "low": round(float(low), 4),
                "high": round(float(high), 4),
                "coverage": round(coverage, 4),
                "f1_confident": round(f1, 4),
                "accuracy_confident": round(acc, 4),
                "hardneg_false_positive_rate": round(hardneg_fp, 4),
                "score": round(score, 4),
            }
            trials.append(row)
            if best is None or row["score"] > best["score"]:
                best = row

    output = {
        "best": best,
        "top_trials": sorted(trials, key=lambda x: x["score"], reverse=True)[:15],
        "num_trials": len(trials),
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"[INFO] saved threshold tuning: {args.output_json}")


if __name__ == "__main__":
    main()
