#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Evaluate real hard-negative set using processed manifests")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--frame-checkpoint", type=str, required=True)
    parser.add_argument("--temporal-checkpoint", type=str, required=True)
    parser.add_argument("--frame-weight", type=float, default=0.95)
    parser.add_argument("--temporal-weight", type=float, default=0.05)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def load_source_map(manifests_dir: Path, splits: list[str]) -> dict[str, str]:
    source_by_video: dict[str, str] = {}
    for split in splits:
        manifest_path = manifests_dir / f"{split}.csv"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_by_video.setdefault(row["video_id"], row.get("source_id", ""))
    return source_by_video


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    task_spec = build_task_spec(cfg["task"])
    device = torch.device("cpu")

    all_true = []
    all_frame_prob = []
    all_temporal_prob = []
    all_video_ids = []

    for split in args.splits:
        y_true_f, y_prob_f, frame_video_ids = collect_frame_predictions(
            config=cfg,
            task_spec=task_spec,
            split=split,
            checkpoint_path=Path(args.frame_checkpoint),
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
            checkpoint_path=Path(args.temporal_checkpoint),
            device=device,
        )

        y_true, frame_aligned, temporal_aligned, common_ids = align_modalities(
            frame_true=y_true_frame_video,
            frame_prob=y_prob_frame_video,
            frame_video_ids=frame_video_ids_sorted,
            temporal_true=y_true_t,
            temporal_prob=y_prob_t,
            temporal_video_ids=temporal_video_ids,
        )

        all_true.append(y_true)
        all_frame_prob.append(frame_aligned)
        all_temporal_prob.append(temporal_aligned)
        all_video_ids.extend(common_ids)

    y_true = np.concatenate(all_true, axis=0)
    frame_prob = np.concatenate(all_frame_prob, axis=0)
    temporal_prob = np.concatenate(all_temporal_prob, axis=0)

    ensemble_prob = args.frame_weight * frame_prob + args.temporal_weight * temporal_prob
    ensemble_prob = ensemble_prob / np.clip(ensemble_prob.sum(axis=1, keepdims=True), 1e-8, None)
    pred = np.argmax(ensemble_prob, axis=1)

    manifests_dir = Path(cfg["paths"]["manifests_dir"])
    source_map = load_source_map(manifests_dir, args.splits)

    by_source: dict[str, dict[str, float | int]] = {}
    for video_id, pred_idx, probs in zip(all_video_ids, pred, ensemble_prob):
        source_id = source_map.get(video_id, "unknown")
        bucket = by_source.setdefault(source_id, {"total": 0, "real": 0, "generated": 0})
        bucket["total"] += 1
        bucket["generated" if int(pred_idx) == 1 else "real"] += 1

    false_positive_rows = []
    for video_id, pred_idx, probs, frame_probs, temporal_probs in zip(
        all_video_ids, pred, ensemble_prob, frame_prob, temporal_prob
    ):
        if int(pred_idx) == 1:
            false_positive_rows.append(
                {
                    "video_id": video_id,
                    "source_id": source_map.get(video_id, "unknown"),
                    "real_prob": float(probs[0]),
                    "generated_prob": float(probs[1]),
                    "frame_generated_prob": float(frame_probs[1]),
                    "temporal_generated_prob": float(temporal_probs[1]),
                }
            )

    false_positive_rows.sort(key=lambda x: x["generated_prob"], reverse=True)

    total = int(len(all_video_ids))
    real_count = int(np.sum(pred == 0))
    generated_count = int(np.sum(pred == 1))
    summary = {
        "expected_label": "real",
        "num_videos": total,
        "frame_weight": args.frame_weight,
        "temporal_weight": args.temporal_weight,
        "real_hit_rate": real_count / total if total else None,
        "false_positive_rate": generated_count / total if total else None,
        "prediction_counts": {
            "real": real_count,
            "generated": generated_count,
        },
        "by_source": by_source,
        "top_false_positives": false_positive_rows[:15],
    }

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[INFO] saved hard-negative summary: {out_path}")


if __name__ == "__main__":
    main()
