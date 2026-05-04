#!/usr/bin/env python3
"""Average two frame-classifier checkpoints and report low-FPR + calibration metrics.

Both configs must point at the same video manifest / preprocess settings so the
dataloader iterates the same samples in the same order.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import build_task_spec
from iseeyou.data.protocol_dataset import VideoManifestFrameDataset
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.evaluator import load_model_from_checkpoint
from iseeyou.utils.dataloader import resolve_num_workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config1", required=True)
    parser.add_argument("--checkpoint1", default="")
    parser.add_argument("--config2", required=True)
    parser.add_argument("--checkpoint2", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "ff_holdout", "df_holdout"])
    parser.add_argument("--tta", default="none", choices=["none", "hflip"])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--label", default="ensemble2")
    parser.add_argument("--video-manifest", default="", help="Override video_manifest_path from configs.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_from_cfg(config: dict, checkpoint: str, device: torch.device) -> torch.nn.Module:
    training_cfg = config["training"]
    task_spec = build_task_spec(config["task"])
    ckpt_path = Path(checkpoint) if checkpoint else Path(config["paths"]["checkpoints_dir"]) / "best.pt"
    model, _ = load_model_from_checkpoint(
        checkpoint_path=ckpt_path,
        backbone=training_cfg["backbone"],
        num_classes=task_spec.num_classes,
        dropout=training_cfg.get("dropout", 0.0),
        freeze_backbone=bool(training_cfg.get("freeze_backbone", False)),
        hidden_dim=int(training_cfg.get("hidden_dim", 0) or 0),
        device=device,
    )
    return model


def predict_with_tta(model: torch.nn.Module, images: torch.Tensor, tta: str) -> torch.Tensor:
    probs = torch.softmax(model(images), dim=1)
    if tta == "hflip":
        probs = (probs + torch.softmax(model(torch.flip(images, dims=[-1])), dim=1)) * 0.5
    return probs


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = np.where(fpr <= target)[0]
    return float(np.max(tpr[eligible])) if len(eligible) else 0.0


def calibration(y_true: np.ndarray, y_score: np.ndarray, bins: int) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & (y_score <= hi if i == bins - 1 else y_score < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        ece += abs(float(np.mean(y_true[mask])) - float(np.mean(y_score[mask]))) * (n / len(y_true))
    return {"ece": float(ece), "brier": float(brier_score_loss(y_true, y_score))}


def main() -> None:
    args = parse_args()
    cfg1 = load_config(args.config1)
    cfg2 = load_config(args.config2)
    task_spec = build_task_spec(cfg1["task"])

    image_size = cfg1["preprocess"]["image_size"]
    eval_cfg = cfg1.get("evaluation", {})
    input_repr = eval_cfg.get("input_representation", cfg1["training"].get("input_representation", "rgb"))

    manifest = args.video_manifest or cfg1["paths"]["video_manifest_path"]
    if not args.video_manifest and manifest != cfg2["paths"]["video_manifest_path"]:
        raise SystemExit("[ERROR] config1 and config2 must point at the same video_manifest_path; pass --video-manifest to override")

    dataset = VideoManifestFrameDataset(
        video_manifest_path=manifest,
        task_spec=task_spec,
        split_tags=(args.split,),
        preprocess_cfg=cfg1["preprocess"],
        augmentation_cfg=None,
        train_mode=False,
        transform=build_eval_transform(image_size, input_representation=input_repr),
    )
    if len(dataset) == 0:
        raise SystemExit(f"[ERROR] empty dataset for split={args.split}")

    device = resolve_device(cfg1["training"].get("device", "auto"))
    num_workers = resolve_num_workers(cfg1["training"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=cfg1["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model1 = load_model_from_cfg(cfg1, args.checkpoint1, device)
    model2 = load_model_from_cfg(cfg2, args.checkpoint2, device)
    model1.eval()
    model2.eval()

    y_true_list, p1_list, p2_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            p1 = predict_with_tta(model1, images, args.tta)
            p2 = predict_with_tta(model2, images, args.tta)
            y_true_list.append(labels.numpy())
            p1_list.append(p1.cpu().numpy())
            p2_list.append(p2.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    p1 = np.concatenate(p1_list)
    p2 = np.concatenate(p2_list)
    avg = (p1 + p2) * 0.5

    def report(name: str, probs: np.ndarray) -> dict:
        pos = probs[:, 1]
        pred = np.argmax(probs, axis=1)
        out = {
            "name": name,
            "accuracy": float((pred == y_true).mean()),
            "f1_macro": float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y_true, pred, average="macro")),
            "auc": float(roc_auc_score(y_true, pos)),
            "tpr_at_fpr_1pct": tpr_at_fpr(y_true, pos, 0.01),
            "tpr_at_fpr_0_1pct": tpr_at_fpr(y_true, pos, 0.001),
            **calibration(y_true, pos, args.bins),
        }
        return out

    results = {
        "split": args.split,
        "tta": args.tta,
        "config1": str(args.config1),
        "config2": str(args.config2),
        "num_frames": int(len(y_true)),
        "model1": report("model1", p1),
        "model2": report("model2", p2),
        "ensemble_avg": report("ensemble_avg", avg),
    }

    print(json.dumps(results, indent=2))
    eval_dir = ensure_dir(cfg1["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = eval_dir / f"ensemble2_{args.label}_{args.split}_tta-{args.tta}_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
