#!/usr/bin/env python3
"""Average N frame-classifier checkpoints and report low-FPR + calibration metrics.

Pass --configs (and optionally matching --checkpoints) for any number of models.
All configs must have the same preprocess / image_size; the manifest is taken
from the first config unless --video-manifest is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score, roc_curve
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
    parser.add_argument("--configs", nargs="+", required=True,
                        help="One or more config yaml paths.")
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        help="Optional matching checkpoint paths; defaults to each config's checkpoints_dir/best.pt.")
    parser.add_argument("--weights", nargs="*", default=None,
                        help="Optional matching ensemble weights; default uniform (1/N each).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "ff_holdout", "df_holdout"])
    parser.add_argument("--tta", default="none", choices=["none", "hflip"])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--label", default="ensembleN")
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


def load_model_from_cfg(config: dict, checkpoint: str | None, device: torch.device) -> torch.nn.Module:
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
    model.eval()
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
        mask = (y_score >= lo) & ((y_score <= hi) if i == bins - 1 else (y_score < hi))
        n = int(mask.sum())
        if n == 0:
            continue
        ece += abs(float(np.mean(y_true[mask])) - float(np.mean(y_score[mask]))) * (n / len(y_true))
    return {"ece": float(ece), "brier": float(brier_score_loss(y_true, y_score))}


def report(name: str, probs: np.ndarray, y_true: np.ndarray, bins: int) -> dict:
    pos = probs[:, 1]
    pred = np.argmax(probs, axis=1)
    return {
        "name": name,
        "accuracy": float((pred == y_true).mean()),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "auc": float(roc_auc_score(y_true, pos)),
        "tpr_at_fpr_1pct": tpr_at_fpr(y_true, pos, 0.01),
        "tpr_at_fpr_0_1pct": tpr_at_fpr(y_true, pos, 0.001),
        **calibration(y_true, pos, bins),
    }


def main() -> None:
    args = parse_args()
    cfgs = [load_config(c) for c in args.configs]
    n = len(cfgs)
    if args.checkpoints and len(args.checkpoints) != n:
        raise SystemExit(f"--checkpoints must match --configs count ({n})")
    checkpoints = args.checkpoints if args.checkpoints else [None] * n

    if args.weights:
        if len(args.weights) != n:
            raise SystemExit(f"--weights must match --configs count ({n})")
        weights = np.array([float(w) for w in args.weights], dtype=np.float64)
    else:
        weights = np.full(n, 1.0 / n, dtype=np.float64)
    weights = weights / weights.sum()

    cfg0 = cfgs[0]
    task_spec = build_task_spec(cfg0["task"])
    image_size = cfg0["preprocess"]["image_size"]
    eval_cfg = cfg0.get("evaluation", {})
    input_repr = eval_cfg.get("input_representation", cfg0["training"].get("input_representation", "rgb"))
    manifest = args.video_manifest or cfg0["paths"]["video_manifest_path"]

    dataset = VideoManifestFrameDataset(
        video_manifest_path=manifest,
        task_spec=task_spec,
        split_tags=(args.split,),
        preprocess_cfg=cfg0["preprocess"],
        augmentation_cfg=None,
        train_mode=False,
        transform=build_eval_transform(image_size, input_representation=input_repr),
    )
    if len(dataset) == 0:
        raise SystemExit(f"[ERROR] empty dataset for split={args.split}")

    device = resolve_device(cfg0["training"].get("device", "auto"))
    num_workers = resolve_num_workers(cfg0["training"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=cfg0["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    models = [load_model_from_cfg(cfgs[i], checkpoints[i], device) for i in range(n)]

    y_true_list: list[np.ndarray] = []
    per_model_probs: list[list[np.ndarray]] = [[] for _ in range(n)]
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            y_true_list.append(batch["label"].numpy())
            for i, m in enumerate(models):
                p = predict_with_tta(m, images, args.tta)
                per_model_probs[i].append(p.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    probs_all = [np.concatenate(per_model_probs[i]) for i in range(n)]

    weighted = np.zeros_like(probs_all[0])
    for i, p in enumerate(probs_all):
        weighted += weights[i] * p

    results = {
        "split": args.split,
        "tta": args.tta,
        "manifest": str(manifest),
        "configs": [str(c) for c in args.configs],
        "checkpoints": [str(c) if c else "<config default best.pt>" for c in checkpoints],
        "weights": weights.tolist(),
        "num_frames": int(len(y_true)),
        "per_model": [report(f"model{i+1}", probs_all[i], y_true, args.bins) for i in range(n)],
        "ensemble_weighted": report("ensemble_weighted", weighted, y_true, args.bins),
    }

    print(json.dumps(results, indent=2))
    eval_dir = ensure_dir(cfg0["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = eval_dir / f"ensembleN_{args.label}_{args.split}_tta-{args.tta}_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
