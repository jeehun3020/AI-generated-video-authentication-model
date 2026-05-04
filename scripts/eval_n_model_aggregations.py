#!/usr/bin/env python3
"""Compare ensemble aggregation strategies (arithmetic mean, geometric mean / logit avg,
median, max, min) for N frame-classifier checkpoints.

Reuses the same per-model forward passes once and applies several aggregations on top.
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
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--checkpoints", nargs="*", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "ff_holdout", "df_holdout"])
    p.add_argument("--video-manifest", default="")
    p.add_argument("--tta", default="hflip", choices=["none", "hflip"])
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--label", default="aggregations")
    return p.parse_args()


def resolve_device(req: str) -> torch.device:
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(config: dict, checkpoint: str | None, device: torch.device) -> torch.nn.Module:
    training_cfg = config["training"]
    task_spec = build_task_spec(config["task"])
    ckpt = Path(checkpoint) if checkpoint else Path(config["paths"]["checkpoints_dir"]) / "best.pt"
    m, _ = load_model_from_checkpoint(
        checkpoint_path=ckpt,
        backbone=training_cfg["backbone"],
        num_classes=task_spec.num_classes,
        dropout=training_cfg.get("dropout", 0.0),
        freeze_backbone=bool(training_cfg.get("freeze_backbone", False)),
        hidden_dim=int(training_cfg.get("hidden_dim", 0) or 0),
        device=device,
    )
    m.eval()
    return m


def predict_with_tta(model, images, tta):
    p = torch.softmax(model(images), dim=1)
    if tta == "hflip":
        p = (p + torch.softmax(model(torch.flip(images, dims=[-1])), dim=1)) * 0.5
    return p


def tpr_at_fpr(y_true, y_score, target):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    elig = np.where(fpr <= target)[0]
    return float(np.max(tpr[elig])) if len(elig) else 0.0


def calibration(y_true, y_score, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & ((y_score <= hi) if i == bins - 1 else (y_score < hi))
        nm = int(mask.sum())
        if nm == 0:
            continue
        ece += abs(float(np.mean(y_true[mask])) - float(np.mean(y_score[mask]))) * (nm / len(y_true))
    return {"ece": float(ece), "brier": float(brier_score_loss(y_true, y_score))}


def metrics_from_probs(y_true, probs2d, bins):
    pos = probs2d[:, 1]
    pred = np.argmax(probs2d, axis=1)
    return {
        "accuracy": float((pred == y_true).mean()),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "auc": float(roc_auc_score(y_true, pos)),
        "tpr_at_fpr_1pct": tpr_at_fpr(y_true, pos, 0.01),
        "tpr_at_fpr_0_1pct": tpr_at_fpr(y_true, pos, 0.001),
        **calibration(y_true, pos, bins),
    }


def aggregate(probs_stack: np.ndarray, method: str) -> np.ndarray:
    """probs_stack shape: [N_models, N_samples, 2]. Returns [N_samples, 2]."""
    eps = 1e-9
    if method == "mean":
        return probs_stack.mean(axis=0)
    if method == "geomean":
        # average in log space, exponentiate, renormalize per row
        log_p = np.log(np.clip(probs_stack, eps, 1.0)).mean(axis=0)
        out = np.exp(log_p)
        out = out / np.clip(out.sum(axis=1, keepdims=True), eps, None)
        return out
    if method == "median":
        m = np.median(probs_stack, axis=0)
        m = m / np.clip(m.sum(axis=1, keepdims=True), eps, None)
        return m
    if method == "max_pos":
        # per-sample, take the model's prob vector with the most extreme p_pos
        idx = np.argmax(probs_stack[:, :, 1], axis=0)
        return probs_stack[idx, np.arange(probs_stack.shape[1])]
    if method == "min_pos":
        idx = np.argmin(probs_stack[:, :, 1], axis=0)
        return probs_stack[idx, np.arange(probs_stack.shape[1])]
    if method == "trimmed_mean":
        # drop the highest and lowest p_pos per sample, average the rest
        if probs_stack.shape[0] < 3:
            return probs_stack.mean(axis=0)
        sorted_idx = np.argsort(probs_stack[:, :, 1], axis=0)
        keep = sorted_idx[1:-1, :]  # drop top and bottom
        out = np.take_along_axis(probs_stack, keep[..., None], axis=0).mean(axis=0)
        return out
    raise ValueError(f"unknown aggregation {method}")


def main():
    args = parse_args()
    cfgs = [load_config(c) for c in args.configs]
    n = len(cfgs)
    if args.checkpoints and len(args.checkpoints) != n:
        raise SystemExit(f"--checkpoints count must match --configs ({n})")
    ckpts = args.checkpoints if args.checkpoints else [None] * n

    cfg0 = cfgs[0]
    device = resolve_device(cfg0["training"].get("device", "auto"))
    num_workers = resolve_num_workers(cfg0["training"]["num_workers"])
    models = [load_model(cfgs[i], ckpts[i], device) for i in range(n)]

    manifest = args.video_manifest or cfg0["paths"]["video_manifest_path"]
    task_spec = build_task_spec(cfg0["task"])
    image_size = cfg0["preprocess"]["image_size"]
    eval_cfg = cfg0.get("evaluation", {})
    input_repr = eval_cfg.get("input_representation", cfg0["training"].get("input_representation", "rgb"))
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
    loader = DataLoader(
        dataset,
        batch_size=cfg0["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_list = []
    per_model: list[list[np.ndarray]] = [[] for _ in range(n)]
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            y_list.append(batch["label"].numpy())
            for i, m in enumerate(models):
                per_model[i].append(predict_with_tta(m, images, args.tta).cpu().numpy())
    y_true = np.concatenate(y_list)
    probs_stack = np.stack([np.concatenate(per_model[i]) for i in range(n)], axis=0)

    aggregations = ["mean", "geomean", "median", "max_pos", "min_pos"]
    if n >= 3:
        aggregations.append("trimmed_mean")

    results = {
        "split": args.split,
        "tta": args.tta,
        "manifest": str(manifest),
        "configs": [str(c) for c in args.configs],
        "checkpoints": [str(c) if c else "<best.pt>" for c in ckpts],
        "num_frames": int(len(y_true)),
        "per_model": [metrics_from_probs(y_true, probs_stack[i], args.bins) for i in range(n)],
        "aggregations": {},
    }

    for method in aggregations:
        agg_probs = aggregate(probs_stack, method)
        results["aggregations"][method] = metrics_from_probs(y_true, agg_probs, args.bins)

    eval_dir = ensure_dir(cfg0["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = eval_dir / f"aggN_{args.label}_{args.split}_tta-{args.tta}_{timestamp}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[INFO] saved: {out}")


if __name__ == "__main__":
    main()
