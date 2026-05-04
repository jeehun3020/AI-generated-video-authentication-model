#!/usr/bin/env python3
"""Tune linear ensemble weights for N frame-classifier checkpoints.

Runs each model once on the val split with TTA hflip, caches per-frame probs,
then grid-searches weights on the simplex to maximize a target metric.
Reports the chosen weights and re-applies them on test (and optionally ff_holdout).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import product
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
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--target-metric", default="auc",
                        choices=["auc", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_0_1pct", "neg_ece"])
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-splits", nargs="+", default=["test"])
    parser.add_argument("--apply-manifests", nargs="*", default=None,
                        help="Optional: per-apply-split manifest override (same length as --apply-splits).")
    parser.add_argument("--tune-manifest", default="")
    parser.add_argument("--grid-step", type=float, default=0.05,
                        help="Resolution of the simplex grid (e.g. 0.05 → 21 ticks per axis).")
    parser.add_argument("--tta", default="hflip", choices=["none", "hflip"])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--label", default="ensembleN_tuned")
    return parser.parse_args()


def resolve_device(req: str) -> torch.device:
    if req != "auto":
        return torch.device(req)
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


def collect_probs(models, cfg0, manifest, split, tta, device, num_workers) -> tuple[np.ndarray, list[np.ndarray]]:
    task_spec = build_task_spec(cfg0["task"])
    image_size = cfg0["preprocess"]["image_size"]
    eval_cfg = cfg0.get("evaluation", {})
    input_repr = eval_cfg.get("input_representation", cfg0["training"].get("input_representation", "rgb"))
    dataset = VideoManifestFrameDataset(
        video_manifest_path=manifest,
        task_spec=task_spec,
        split_tags=(split,),
        preprocess_cfg=cfg0["preprocess"],
        augmentation_cfg=None,
        train_mode=False,
        transform=build_eval_transform(image_size, input_representation=input_repr),
    )
    if len(dataset) == 0:
        raise SystemExit(f"[ERROR] empty dataset for split={split}")
    loader = DataLoader(
        dataset,
        batch_size=cfg0["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    n = len(models)
    y_true_list = []
    per_model: list[list[np.ndarray]] = [[] for _ in range(n)]
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            y_true_list.append(batch["label"].numpy())
            for i, m in enumerate(models):
                per_model[i].append(predict_with_tta(m, images, tta).cpu().numpy())
    y_true = np.concatenate(y_true_list)
    probs = [np.concatenate(per_model[i]) for i in range(n)]
    return y_true, probs


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


def metrics_for(y_true, probs_combined, bins):
    pos = probs_combined[:, 1]
    pred = np.argmax(probs_combined, axis=1)
    return {
        "accuracy": float((pred == y_true).mean()),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "auc": float(roc_auc_score(y_true, pos)),
        "tpr_at_fpr_1pct": tpr_at_fpr(y_true, pos, 0.01),
        "tpr_at_fpr_0_1pct": tpr_at_fpr(y_true, pos, 0.001),
        **calibration(y_true, pos, bins),
    }


def simplex_weights(n: int, step: float):
    """Yield all weight vectors on the n-simplex with given grid resolution."""
    ticks = int(round(1.0 / step))
    for combo in product(range(ticks + 1), repeat=n - 1):
        s = sum(combo)
        if s > ticks:
            continue
        last = ticks - s
        w = np.array([c / ticks for c in combo] + [last / ticks], dtype=np.float64)
        yield w


def score_weights(y_true, probs_list, w, bins, target):
    combined = np.zeros_like(probs_list[0])
    for i, p in enumerate(probs_list):
        combined += w[i] * p
    m = metrics_for(y_true, combined, bins)
    if target == "neg_ece":
        return -m["ece"]
    if target == "f1":
        return m["f1_macro"]
    return m[target]


def main() -> None:
    args = parse_args()
    cfgs = [load_config(c) for c in args.configs]
    n = len(cfgs)
    if args.checkpoints and len(args.checkpoints) != n:
        raise SystemExit(f"--checkpoints must match --configs count ({n})")
    checkpoints = args.checkpoints if args.checkpoints else [None] * n

    cfg0 = cfgs[0]
    device = resolve_device(cfg0["training"].get("device", "auto"))
    num_workers = resolve_num_workers(cfg0["training"]["num_workers"])
    models = [load_model_from_cfg(cfgs[i], checkpoints[i], device) for i in range(n)]

    tune_manifest = args.tune_manifest or cfg0["paths"]["video_manifest_path"]
    print(f"[INFO] tuning on split={args.tune_split} manifest={tune_manifest} target={args.target_metric}")
    y_val, probs_val = collect_probs(models, cfg0, tune_manifest, args.tune_split, args.tta, device, num_workers)

    best = {"score": -np.inf, "weights": None}
    for w in simplex_weights(n, args.grid_step):
        s = score_weights(y_val, probs_val, w, args.bins, args.target_metric)
        if s > best["score"]:
            best = {"score": float(s), "weights": w.tolist()}

    print(f"[INFO] best on {args.tune_split}: weights={best['weights']} target_metric={best['score']:.4f}")

    val_uniform = score_weights(y_val, probs_val, np.full(n, 1.0 / n), args.bins, args.target_metric)
    print(f"[INFO] uniform 1/N on {args.tune_split}: target={val_uniform:.4f}")

    val_combined = np.zeros_like(probs_val[0])
    for i, p in enumerate(probs_val):
        val_combined += best["weights"][i] * p
    val_metrics = metrics_for(y_val, val_combined, args.bins)
    val_per_model = [metrics_for(y_val, probs_val[i], args.bins) for i in range(n)]

    apply_manifests = args.apply_manifests if args.apply_manifests else [None] * len(args.apply_splits)
    if len(apply_manifests) != len(args.apply_splits):
        raise SystemExit("--apply-manifests must match --apply-splits if provided")

    apply_results = []
    for split, manifest_override in zip(args.apply_splits, apply_manifests):
        manifest = manifest_override or cfg0["paths"]["video_manifest_path"]
        print(f"[INFO] applying weights to split={split} manifest={manifest}")
        y_apply, probs_apply = collect_probs(models, cfg0, manifest, split, args.tta, device, num_workers)
        combined = np.zeros_like(probs_apply[0])
        for i, p in enumerate(probs_apply):
            combined += best["weights"][i] * p
        per_model = [metrics_for(y_apply, probs_apply[i], args.bins) for i in range(n)]
        uniform_combined = np.zeros_like(probs_apply[0])
        for p in probs_apply:
            uniform_combined += (1.0 / n) * p
        uniform_metrics = metrics_for(y_apply, uniform_combined, args.bins)
        tuned_metrics = metrics_for(y_apply, combined, args.bins)
        apply_results.append({
            "split": split,
            "manifest": str(manifest),
            "num_frames": int(len(y_apply)),
            "per_model": per_model,
            "uniform": uniform_metrics,
            "tuned": tuned_metrics,
        })

    payload = {
        "configs": [str(c) for c in args.configs],
        "checkpoints": [str(c) if c else "<best.pt>" for c in checkpoints],
        "tta": args.tta,
        "target_metric": args.target_metric,
        "tune_split": args.tune_split,
        "tune_manifest": str(tune_manifest),
        "best_weights": best["weights"],
        "tune_target_value": best["score"],
        "tune_uniform_value": val_uniform,
        "val_tuned_metrics": val_metrics,
        "val_per_model": val_per_model,
        "apply": apply_results,
    }

    eval_dir = ensure_dir(cfg0["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = eval_dir / f"tune_ensembleN_{args.label}_{args.target_metric}_{timestamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[INFO] saved: {out}")


if __name__ == "__main__":
    main()
