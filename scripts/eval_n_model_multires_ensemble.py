#!/usr/bin/env python3
"""Ensemble of N frame-classifier checkpoints that may use *different* preprocess
settings (e.g., one model at image_size=224, another at 320).

For each config we build its own dataset / dataloader / transform pipeline using
that config's `preprocess` block; with shuffle=False and identical split filtering
the per-frame ordering is the same across configs, so we can align probs by index.
Aggregations available: mean, geomean (logit-avg), median, trimmed_mean.
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
    p.add_argument("--video-manifest", default="", help="Override manifest used by ALL configs.")
    p.add_argument("--tta", default="hflip", choices=["none", "hflip"])
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--label", default="multires")
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


def collect_probs_one(config: dict, model: torch.nn.Module, manifest: str, split: str,
                       tta: str, device: torch.device, num_workers: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    task_spec = build_task_spec(config["task"])
    image_size = config["preprocess"]["image_size"]
    eval_cfg = config.get("evaluation", {})
    input_repr = eval_cfg.get("input_representation", config["training"].get("input_representation", "rgb"))
    dataset = VideoManifestFrameDataset(
        video_manifest_path=manifest,
        task_spec=task_spec,
        split_tags=(split,),
        preprocess_cfg=config["preprocess"],
        augmentation_cfg=None,
        train_mode=False,
        transform=build_eval_transform(image_size, input_representation=input_repr),
    )
    if len(dataset) == 0:
        raise SystemExit(f"[ERROR] empty dataset for split={split} (config image_size={image_size})")
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    y_list, p_list, vid_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            y_list.append(batch["label"].numpy())
            vid_list.extend(list(batch["video_id"]))
            p = torch.softmax(model(images), dim=1)
            if tta == "hflip":
                p = (p + torch.softmax(model(torch.flip(images, dims=[-1])), dim=1)) * 0.5
            p_list.append(p.cpu().numpy())
    return np.concatenate(y_list), np.concatenate(p_list), vid_list


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


def aggregate(stack: np.ndarray, method: str) -> np.ndarray:
    eps = 1e-9
    if method == "mean":
        return stack.mean(axis=0)
    if method == "geomean":
        log_p = np.log(np.clip(stack, eps, 1.0)).mean(axis=0)
        out = np.exp(log_p)
        return out / np.clip(out.sum(axis=1, keepdims=True), eps, None)
    if method == "median":
        m = np.median(stack, axis=0)
        return m / np.clip(m.sum(axis=1, keepdims=True), eps, None)
    if method == "trimmed_mean":
        if stack.shape[0] < 3:
            return stack.mean(axis=0)
        sorted_idx = np.argsort(stack[:, :, 1], axis=0)
        keep = sorted_idx[1:-1, :]
        return np.take_along_axis(stack, keep[..., None], axis=0).mean(axis=0)
    raise ValueError(method)


def main() -> None:
    args = parse_args()
    cfgs = [load_config(c) for c in args.configs]
    n = len(cfgs)
    if args.checkpoints and len(args.checkpoints) != n:
        raise SystemExit(f"--checkpoints must match --configs count ({n})")
    ckpts = args.checkpoints if args.checkpoints else [None] * n

    cfg0 = cfgs[0]
    device = resolve_device(cfg0["training"].get("device", "auto"))
    num_workers = resolve_num_workers(cfg0["training"]["num_workers"])
    manifest = args.video_manifest or cfg0["paths"]["video_manifest_path"]

    models = [load_model(cfgs[i], ckpts[i], device) for i in range(n)]

    y_ref = None
    vid_ref = None
    probs_per_model = []
    for i, (cfg, m) in enumerate(zip(cfgs, models)):
        img_size = cfg["preprocess"]["image_size"]
        print(f"[INFO] forwarding model{i+1} (image_size={img_size})...")
        y, p, vid = collect_probs_one(cfg, m, manifest, args.split, args.tta, device, num_workers)
        if y_ref is None:
            y_ref = y
            vid_ref = vid
        else:
            if len(y) != len(y_ref):
                raise SystemExit(f"[ERROR] model{i+1} returned {len(y)} samples, expected {len(y_ref)}")
            if not np.array_equal(y, y_ref):
                raise SystemExit(f"[ERROR] model{i+1} label order mismatch")
            if vid != vid_ref:
                raise SystemExit(f"[ERROR] model{i+1} video_id order mismatch")
        probs_per_model.append(p)

    stack = np.stack(probs_per_model, axis=0)

    aggregations = ["mean", "geomean", "median"]
    if n >= 3:
        aggregations.append("trimmed_mean")

    out = {
        "split": args.split,
        "tta": args.tta,
        "manifest": str(manifest),
        "configs": [str(c) for c in args.configs],
        "checkpoints": [str(c) if c else "<best.pt>" for c in ckpts],
        "image_sizes": [c["preprocess"]["image_size"] for c in cfgs],
        "num_frames": int(len(y_ref)),
        "per_model": [metrics_from_probs(y_ref, probs_per_model[i], args.bins) for i in range(n)],
        "aggregations": {agg: metrics_from_probs(y_ref, aggregate(stack, agg), args.bins)
                         for agg in aggregations},
    }

    eval_dir = ensure_dir(cfg0["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = eval_dir / f"multires_{args.label}_{args.split}_tta-{args.tta}_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    main()
