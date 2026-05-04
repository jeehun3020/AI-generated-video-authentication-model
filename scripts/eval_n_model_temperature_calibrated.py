#!/usr/bin/env python3
"""Per-model temperature scaling for N frame-classifier checkpoints.

For each model:
  1. Forward on val (with TTA hflip averaging at logit level), collect averaged logits.
  2. Fit a single temperature T_i on val by minimizing NLL (LBFGS).
  3. Forward on the apply split with calibrated logits = avg_logits / T_i.
  4. Report per-model metrics with/without calibration, plus mean and geomean
     ensemble metrics with/without per-model calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    p.add_argument("--tune-split", default="val")
    p.add_argument("--apply-splits", nargs="+", default=["test"])
    p.add_argument("--apply-manifests", nargs="*", default=None)
    p.add_argument("--tune-manifest", default="")
    p.add_argument("--tta", default="hflip", choices=["none", "hflip"])
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--label", default="tempscale")
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


def collect_logits(models, cfg0, manifest, split, tta, device, num_workers) -> tuple[np.ndarray, list[np.ndarray]]:
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
    y_list = []
    per_model: list[list[np.ndarray]] = [[] for _ in range(n)]
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            y_list.append(batch["label"].numpy())
            for i, m in enumerate(models):
                logits = m(images)
                if tta == "hflip":
                    logits_flip = m(torch.flip(images, dims=[-1]))
                    logits = (logits + logits_flip) * 0.5
                per_model[i].append(logits.cpu().numpy())
    y_true = np.concatenate(y_list)
    logits_per_model = [np.concatenate(per_model[i]) for i in range(n)]
    return y_true, logits_per_model


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a scalar temperature T (>0) by LBFGS minimizing NLL on (logits/T) vs y_true."""
    device = torch.device("cpu")
    z = torch.tensor(logits, dtype=torch.float64, device=device)
    y = torch.tensor(y_true, dtype=torch.long, device=device)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(z / T, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_T).detach().item())
    return T


def softmax_np(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)


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


def aggregate_mean(probs_stack: np.ndarray) -> np.ndarray:
    return probs_stack.mean(axis=0)


def aggregate_geomean(probs_stack: np.ndarray) -> np.ndarray:
    eps = 1e-9
    log_p = np.log(np.clip(probs_stack, eps, 1.0)).mean(axis=0)
    out = np.exp(log_p)
    out = out / np.clip(out.sum(axis=1, keepdims=True), eps, None)
    return out


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
    models = [load_model(cfgs[i], ckpts[i], device) for i in range(n)]

    tune_manifest = args.tune_manifest or cfg0["paths"]["video_manifest_path"]
    print(f"[INFO] tuning T on split={args.tune_split} manifest={tune_manifest}")
    y_val, logits_val = collect_logits(models, cfg0, tune_manifest, args.tune_split, args.tta, device, num_workers)

    Ts = []
    for i in range(n):
        T = fit_temperature(logits_val[i], y_val)
        Ts.append(T)
        print(f"[INFO] model{i+1} T={T:.4f}")

    apply_manifests = args.apply_manifests if args.apply_manifests else [None] * len(args.apply_splits)
    if len(apply_manifests) != len(args.apply_splits):
        raise SystemExit("--apply-manifests must match --apply-splits if provided")

    apply_results = []
    for split, manifest_override in zip(args.apply_splits, apply_manifests):
        manifest = manifest_override or cfg0["paths"]["video_manifest_path"]
        print(f"[INFO] applying T to split={split} manifest={manifest}")
        y_apply, logits_apply = collect_logits(models, cfg0, manifest, split, args.tta, device, num_workers)

        per_model_uncal = []
        per_model_cal = []
        for i in range(n):
            probs_uncal = softmax_np(logits_apply[i])
            probs_cal = softmax_np(logits_apply[i] / Ts[i])
            per_model_uncal.append(metrics_from_probs(y_apply, probs_uncal, args.bins))
            per_model_cal.append(metrics_from_probs(y_apply, probs_cal, args.bins))

        # Build stacks for ensembling
        stack_uncal = np.stack([softmax_np(logits_apply[i]) for i in range(n)], axis=0)
        stack_cal = np.stack([softmax_np(logits_apply[i] / Ts[i]) for i in range(n)], axis=0)

        ens_uncal = {
            "mean": metrics_from_probs(y_apply, aggregate_mean(stack_uncal), args.bins),
            "geomean": metrics_from_probs(y_apply, aggregate_geomean(stack_uncal), args.bins),
        }
        ens_cal = {
            "mean": metrics_from_probs(y_apply, aggregate_mean(stack_cal), args.bins),
            "geomean": metrics_from_probs(y_apply, aggregate_geomean(stack_cal), args.bins),
        }

        apply_results.append({
            "split": split,
            "manifest": str(manifest),
            "num_frames": int(len(y_apply)),
            "per_model_uncalibrated": per_model_uncal,
            "per_model_calibrated": per_model_cal,
            "ensemble_uncalibrated": ens_uncal,
            "ensemble_calibrated": ens_cal,
        })

    payload = {
        "configs": [str(c) for c in args.configs],
        "checkpoints": [str(c) if c else "<best.pt>" for c in ckpts],
        "tta": args.tta,
        "tune_split": args.tune_split,
        "tune_manifest": str(tune_manifest),
        "temperatures": Ts,
        "apply": apply_results,
    }

    eval_dir = ensure_dir(cfg0["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = eval_dir / f"tempcal_{args.label}_{timestamp}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[INFO] saved: {out}")


if __name__ == "__main__":
    main()
