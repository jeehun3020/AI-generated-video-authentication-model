from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from iseeyou.constants import TaskSpec
from iseeyou.utils.metrics import compute_classification_metrics


def _amp_enabled(device: torch.device, requested_amp: bool) -> bool:
    return bool(requested_amp and device.type == "cuda")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip_norm: float = 0.0,
) -> float:
    model.train()
    losses = []

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else math.nan


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()

    losses = []
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            losses.append(float(loss.item()))
            y_true_list.append(labels.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())

    if not y_true_list:
        return {"loss": math.nan, "accuracy": math.nan, "f1": math.nan, "auc": math.nan}

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)

    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    return metrics


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    task_spec: TaskSpec,
    training_cfg: dict,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(training_cfg["epochs"])
    monitor_name = training_cfg.get("monitor", "f1")
    amp = _amp_enabled(device, bool(training_cfg.get("amp", False)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_metric = -float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    early_stopping_cfg = training_cfg.get("early_stopping", {})
    patience = int(early_stopping_cfg.get("patience", 0) or 0)
    min_delta = float(early_stopping_cfg.get("min_delta", 0.0) or 0.0)
    stale_epochs = 0
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0) or 0.0)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp=amp,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
        )

        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=task_spec.num_classes,
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        current_metric = float(val_metrics.get(monitor_name, float("nan")))
        is_improved = not math.isnan(current_metric) and current_metric > (best_metric + min_delta)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "task_spec": asdict(task_spec),
            "training_cfg": training_cfg,
        }

        torch.save(checkpoint, output_dir / "last.pt")

        if is_improved:
            best_metric = current_metric
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1

        if patience > 0 and stale_epochs >= patience:
            print(
                f"[INFO] early stopping at epoch={epoch} "
                f"(best_epoch={best_epoch}, best_{monitor_name}={best_metric:.4f})"
            )
            break

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "history_path": str(history_path),
        "checkpoint_dir": str(output_dir),
    }
