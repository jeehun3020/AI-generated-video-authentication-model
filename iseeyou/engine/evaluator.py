from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from iseeyou.constants import TaskSpec
from iseeyou.models.builder import build_model

from .trainer import evaluate_loader


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    backbone: str,
    num_classes: int,
    dropout: float,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=False,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def run_evaluation(
    checkpoint_path: str | Path,
    backbone: str,
    dropout: float,
    task_spec: TaskSpec,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model, _ = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        num_classes=task_spec.num_classes,
        dropout=dropout,
        device=device,
    )

    return evaluate_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        num_classes=task_spec.num_classes,
    )
