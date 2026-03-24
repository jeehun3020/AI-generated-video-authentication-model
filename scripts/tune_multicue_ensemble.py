#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import build_task_spec
from iseeyou.utils.metrics import compute_classification_metrics
from scripts.tune_ensemble_weights import (
    align_modalities,
    build_frame_video_probs,
    collect_frame_predictions,
    collect_temporal_predictions,
)


@dataclass
class ComponentSpec:
    name: str
    kind: str
    config_path: str
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune multi-cue ensemble weights across arbitrary frame/temporal components.")
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        help="Component spec: name:kind:config:checkpoint where kind is frame or temporal",
    )
    parser.add_argument("--val-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--test-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=400, help="Number of random Dirichlet weight samples")
    parser.add_argument(
        "--monitor",
        default="f1",
        choices=["f1", "auc", "accuracy"],
        help="Validation metric to optimize",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def parse_component(spec: str) -> ComponentSpec:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid --component spec: {spec}")
    name, kind, config_path, checkpoint_path = parts
    kind = kind.strip().lower()
    if kind not in {"frame", "temporal"}:
        raise ValueError(f"Unsupported component kind: {kind}")
    return ComponentSpec(name=name.strip(), kind=kind, config_path=config_path.strip(), checkpoint_path=checkpoint_path.strip())


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_component_predictions(
    component: ComponentSpec,
    split: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cfg = load_config(component.config_path)
    task_spec = build_task_spec(cfg["task"])
    ckpt = Path(component.checkpoint_path)

    if component.kind == "frame":
        y_true_f, y_prob_f, frame_ids = collect_frame_predictions(
            config=cfg,
            task_spec=task_spec,
            split=split,
            checkpoint_path=ckpt,
            device=device,
        )
        y_true_v, y_prob_v, video_ids = build_frame_video_probs(
            config=cfg,
            y_true=y_true_f,
            y_prob=y_prob_f,
            video_ids=frame_ids,
        )
        return y_true_v, y_prob_v, video_ids

    y_true_t, y_prob_t, video_ids_t = collect_temporal_predictions(
        config=cfg,
        task_spec=task_spec,
        split=split,
        checkpoint_path=ckpt,
        device=device,
    )
    return y_true_t, y_prob_t, video_ids_t


def align_all_components(
    base_true: np.ndarray,
    base_prob: np.ndarray,
    base_ids: list[str],
    others: list[tuple[str, np.ndarray, np.ndarray, list[str]]],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    current_true = base_true
    current_prob = {"base": base_prob}
    current_ids = base_ids

    for name, y_true, y_prob, ids in others:
        aligned_true, base_aligned, other_aligned, common_ids = align_modalities(
            frame_true=current_true,
            frame_prob=current_prob["base"],
            frame_video_ids=current_ids,
            temporal_true=y_true,
            temporal_prob=y_prob,
            temporal_video_ids=ids,
        )
        for key, probs in list(current_prob.items()):
            if key == "base":
                continue
            mapping = {vid: prob for vid, prob in zip(current_ids, probs)}
            current_prob[key] = np.array([mapping[vid] for vid in common_ids], dtype=np.float64)
        current_true = aligned_true
        current_prob["base"] = base_aligned
        current_prob[name] = other_aligned
        current_ids = common_ids

    return current_true, current_prob, current_ids


def sample_weight_vectors(num_components: int, samples: int, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    yield np.full(shape=(num_components,), fill_value=1.0 / num_components, dtype=np.float64)
    for idx in range(num_components):
        one_hot = np.zeros(num_components, dtype=np.float64)
        one_hot[idx] = 1.0
        yield one_hot
    for _ in range(max(0, samples)):
        yield rng.dirichlet(np.ones(num_components, dtype=np.float64))


def ensemble_probs(weight_vec: np.ndarray, component_probs: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(component_probs[0], dtype=np.float64)
    for w, probs in zip(weight_vec, component_probs):
        total += float(w) * probs
    total = total / np.clip(total.sum(axis=1, keepdims=True), 1e-8, None)
    return total


def evaluate_component_set(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> dict:
    return compute_classification_metrics(y_true=y_true, y_prob=probs, num_classes=num_classes)


def main() -> None:
    args = parse_args()
    components = [parse_component(spec) for spec in args.component]
    if len(components) < 2:
        raise SystemExit("Provide at least two --component entries")

    base_cfg = load_config(components[0].config_path)
    task_spec = build_task_spec(base_cfg["task"])
    device = resolve_device(base_cfg.get("training", {}).get("device", "auto"))

    val_collected = []
    test_collected = []
    for comp in components:
        y_true_val, y_prob_val, ids_val = collect_component_predictions(comp, args.val_split, device)
        y_true_test, y_prob_test, ids_test = collect_component_predictions(comp, args.test_split, device)
        val_collected.append((comp.name, y_true_val, y_prob_val, ids_val))
        test_collected.append((comp.name, y_true_test, y_prob_test, ids_test))

    base_name, base_true_val, base_prob_val, base_ids_val = val_collected[0]
    y_true_val, aligned_val_map, common_val_ids = align_all_components(
        base_true_val,
        base_prob_val,
        base_ids_val,
        [(name, yt, yp, ids) for name, yt, yp, ids in val_collected[1:]],
    )
    aligned_val_map[base_name] = aligned_val_map.pop("base")

    base_name_test, base_true_test, base_prob_test, base_ids_test = test_collected[0]
    y_true_test, aligned_test_map, common_test_ids = align_all_components(
        base_true_test,
        base_prob_test,
        base_ids_test,
        [(name, yt, yp, ids) for name, yt, yp, ids in test_collected[1:]],
    )
    aligned_test_map[base_name_test] = aligned_test_map.pop("base")

    ordered_names = [comp.name for comp in components]
    val_prob_list = [aligned_val_map[name] for name in ordered_names]
    test_prob_list = [aligned_test_map[name] for name in ordered_names]

    best = None
    trials = []
    for weights in sample_weight_vectors(len(ordered_names), args.samples, args.seed):
        val_probs = ensemble_probs(weights, val_prob_list)
        val_metrics = evaluate_component_set(y_true_val, val_probs, task_spec.num_classes)
        test_probs = ensemble_probs(weights, test_prob_list)
        test_metrics = evaluate_component_set(y_true_test, test_probs, task_spec.num_classes)
        row = {
            "weights": {name: round(float(w), 4) for name, w in zip(ordered_names, weights)},
            "val": val_metrics,
            "test": test_metrics,
        }
        trials.append(row)
        metric = float(val_metrics[args.monitor])
        if best is None or metric > float(best["val"][args.monitor]):
            best = row

    output = {
        "components": [
            {
                "name": comp.name,
                "kind": comp.kind,
                "config": comp.config_path,
                "checkpoint": comp.checkpoint_path,
            }
            for comp in components
        ],
        "common_val_videos": len(common_val_ids),
        "common_test_videos": len(common_test_ids),
        "best": best,
        "top_trials": sorted(trials, key=lambda x: float(x["val"][args.monitor]), reverse=True)[:20],
    }

    output_path = (
        Path(args.output_json)
        if args.output_json
        else ensure_dir(base_cfg["paths"].get("eval_dir", "outputs/eval"))
        / f"multicue_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["best"], indent=2))
    print(f"[INFO] saved multicue tuning: {output_path}")


if __name__ == "__main__":
    main()
