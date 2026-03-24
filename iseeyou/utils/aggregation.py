from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def aggregate_probs(
    probs: np.ndarray,
    method: str = "mean",
    topk_ratio: float = 0.5,
    conf_power: float = 2.0,
) -> np.ndarray:
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("probs must be a non-empty 2D array")

    if method == "vote":
        votes = np.argmax(probs, axis=1)
        counts = np.bincount(votes, minlength=probs.shape[1]).astype(np.float64)
        return counts / max(1.0, counts.sum())

    if method == "confidence_mean":
        conf = np.max(probs, axis=1)
        weights = np.power(np.clip(conf, 1e-8, 1.0), conf_power)
        weights = weights / max(1e-8, weights.sum())
        return np.sum(probs * weights[:, None], axis=0)

    if method == "topk_mean":
        conf = np.max(probs, axis=1)
        k = max(1, int(math.ceil(len(conf) * topk_ratio)))
        selected_idx = np.argsort(conf)[-k:]
        return probs[selected_idx].mean(axis=0)

    # default: mean
    return probs.mean(axis=0)


def build_video_level_predictions(
    video_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "mean",
    topk_ratio: float = 0.5,
    conf_power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    bucket: dict[str, dict[str, list]] = defaultdict(lambda: {"labels": [], "probs": []})

    for vid, label, prob in zip(video_ids, y_true, y_prob):
        bucket[vid]["labels"].append(int(label))
        bucket[vid]["probs"].append(prob)

    y_true_video = []
    y_prob_video = []

    for vid in sorted(bucket.keys()):
        labels = np.array(bucket[vid]["labels"], dtype=np.int64)
        probs = np.array(bucket[vid]["probs"], dtype=np.float64)

        # Most common frame label as video GT fallback when noisy manifests are mixed.
        values, counts = np.unique(labels, return_counts=True)
        y_true_video.append(int(values[np.argmax(counts)]))

        agg = aggregate_probs(
            probs=probs,
            method=method,
            topk_ratio=topk_ratio,
            conf_power=conf_power,
        )
        y_prob_video.append(agg)

    return np.array(y_true_video), np.array(y_prob_video)
