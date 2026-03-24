from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    y_pred = np.argmax(y_prob, axis=1)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    try:
        if num_classes == 2:
            # y_prob shape: [N, 2]
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auc"] = float(
                roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
            )
    except ValueError:
        # TODO: log this as warning when class coverage in validation/test is incomplete.
        metrics["auc"] = float("nan")

    return metrics
