"""Classification metrics (implemented)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Macro-averaged F1 across the two classes.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Macro-F1 in [0, 1].

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) lengths differ"
        )
    if len(y_true) == 0:
        raise ValueError("metric inputs are empty")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
) -> dict[str, float]:
    """Standard headline metrics for the paper's results tables.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_score: Optional P(hateful) scores for AUC; None skips AUC.

    Returns:
        Dict with ``macro_f1``, ``f1_hateful``, ``precision``, ``recall``,
        ``accuracy``, and optionally ``auc``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) lengths differ"
        )
    metrics = {
        "macro_f1": macro_f1(y_true, y_pred),
        "f1_hateful": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_score is not None:
        y_true_arr = np.asarray(y_true)
        y_score_arr = np.asarray(y_score, dtype=float)
        if len(y_true_arr) == len(y_score_arr) and len(np.unique(y_true_arr)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
    return metrics
