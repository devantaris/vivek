"""Fairness / bias audit metrics (implemented): subgroup AUC, BPSN, BNSP.

References: Borkan et al., 2019, "Nuanced Metrics for Measuring Unintended
Bias with Real Evaluation Data" (Jigsaw / Civil Comments).

All three metrics use only P(hateful) scores and group membership:

- Subgroup AUC: AUC within the examples of one identity group. Low = the
  model cannot distinguish hateful vs non-hateful *within* the group.
- BPSN (Background Positive, Subgroup Negative): AUC between the combined
  negatives of the subgroup and positives of the background. Low = the model
  scores non-hateful in-group posts high (over-predicting on the identity).
- BNSP (Background Negative, Subgroup Positive): AUC between the combined
  negatives of the background and positives of the subgroup. Low = hateful
  in-group posts are scored low (missing in-group hate).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

_Score = Sequence[float]


def _pair_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC for a binary-labeled pair set; returns NaN if only one class present."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def subgroup_auc(
    y_true: Sequence[int],
    scores: _Score,
    is_group: Sequence[bool],
) -> float:
    """AUC computed only within the given identity group.

    Args:
        y_true: Binary labels over the full evaluation set.
        scores: P(hateful) over the full set.
        is_group: Boolean mask, True where the example belongs to the group.

    Returns:
        Subgroup AUC in [0, 1], or NaN if the subgroup lacks both classes.
    """
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    g_arr = np.asarray(is_group, dtype=bool)
    if not (len(y_arr) == len(s_arr) == len(g_arr)):
        raise ValueError("y_true, scores and is_group must have the same length")
    return _pair_auc(y_arr[g_arr], s_arr[g_arr])


def bpsn_auc(
    y_true: Sequence[int],
    scores: _Score,
    is_group: Sequence[bool],
) -> float:
    """BPSN AUC: subgroup negatives vs background positives.

    Low BPSN means non-hateful posts mentioning the identity group are being
    scored as hateful (false positives on the group).

    Returns:
        AUC in [0, 1], or NaN if a side is empty or single-class.
    """
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    g_arr = np.asarray(is_group, dtype=bool)
    if not (len(y_arr) == len(s_arr) == len(g_arr)):
        raise ValueError("y_true, scores and is_group must have the same length")
    # 0 = subgroup non-hateful, 1 = background hateful.
    labels = np.where(g_arr, 1 - y_arr, y_arr)
    return _pair_auc(labels, s_arr)


def bnsp_auc(
    y_true: Sequence[int],
    scores: _Score,
    is_group: Sequence[bool],
) -> float:
    """BNSP AUC: background negatives vs subgroup positives.

    Low BNSP means hateful posts targeting the group are being scored as
    non-hateful (false negatives on the group).

    Returns:
        AUC in [0, 1], or NaN if a side is empty or single-class.
    """
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    g_arr = np.asarray(is_group, dtype=bool)
    if not (len(y_arr) == len(s_arr) == len(g_arr)):
        raise ValueError("y_true, scores and is_group must have the same length")
    # 0 = background non-hateful, 1 = subgroup hateful.
    labels = np.where(g_arr, y_arr, 1 - y_arr)
    return _pair_auc(labels, s_arr)


def bias_audit(
    y_true: Sequence[int],
    scores: _Score,
    group_masks: dict[str, Sequence[bool]],
) -> dict[str, dict[str, float]]:
    """Run the full per-group audit and return a table-friendly dict.

    Args:
        y_true: Binary labels over the evaluation set.
        scores: P(hateful) over the set.
        group_masks: Mapping group name -> boolean mask of membership.

    Returns:
        ``{group_name: {"subgroup_auc": ..., "bpsn_auc": ..., "bnsp_auc": ...}}``
        where NaN marks an undefined AUC (group too small / single-class).
    """
    return {
        name: {
            "subgroup_auc": subgroup_auc(y_true, scores, mask),
            "bpsn_auc": bpsn_auc(y_true, scores, mask),
            "bnsp_auc": bnsp_auc(y_true, scores, mask),
        }
        for name, mask in group_masks.items()
    }
