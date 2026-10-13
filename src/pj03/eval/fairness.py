"""Fairness / bias audit metrics (implemented): subgroup AUC, BPSN, BNSP.

Reference: Borkan et al., 2019, "Nuanced Metrics for Measuring Unintended
Bias with Real Evaluation Data" (Jigsaw / Civil Comments).

All three metrics use only P(hateful) scores and group membership, and all
follow the "higher is better" convention:

- Subgroup AUC: AUC computed *within* one identity group's examples only.
  Low = the model cannot distinguish hateful vs non-hateful inside the group.
- BPSN (Background Positive, Subgroup Negative): AUC over exactly two sets —
  background positives (hateful, not in group) vs subgroup negatives
  (non-hateful, in group). The positive class is the background positive.
  Low = non-hateful posts that merely *mention* the identity score as high as
  genuinely hateful background posts (identity-term false-positive bias).
- BNSP (Background Negative, Subgroup Positive): AUC over background
  negatives (non-hateful, not in group) vs subgroup positives (hateful, in
  group). The positive class is the subgroup positive. Low = hateful posts
  targeting the group score below ordinary non-hateful posts (missing
  in-group hate).

Each pair metric uses ONLY its two named sets (this is what makes them
independent diagnostics; pooling all four sets would collapse BPSN and BNSP
into complements of each other and destroy their meaning).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

_Score = Sequence[float]


def _pair_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """AUC of P(pos > neg) over exactly two score sets; NaN if a side is empty."""
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return float("nan")
    labels = np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))])
    scores = np.concatenate([positive_scores, negative_scores])
    return float(roc_auc_score(labels, scores))


def _arrays(
    y_true: Sequence[int], scores: _Score, is_group: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    g_arr = np.asarray(is_group, dtype=bool)
    if not (len(y_arr) == len(s_arr) == len(g_arr)):
        raise ValueError("y_true, scores and is_group must have the same length")
    return y_arr, s_arr, g_arr


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
    y_arr, s_arr, g_arr = _arrays(y_true, scores, is_group)
    return _pair_auc(s_arr[g_arr & (y_arr == 1)], s_arr[g_arr & (y_arr == 0)])


def bpsn_auc(
    y_true: Sequence[int],
    scores: _Score,
    is_group: Sequence[bool],
) -> float:
    """BPSN AUC: background positives vs subgroup negatives.

    Positive class = background positives (hateful, not in group);
    negative class = subgroup negatives (non-hateful, in group).
    High = benign identity-mentioning posts are scored below hateful
    background posts (desired).
    """
    y_arr, s_arr, g_arr = _arrays(y_true, scores, is_group)
    return _pair_auc(s_arr[(~g_arr) & (y_arr == 1)], s_arr[g_arr & (y_arr == 0)])


def bnsp_auc(
    y_true: Sequence[int],
    scores: _Score,
    is_group: Sequence[bool],
) -> float:
    """BNSP AUC: subgroup positives vs background negatives.

    Positive class = subgroup positives (hateful, in group); negative class =
    background negatives (non-hateful, not in group). High = hateful posts
    targeting the group are scored above benign background posts (desired).
    """
    y_arr, s_arr, g_arr = _arrays(y_true, scores, is_group)
    return _pair_auc(s_arr[g_arr & (y_arr == 1)], s_arr[(~g_arr) & (y_arr == 0)])


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
        where NaN marks an undefined AUC (a needed set is empty).
    """
    return {
        name: {
            "subgroup_auc": subgroup_auc(y_true, scores, mask),
            "bpsn_auc": bpsn_auc(y_true, scores, mask),
            "bnsp_auc": bnsp_auc(y_true, scores, mask),
        }
        for name, mask in group_masks.items()
    }
