"""Faithfulness metrics: comprehensiveness and sufficiency (implemented).

Definitions (following DeYoung et al., 2020, "ERASER"):

- Comprehensiveness: how much does removing the *top-attributed* tokens
  change the prediction? High = the rationale captures the important parts.

      comp = p(y_hat | x) - p(y_hat | x_without_important_tokens)

- Sufficiency: how much can the *top-attributed tokens alone* retain the
  prediction? High = the rationale is enough on its own.

      suff = p(y_hat | x) - p(y_hat | x_only_important_tokens)

Both take a probability callable rather than a model object, so they work
with sklearn pipelines, HF models, or mocks.
"""

from __future__ import annotations

import numpy as np


def _safe_probs(p: float) -> float:
    return float(min(max(p, 0.0), 1.0))


def comprehensiveness(
    predict_probs: callable[[str], float],
    tokens: list[str],
    attribution: list[float] | np.ndarray,
    k: int = 3,
) -> float:
    """Comprehensiveness of an attribution for P(hateful).

    Removes the ``k`` highest-attributed tokens from the text and measures
    how much the predicted hateful-probability drops. Positive values mean
    the attributed tokens were important.

    Args:
        predict_probs: Callable mapping text -> P(hateful) in [0, 1].
        tokens: Token sequence of the original text.
        attribution: Per-token attribution scores (same length as tokens).
        k: Number of top-attributed tokens to remove.

    Returns:
        ``p_full - p_without_top_k`` in roughly [-1, 1].

    Raises:
        ValueError: On mismatched lengths or ``k`` larger than the tokens.
    """
    if len(attribution) != len(tokens):
        raise ValueError(f"attribution length {len(attribution)} != tokens length {len(tokens)}")
    if k < 0 or k > len(tokens):
        raise ValueError(f"k={k} must be between 0 and len(tokens)={len(tokens)}")

    original = " ".join(tokens)
    p_full = _safe_probs(predict_probs(original))

    order = np.argsort(np.asarray(attribution, dtype=float))[::-1]
    drop = set(int(i) for i in order[:k])
    reduced = " ".join(t for i, t in enumerate(tokens) if i not in drop)
    p_reduced = _safe_probs(predict_probs(reduced)) if reduced else 0.0
    return p_full - p_reduced


def sufficiency(
    predict_probs: callable[[str], float],
    tokens: list[str],
    attribution: list[float] | np.ndarray,
    k: int = 3,
) -> float:
    """Sufficiency of an attribution for P(hateful).

    Keeps only the ``k`` highest-attributed tokens and measures how much of
    the prediction is retained. Values close to 0 mean the kept tokens alone
    reproduce the original prediction.

    Args:
        predict_probs: Callable mapping text -> P(hateful) in [0, 1].
        tokens: Token sequence of the original text.
        attribution: Per-token attribution scores (same length as tokens).
        k: Number of top-attributed tokens to keep.

    Returns:
        ``p_full - p_only_top_k`` in roughly [-1, 1]; lower (closer to 0) is
        more faithful.

    Raises:
        ValueError: On mismatched lengths or invalid ``k``.
    """
    if len(attribution) != len(tokens):
        raise ValueError(f"attribution length {len(attribution)} != tokens length {len(tokens)}")
    if k < 0 or k > len(tokens):
        raise ValueError(f"k={k} must be between 0 and len(tokens)={len(tokens)}")

    original = " ".join(tokens)
    p_full = _safe_probs(predict_probs(original))

    order = np.argsort(np.asarray(attribution, dtype=float))[::-1]
    keep = sorted(int(i) for i in order[:k])
    kept = " ".join(tokens[i] for i in keep)
    p_kept = _safe_probs(predict_probs(kept)) if kept else 0.0
    return p_full - p_kept
