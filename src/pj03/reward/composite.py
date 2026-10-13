"""Composite reward R = alpha*R_label + beta*R_faith - gamma*R_bias.

Weights come from ``configs/reward.yaml`` (``reward.weights.{alpha,beta,gamma}``)
and can be overridden per run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pj03.reward.r_bias import r_bias
from pj03.reward.r_faith import r_faith
from pj03.reward.r_label import r_label

PredictProbs = Callable[[str], float]

DEFAULT_WEIGHTS: Mapping[str, float] = {"alpha": 1.0, "beta": 0.5, "gamma": 0.3}


def composite_reward(
    generation: str,
    gold_label: int,
    weights: Mapping[str, float] | None = None,
    predict_probs: PredictProbs | None = None,
) -> float:
    """Compute the composite reward for one generation.

    R = alpha*R_label + beta*R_faith - gamma*R_bias

    Args:
        generation: Raw model output text.
        gold_label: Ground-truth binary label (0/1).
        weights: Mapping with keys ``alpha``/``beta``/``gamma``; missing keys
            fall back to :data:`DEFAULT_WEIGHTS`.
        predict_probs: Optional model probability callable for R_faith.

    Returns:
        The scalar composite reward (may be negative when the bias penalty
        outweighs the positive terms).
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for key in ("alpha", "beta", "gamma"):
            if key in weights:
                w[key] = float(weights[key])
    return (
        w["alpha"] * r_label(generation, gold_label)
        + w["beta"] * r_faith(generation, gold_label, predict_probs)
        - w["gamma"] * r_bias(generation)
    )


def batch_rewards(
    generations: list[str],
    gold_labels: list[int],
    config: Mapping[str, Any] | None = None,
    predict_probs: PredictProbs | None = None,
) -> list[float]:
    """Composite rewards for a batch, weights read from a loaded config dict.

    Args:
        generations: Model outputs.
        gold_labels: Ground-truth labels, same length as ``generations``.
        config: Full PJ-03 config dict (e.g. from ``load_config``); the
            weights are read from ``reward.weights`` if present.
        predict_probs: Optional model probability callable.

    Raises:
        ValueError: If ``generations`` and ``gold_labels`` lengths differ.
    """
    if len(generations) != len(gold_labels):
        raise ValueError(
            f"generations ({len(generations)}) and gold_labels "
            f"({len(gold_labels)}) must have the same length"
        )
    weights = (config or {}).get("reward", {}).get("weights", None)
    return [
        composite_reward(gen, gold, weights, predict_probs)
        for gen, gold in zip(generations, gold_labels, strict=False)
    ]
