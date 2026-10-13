"""Faithfulness reward: is the stated rationale actually faithful to the model?

TODO(Phase 3): plan - score the rationale by how much the model's own
prediction depends on it:

1. Tokenise the ``Rationale:`` text; compute the model's label probability.
2. Remove/mask the rationale-named words from the post and recompute the
   probability; reward = |p_full - p_masked| (prediction change), or use the
   comprehensiveness/sufficiency functions from :mod:`pj03.xai.faithfulness`
   with the rationale as an "attribution mask".
3. Cache base probabilities per (post, generation) to keep GRPO affordable.

Keep the contract: return a float in [0, 1].
"""

from __future__ import annotations


def r_faith(
    generation: str,
    gold_label: int,
    predict_probs: callable[[str], float] | None = None,
) -> float:
    """Reward for rationale faithfulness.

    Args:
        generation: Raw model output text (must contain the rationale).
        gold_label: Ground-truth label (context for masking experiments).
        predict_probs: Optional callable mapping a text to the model's
            P(hateful). If None (no model available, e.g. tests), returns
            the neutral placeholder 0.5.

    Returns:
        Float in [0, 1]: how faithfully the rationale explains the decision.
    """
    # TODO(Phase 3): implement masking-based faithfulness; placeholder keeps
    # the composite reward computable without a model in the loop.
    _ = generation, gold_label, predict_probs
    return 0.5
