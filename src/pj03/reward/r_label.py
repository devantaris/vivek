"""Label reward: does the generation contain the correct binary label?

TODO(Phase 3): currently a faithful placeholder based on parsing the
``Label:`` line out of the generation. After Phase 2 lands, revisit whether
partial credit (e.g. parseable-but-wrong > unparseable) helps GRPO stability;
keep the contract: return a float in [0, 1].
"""

from __future__ import annotations

from pj03.sft.prompt_templates import parse_output


def r_label(generation: str, gold_label: int) -> float:
    """Reward for label correctness.

    Args:
        generation: Raw model output text.
        gold_label: Ground-truth binary label (0/1).

    Returns:
        1.0 if the parsed label matches gold, 0.0 otherwise (unparseable
        outputs score 0).
    """
    parsed, _ = parse_output(generation)
    if parsed is None:
        return 0.0
    return 1.0 if parsed == int(gold_label) else 0.0
