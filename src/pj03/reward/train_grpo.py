"""GRPO (or DPO) post-training with the composite reward (Phase 3).

TODO(Phase 3): implement with HF ``trl.GRPOTrainer`` (preferred) falling back
to ``DPOTrainer`` if GRPO is unstable on the chosen 1-3B model:

- Reward function: :func:`pj03.reward.composite.batch_rewards` with weights
  from ``configs/reward.yaml``.
- GRPO works directly with a reward function over generations; for DPO,
  build preference pairs per prompt from sampled generations by composite
  reward ranking.
- num_generations=8 per prompt, KL coef ~0.04, lr ~1e-6..5e-6 (starting
  points - tune on validation).
- GPU required; keep imports CPU-safe via lazy import.
"""

from __future__ import annotations

from typing import Any


def run_grpo(
    model_name: str,
    train_dataset: Any,
    output_dir: str,
    reward_config: dict[str, Any] | None = None,
    num_generations: int = 8,
    epochs: int = 1,
    seed: int = 42,
) -> Any:
    """Run GRPO post-training with the composite reward.

    Args:
        model_name: HF id of the SFT checkpoint (adapter) from Phase 2.
        train_dataset: Prompts (from the unified schema train split).
        output_dir: Output dir for the trained adapter (gitignored).
        reward_config: Loaded config dict; weights at ``reward.weights``.
        num_generations: Samples per prompt for group-relative advantage.
        epochs: Training epochs over the prompt set.
        seed: Random seed.

    Raises:
        NotImplementedError: Until Phase 3 is implemented.
    """
    raise NotImplementedError(
        "TODO(Phase 3): implement trl.GRPOTrainer with composite reward "
        "(see module docstring). Requires GPU; scaffold only."
    )
