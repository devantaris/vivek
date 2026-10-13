#!/usr/bin/env python
"""Run Phase 3: reward-optimised post-training (GRPO/DPO).

Usage:
    python scripts/run_reward.py [--config configs/reward.yaml] [key=value ...]
    e.g. run_reward.py reward.weights.beta=0.7 reward.weights.gamma=0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.utils.config import load_config  # noqa: E402
from pj03.utils.logging import get_logger  # noqa: E402
from pj03.utils.seed import set_seed  # noqa: E402

logger = get_logger("pj03.reward")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reward.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. reward.weights.beta=0.7")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    weights = config.get("reward", {}).get("weights", {})
    logger.info("Reward weights: %s", weights)
    logger.info("GRPO/DPO post-training is not implemented yet (Phase 3 TODO).")
    raise NotImplementedError(
        "TODO(Phase 3): call pj03.reward.train_grpo.run_grpo with the SFT "
        "checkpoint and these reward weights."
    )


if __name__ == "__main__":
    main()
