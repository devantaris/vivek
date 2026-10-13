#!/usr/bin/env python
"""Run Phase 4 evaluation: metrics, XAI faithfulness, bias audit, HateCheck, OOD.

Usage:
    python scripts/run_eval.py [--config configs/eval.yaml] [key=value ...]
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

logger = get_logger("pj03.eval")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. faithfulness.k=5")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    logger.info("Evaluation pipeline is not implemented yet (Phase 4 TODO).")
    raise NotImplementedError(
        "TODO(Phase 4): run pj03.eval.metrics / fairness / hatecheck / ood and "
        "pj03.xai over the trained checkpoints."
    )


if __name__ == "__main__":
    main()
