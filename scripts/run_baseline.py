#!/usr/bin/env python
"""Run Phase 1 baselines: TF-IDF+LogReg and encoder fine-tuning.

Usage:
    python scripts/run_baseline.py [--config configs/baselines.yaml] [key=value ...]
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

logger = get_logger("pj03.baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. baselines.tfidf_lr.C=3.0")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    logger.info("Baseline training is not implemented yet (Phase 1 TODO).")
    logger.info("Loaded config from %s with %d override(s).", args.config, len(args.overrides))
    raise NotImplementedError(
        "TODO(Phase 1): wire loaders -> tfidf_lr / encoder_finetune here. "
        "The config pipeline works; see src/pj03/baselines/."
    )


if __name__ == "__main__":
    main()
