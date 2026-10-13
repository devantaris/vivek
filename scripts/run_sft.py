#!/usr/bin/env python
"""Run Phase 2: LoRA SFT of a small instruction-tuned LLM.

Usage:
    python scripts/run_sft.py [--config configs/sft.yaml] [key=value ...]

Requires a GPU in practice; the script itself must stay importable on CPU.
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

logger = get_logger("pj03.sft")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. sft.epochs=5")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    logger.info("SFT is not implemented yet (Phase 2 TODO).")
    raise NotImplementedError(
        "TODO(Phase 2): build prompt/completion pairs from the unified schema "
        "and call pj03.sft.lora_sft.run_lora_sft."
    )


if __name__ == "__main__":
    main()
