#!/usr/bin/env python
"""Build processed (unified-schema) splits from raw downloads.

Usage:
    python scripts/build_data.py [--dataset all|hatexplain]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.preprocess import (  # noqa: E402
    loaders_hatecheck,
    loaders_hatexplain,
    loaders_hinglish,
)
from pj03.utils.logging import get_logger  # noqa: E402

logger = get_logger("pj03.build")


def build_hatexplain(mode: str = "strict_hate") -> None:
    raw = REPO_ROOT / "data" / "raw" / "hatexplain"
    out = REPO_ROOT / "data" / "processed" / "hatexplain"
    if not raw.is_dir():
        logger.error("Raw HateXplain not found at %s - run make data first.", raw)
        return
    stats = loaders_hatexplain.build_processed(str(raw), str(out), mode=mode)
    logger.info(
        "HateXplain [%s]: train=%d val=%d test=%d positive=%d negative=%d -> %s",
        mode,
        stats.n_train,
        stats.n_validation,
        stats.n_test,
        stats.n_positive,
        stats.n_negative,
        out,
    )


def build_hatecheck(mode: str = "strict_hate") -> None:
    del mode
    raw = REPO_ROOT / "data" / "raw" / "hatecheck"
    out = REPO_ROOT / "data" / "processed" / "hatecheck"
    if not raw.is_dir():
        logger.error("Raw HateCheck not found at %s - run make data first.", raw)
        return
    stats = loaders_hatecheck.build_processed(str(raw), str(out))
    logger.info(
        "HateCheck: test=%d positive=%d negative=%d -> %s (eval-only, never train)",
        stats.n_test,
        stats.n_positive,
        stats.n_negative,
        out,
    )


def build_hinglish(mode: str = "strict_hate") -> None:
    del mode
    raw = REPO_ROOT / "data" / "raw" / "hinglish"
    out = REPO_ROOT / "data" / "processed" / "hinglish_tweets"
    stats = loaders_hinglish.build_processed(raw, out)
    logger.info(
        "Hinglish: test=%d positive=%d negative=%d -> %s",
        stats.n_test,
        stats.n_positive,
        stats.n_negative,
        out,
    )


BUILDERS = {
    "hatexplain": build_hatexplain,
    "hatecheck": build_hatecheck,
    "hinglish": build_hinglish,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*BUILDERS, "all"], default="all")
    parser.add_argument(
        "--mode",
        choices=["strict_hate", "toxic_abusive"],
        default="strict_hate",
        help="Label mapping mode for HateXplain (strict_hate: 1=hate, 0=off/norm; toxic_abusive: 1=hate+off, 0=norm)",
    )
    args = parser.parse_args()
    names = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    for name in names:
        BUILDERS[name](mode=args.mode)


if __name__ == "__main__":
    main()
