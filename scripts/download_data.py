#!/usr/bin/env python
"""Download datasets for PJ-03.

Usage:
    python scripts/download_data.py [--dataset all|hatexplain|hasoc|hinglish|hatecheck]

Rules of this script:
- Never invents download URLs. Where a public HF ``datasets`` loader exists,
  it is used directly. Where registration / manual download is required,
  clear manual instructions and the expected file paths are printed instead.
- Every download reminds you to verify the dataset licence before use
  (see docs/dataset_notes.md and docs/ethics.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.utils.logging import get_logger  # noqa: E402

logger = get_logger("pj03.download")

LICENCE_REMINDER = (
    "REMINDER: verify each dataset's licence before use and record it in "
    "docs/dataset_notes.md. Some datasets restrict redistribution or "
    "commercial use; this is academic use but we still must comply."
)


def download_hatexplain(out_dir: Path) -> None:
    """Download HateXplain via the public HF hub loader (internet required)."""
    logger.info("Downloading HateXplain from the Hugging Face hub ...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("`datasets` is not installed. Run: pip install -r requirements.txt")
        return
    target = out_dir / "hatexplain"
    target.mkdir(parents=True, exist_ok=True)
    try:
        # HateXplain is a script-based dataset (unsupported since datasets 3.x),
        # so we read the hub's auto-converted parquet files directly.
        base = "hf://datasets/Hate-speech-CNERG/hatexplain@refs/convert/parquet/plain_text"
        dataset = load_dataset(
            "parquet",
            data_files={
                "train": f"{base}/train/0000.parquet",
                "validation": f"{base}/validation/0000.parquet",
                "test": f"{base}/test/0000.parquet",
            },
        )
        for split, part in dataset.items():
            path = target / f"{split}.jsonl"
            part.to_json(path)
            logger.info("Wrote %s (%d rows) -> %s", split, len(part), path)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
        logger.error("HateXplain download failed: %s", exc)
        logger.error(
            "Check the dataset id on huggingface.co/datasets and your network/"
            "HF_TOKEN (see .env.example)."
        )
    print(LICENCE_REMINDER)


def download_hasoc(out_dir: Path) -> None:
    """HASOC requires registration; print manual instructions (no URL guessing)."""
    target = out_dir / "hasoc"
    target.mkdir(parents=True, exist_ok=True)
    print(
        "HASOC (Hindi/English/code-mixed) requires MANUAL download:\n"
        "  1. Register with the HASOC shared-task organisers on their official\n"
        "     website and accept their terms of use. DO NOT ask us for a URL -\n"
        "     get the current one from the organisers or the ACL anthology page\n"
        "     of the HASOC overview papers.\n"
        f"  2. Place the downloaded TSV files under: {target}/<year>/<language>/\n"
        f"     e.g. {target}/2021/hi/train.tsv\n"
        "  3. Record the exact version, files and licence in docs/dataset_notes.md.\n"
    )
    print(LICENCE_REMINDER)


def download_hinglish(out_dir: Path) -> None:
    """Hinglish tweets dataset is not chosen yet; print guidance."""
    target = out_dir / "hinglish_tweets"
    target.mkdir(parents=True, exist_ok=True)
    print(
        "The Hinglish offensive-tweet dataset has NOT been chosen yet.\n"
        "  1. Team decision required - candidates are listed in\n"
        "     docs/dataset_notes.md (all need a licence check; some are gated\n"
        "     on the HF hub or require a request form).\n"
        f"  2. Once chosen, place the raw files under: {target}/\n"
        "  3. Record the choice, version and licence in docs/dataset_notes.md.\n"
    )
    print(LICENCE_REMINDER)


def download_hatecheck(out_dir: Path) -> None:
    """Download HateCheck (public on HF hub; verify licence at download time)."""
    logger.info("Downloading HateCheck ...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("`datasets` is not installed. Run: pip install -r requirements.txt")
        return
    target = out_dir / "hatecheck"
    target.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_dataset("Paul/hatecheck")
        for split, part in dataset.items():
            path = target / f"{split}.jsonl"
            part.to_json(path)
            logger.info("Wrote %s (%d rows) -> %s", split, len(part), path)
    except Exception as exc:  # noqa: BLE001
        logger.error("HateCheck download failed: %s", exc)
        logger.error("Verify the dataset id on huggingface.co/datasets.")
    print(LICENCE_REMINDER)


DOWNLOADERS = {
    "hatexplain": download_hatexplain,
    "hasoc": download_hasoc,
    "hinglish": download_hinglish,
    "hatecheck": download_hatecheck,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DOWNLOADERS, "all"],
        default="all",
        help="Which dataset to fetch (default: all).",
    )
    args = parser.parse_args()

    raw_dir = REPO_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = list(DOWNLOADERS) if args.dataset == "all" else [args.dataset]
    for name in names:
        logger.info("=== %s ===", name)
        DOWNLOADERS[name](raw_dir)


if __name__ == "__main__":
    main()
