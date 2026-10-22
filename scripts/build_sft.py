#!/usr/bin/env python
"""Build the Phase-2 SFT dataset (prompt/completion JSONL) from processed splits.

Usage:
    python scripts/build_sft.py            # HateXplain train+validation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.preprocess import loaders_hatexplain  # noqa: E402
from pj03.preprocess.schema import Split  # noqa: E402
from pj03.sft.dataset import build_sft_records  # noqa: E402
from pj03.utils.logging import get_logger  # noqa: E402

logger = get_logger("pj03.build_sft")


def main() -> None:
    out_dir = REPO_ROOT / "data" / "processed" / "sft_hatexplain"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in (Split.TRAIN, Split.VALIDATION):
        examples = list(loaders_hatexplain.iter_examples(split))
        records = build_sft_records(examples)
        path = out_dir / f"{split.value}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_rationale = sum(1 for ex in examples if ex.rationale_mask)
        logger.info(
            "%s: %d records (%d with annotator rationales) -> %s",
            split.value,
            len(records),
            n_rationale,
            path,
        )


if __name__ == "__main__":
    main()
