"""HateCheck loader (functional tests for hate-speech detectors).

Raw layout (from ``scripts/download_data.py``, HF hub ``Paul/hatecheck``):
``data/raw/hatecheck/test.jsonl`` with columns ``case_id``, ``test_case``,
``label_gold`` (hateful / non-hateful), ``target_ident``, ``functionality``.

All 3,728 cases go to the TEST split — HateCheck is an evaluation-only
resource and must never be trained on (recorded in docs/dataset_notes.md).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pj03.preprocess.schema import DatasetStats, Example, Language, Split

SOURCE = "hatecheck"


def build_processed(raw_dir: str, out_dir: str) -> DatasetStats:
    """Convert raw HateCheck JSONL to the unified schema (test split only).

    Writes ``test.jsonl`` under ``out_dir`` and returns stats.
    """
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = raw / "test.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found - run make data (hatecheck) first.")
    examples: list[Example] = []
    with source.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            examples.append(
                Example(
                    id=f"{SOURCE}:{row['case_id']}",
                    text=row["test_case"].strip(),
                    label=1 if row["label_gold"] == "hateful" else 0,
                    language=Language.EN,
                    source_dataset=SOURCE,
                    split=Split.TEST,
                    target_group=(
                        row["target_ident"].lower().replace(" people", "")
                        if row.get("target_ident")
                        else None
                    ),
                    functionality=row.get("functionality"),
                )
            )
    path = out / "test.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")
    return DatasetStats(
        source_dataset=SOURCE,
        n_test=len(examples),
        n_positive=sum(e.label for e in examples),
        n_negative=sum(1 - e.label for e in examples),
    )


def iter_examples(
    processed_dir: str = "data/processed/hatecheck",
) -> Iterator[Example]:
    """Yield every HateCheck case as an :class:`Example` (test split only)."""
    path = Path(processed_dir) / "test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found - run scripts/build_data.py --dataset hatecheck."
        )
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield Example.model_validate_json(line)
