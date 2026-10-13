"""Tests for the HateCheck loader."""

from __future__ import annotations

import json

from pj03.preprocess.loaders_hatecheck import build_processed, iter_examples
from pj03.preprocess.schema import Split


def _write_raw(path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "test.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_build_and_iter(tmp_path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    _write_raw(
        raw,
        [
            {
                "case_id": 1,
                "test_case": "I hate women.",
                "label_gold": "hateful",
                "target_ident": "women",
                "functionality": "derog_neg_emote_h",
            },
            {
                "case_id": 2,
                "test_case": "I love women.",
                "label_gold": "non-hateful",
                "target_ident": "black people",
                "functionality": "negation_non_hateful",
            },
            {
                "case_id": 3,
                "test_case": "just a sentence",
                "label_gold": "non-hateful",
                "target_ident": None,
                "functionality": "slur_non_hateful",
            },
        ],
    )
    stats = build_processed(str(raw), str(out))
    assert stats.n_test == 3 and stats.n_positive == 1

    cases = list(iter_examples(str(out)))
    assert all(c.split is Split.TEST for c in cases)  # eval-only, always TEST
    by_id = {c.id: c for c in cases}
    assert by_id["hatecheck:1"].label == 1
    assert by_id["hatecheck:1"].functionality == "derog_neg_emote_h"
    assert by_id["hatecheck:1"].target_group == "women"
    assert by_id["hatecheck:2"].target_group == "black"  # ' people' stripped
    assert by_id["hatecheck:3"].target_group is None


def test_missing_raw_raises(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError, match="run make data"):
        build_processed(str(tmp_path / "raw"), str(tmp_path / "out"))
