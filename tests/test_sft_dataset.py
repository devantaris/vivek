"""Tests for the SFT dataset builder."""

from __future__ import annotations

import pytest

from pj03.preprocess.schema import Example, Language, Split
from pj03.sft.dataset import build_completion, build_sft_records
from pj03.sft.prompt_templates import parse_output


def make_example(**overrides: object) -> Example:
    defaults: dict = {
        "id": "hatexplain:1",
        "text": "you are terrible people",
        "label": 1,
        "language": Language.EN,
        "source_dataset": "hatexplain",
        "split": Split.TRAIN,
    }
    defaults.update(overrides)
    return Example(**defaults)


def test_completion_with_rationale_mask() -> None:
    ex = make_example(rationale_mask=[0, 1])
    completion = build_completion(ex)
    label, rationale = parse_output(completion)
    assert label == 1
    assert "you are" in rationale


def test_completion_target_group_fallback() -> None:
    ex = make_example(target_group="islam", rationale_mask=None)
    _, rationale = parse_output(build_completion(ex))
    assert "islam" in rationale


def test_completion_non_hateful_generic() -> None:
    ex = make_example(label=0, rationale_mask=None)
    label, rationale = parse_output(build_completion(ex))
    assert label == 0
    assert "does not contain hateful" in rationale


def test_rationale_word_cap() -> None:
    text = " ".join(f"w{i}" for i in range(50))
    ex = make_example(text=text, rationale_mask=list(range(50)))
    _, rationale = parse_output(build_completion(ex))
    quoted = rationale.split(":", 1)[1].split()
    assert len(quoted) <= 20


def test_records_round_trip_and_prompt_embedded() -> None:
    records = build_sft_records(
        [
            make_example(rationale_mask=[0]),
            make_example(id="hatexplain:2", label=0, rationale_mask=None),
        ]
    )
    assert [r["id"] for r in records] == ["hatexplain:1", "hatexplain:2"]
    for rec in records:
        assert "Post: you are terrible people" in rec["prompt"]
        label, _ = parse_output(rec["completion"])
        assert label in (0, 1)


def test_invalid_label_raises() -> None:
    ex = make_example()
    ex.label = 7  # bypass pydantic to simulate corruption
    with pytest.raises(ValueError, match="does not parse back"):
        build_sft_records([ex])
