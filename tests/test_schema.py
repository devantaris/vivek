"""Smoke tests for the unified schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pj03.preprocess.schema import DatasetStats, Example, Language, Split


def make_example(**overrides: object) -> Example:
    defaults: dict = {
        "id": "hatexplain:123",
        "text": "this is a test post",
        "label": 0,
        "language": Language.EN,
        "source_dataset": "hatexplain",
        "split": Split.TRAIN,
    }
    defaults.update(overrides)
    return Example(**defaults)


def test_valid_example() -> None:
    ex = make_example(rationale_mask=[0, 2], target_group="women")
    assert ex.label == 0
    assert ex.rationale_mask == [0, 2]
    assert ex.split is Split.TRAIN


def test_label_must_be_binary() -> None:
    with pytest.raises(ValidationError, match="label must be 0 or 1"):
        make_example(label=2)
    with pytest.raises(ValidationError):
        make_example(label=-1)


def test_rationale_mask_nonnegative() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        make_example(rationale_mask=[-1])


def test_required_fields_nonempty() -> None:
    with pytest.raises(ValidationError):
        make_example(id="")


def test_dataset_stats_total() -> None:
    stats = DatasetStats(source_dataset="x", n_train=10, n_validation=2, n_test=3)
    assert stats.total == 15
