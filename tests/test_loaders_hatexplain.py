"""Tests for the HateXplain label/rationale mapping helpers."""

from __future__ import annotations

from pj03.preprocess.loaders_hatexplain import (
    CLASS_HATESPEECH,
    CLASS_NORMAL,
    CLASS_OFFENSIVE,
    _majority,
    _majority_rationale_mask,
    _majority_targets,
    _to_binary,
)


def test_majority_clear_vote() -> None:
    assert _majority([CLASS_HATESPEECH, CLASS_HATESPEECH, CLASS_OFFENSIVE]) == CLASS_HATESPEECH
    assert _majority([CLASS_NORMAL, CLASS_NORMAL, CLASS_OFFENSIVE]) == CLASS_NORMAL


def test_majority_tie_resolves_toward_hateful() -> None:
    # 1-1-1 disagreement: hatespeech wins.
    assert _majority([CLASS_NORMAL, CLASS_OFFENSIVE, CLASS_HATESPEECH]) == CLASS_HATESPEECH
    # 1-1 tie between offensive and normal: offensive wins.
    assert _majority([CLASS_NORMAL, CLASS_OFFENSIVE, 99]) == CLASS_OFFENSIVE


def test_to_binary() -> None:
    assert _to_binary(CLASS_HATESPEECH) == 1
    assert _to_binary(CLASS_OFFENSIVE) == 0
    assert _to_binary(CLASS_NORMAL) == 0

    assert _to_binary(CLASS_HATESPEECH, mode="toxic_abusive") == 1
    assert _to_binary(CLASS_OFFENSIVE, mode="toxic_abusive") == 1
    assert _to_binary(CLASS_NORMAL, mode="toxic_abusive") == 0


def test_majority_targets_majority_rule() -> None:
    targets = [["Islam", "Hindu"], ["Islam"], ["Islam", "Other"]]
    assert _majority_targets(targets) == "islam"
    assert _majority_targets([["Other"], ["Other"], ["Women"]]) is None
    assert _majority_targets([]) is None


def test_majority_rationale_mask_two_of_three() -> None:
    rationales = [[1, 0, 1], [1, 0, 0], [1, 1, 0]]
    assert _majority_rationale_mask(rationales) == [0]
    assert _majority_rationale_mask([]) is None
