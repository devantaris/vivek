"""Unit tests for rationale evaluation suite (IoU, F1, ROUGE)."""

from __future__ import annotations

from pj03.eval.rationale_eval import (
    compute_rouge_l,
    compute_rouge_n,
    evaluate_rationale_suite,
    token_iou,
    token_precision_recall_f1,
)


def test_token_iou_exact_match() -> None:
    tokens = ["kill", "all", "traitors"]
    assert token_iou(tokens, tokens) == 1.0


def test_token_iou_partial_match() -> None:
    pred = ["kill", "traitors"]
    gold = ["kill", "all", "traitors"]
    # Intersection = 2, Union = 3 -> 2/3
    assert abs(token_iou(pred, gold) - 2 / 3) < 1e-4


def test_token_iou_disjoint() -> None:
    pred = ["good", "morning"]
    gold = ["bad", "night"]
    assert token_iou(pred, gold) == 0.0


def test_token_precision_recall_f1() -> None:
    pred = ["kill", "traitors", "now"]
    gold = ["kill", "traitors"]
    # Overlap = 2. Precision = 2/3, Recall = 2/2 = 1.0. F1 = 4/5 = 0.8
    p, r, f1 = token_precision_recall_f1(pred, gold)
    assert abs(p - 2 / 3) < 1e-4
    assert r == 1.0
    assert abs(f1 - 0.8) < 1e-4


def test_compute_rouge() -> None:
    hyp = "the post contains hateful threats against women"
    ref = "the post contains hateful threats against immigrants"
    r1 = compute_rouge_n(hyp, ref, n=1)
    assert r1 > 0.7
    rl = compute_rouge_l(hyp, ref)
    assert rl > 0.7


def test_evaluate_rationale_suite() -> None:
    hyps = ["hateful slurs detected", "benign greeting"]
    refs = ["hateful slurs used here", "benign greeting today"]
    pred_spans = [["slurs"], []]
    gold_spans = [["slurs"], []]
    metrics = evaluate_rationale_suite(hyps, refs, pred_spans=pred_spans, gold_spans=gold_spans)
    assert "rouge_1" in metrics
    assert "rouge_l" in metrics
    assert "mean_token_iou" in metrics
    assert metrics["mean_token_iou"] == 1.0
