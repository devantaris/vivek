"""Evaluation metrics for generated and extracted rationales.

Computes:
  - Token-level Intersection-over-Union (IoU) against human annotation masks.
  - Token-level Precision, Recall, and F1.
  - Word-overlap ROUGE-1, ROUGE-2, and ROUGE-L scores for natural language explanations.
"""

from __future__ import annotations

import re


def token_iou(pred_tokens: set[str] | list[str], gold_tokens: set[str] | list[str]) -> float:
    """Compute Intersection-over-Union (Jaccard Index) between two token sets.

    Args:
        pred_tokens: Predicted rationale tokens.
        gold_tokens: Ground-truth human annotator rationale tokens.

    Returns:
        IoU in [0.0, 1.0]. Returns 1.0 if both sets are empty, 0.0 if one is empty.
    """
    s_pred = {t.lower() for t in pred_tokens if t}
    s_gold = {t.lower() for t in gold_tokens if t}
    if not s_pred and not s_gold:
        return 1.0
    if not s_pred or not s_gold:
        return 0.0
    intersection = len(s_pred & s_gold)
    union = len(s_pred | s_gold)
    return float(intersection / union) if union > 0 else 0.0


def token_precision_recall_f1(
    pred_tokens: set[str] | list[str],
    gold_tokens: set[str] | list[str],
) -> tuple[float, float, float]:
    """Compute token-level Precision, Recall, and F1 between rationale spans."""
    s_pred = {t.lower() for t in pred_tokens if t}
    s_gold = {t.lower() for t in gold_tokens if t}
    if not s_pred and not s_gold:
        return 1.0, 1.0, 1.0
    if not s_pred or not s_gold:
        return 0.0, 0.0, 0.0
    overlap = len(s_pred & s_gold)
    precision = overlap / len(s_pred)
    recall = overlap / len(s_gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_rouge_n(hypothesis: str, reference: str, n: int = 1) -> float:
    """Compute basic ROUGE-N recall score for explanation text."""
    hyp_tokens = [w.lower() for w in re.findall(r"\b\w+\b", hypothesis)]
    ref_tokens = [w.lower() for w in re.findall(r"\b\w+\b", reference)]
    if not hyp_tokens or not ref_tokens:
        return 0.0
    hyp_ngrams = _get_ngrams(hyp_tokens, n)
    ref_ngrams = _get_ngrams(ref_tokens, n)
    if not ref_ngrams:
        return 0.0
    overlap = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
    return float(overlap / len(ref_ngrams))


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """Compute standard LCS-based ROUGE-L F1 score."""
    hyp_tokens = [w.lower() for w in re.findall(r"\b\w+\b", hypothesis)]
    ref_tokens = [w.lower() for w in re.findall(r"\b\w+\b", reference)]
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0

    # Longest Common Subsequence table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if hyp_tokens[i] == ref_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0

    r_lcs = lcs / n
    p_lcs = lcs / m
    return float(2 * r_lcs * p_lcs / (r_lcs + p_lcs))


def evaluate_rationale_suite(
    hypotheses: list[str],
    references: list[str],
    pred_spans: list[list[str]] | None = None,
    gold_spans: list[list[str]] | None = None,
) -> dict[str, float]:
    """Aggregate rationale evaluation across a test corpus."""
    if not hypotheses or not references or len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must be non-empty and of equal length.")

    r1_scores = [compute_rouge_n(h, r, n=1) for h, r in zip(hypotheses, references, strict=False)]
    r2_scores = [compute_rouge_n(h, r, n=2) for h, r in zip(hypotheses, references, strict=False)]
    rl_scores = [compute_rouge_l(h, r) for h, r in zip(hypotheses, references, strict=False)]

    metrics: dict[str, float] = {
        "rouge_1": float(sum(r1_scores) / len(r1_scores)),
        "rouge_2": float(sum(r2_scores) / len(r2_scores)),
        "rouge_l": float(sum(rl_scores) / len(rl_scores)),
    }

    if pred_spans and gold_spans and len(pred_spans) == len(gold_spans):
        ious = [token_iou(p, g) for p, g in zip(pred_spans, gold_spans, strict=False)]
        f1s = [token_precision_recall_f1(p, g)[2] for p, g in zip(pred_spans, gold_spans, strict=False)]
        metrics["mean_token_iou"] = float(sum(ious) / len(ious))
        metrics["mean_token_f1"] = float(sum(f1s) / len(f1s))

    return metrics
