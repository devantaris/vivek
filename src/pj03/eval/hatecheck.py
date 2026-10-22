"""HateCheck functional-test evaluation (Phase 4).

Evaluates a model across the 29 HateCheck functional tests (slurs, negations,
profanity, phrasing variants) and reports per-functionality pass/failure rates
and per-target-group accuracies.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from pj03.preprocess import loaders_hatecheck
from pj03.utils.logging import get_logger

logger = get_logger("pj03.eval.hatecheck")


def run_hatecheck(
    predict: Callable[[list[str]], list[int]] | Callable[[str], int],
    model_name: str = "model",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run HateCheck cases through the model and return detailed failure & pass rates.

    Args:
        predict: Callable mapping a list of texts (or a single text) to binary label(s).
        model_name: Name of model for reporting.
        batch_size: Evaluation batch size.

    Returns:
        Dict with overall accuracy, per-functionality accuracies, and per-target-group accuracies.
    """
    examples = list(loaders_hatecheck.iter_examples())
    texts = [ex.text for ex in examples]
    golds = [ex.label for ex in examples]

    # Predict in batches or single items
    try:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = predict(batch)
            if isinstance(res, (int, float)):
                preds.append(int(res))
            else:
                preds.extend([int(p) for p in res])
    except Exception:
        # Fall back to item-by-item prediction if batching is not supported
        preds = [int(predict(t)) for t in texts]

    total = len(examples)
    correct = sum(p == g for p, g in zip(preds, golds, strict=False))
    overall_accuracy = correct / max(total, 1)

    # Per-functionality breakdowns
    func_total: dict[str, int] = defaultdict(int)
    func_correct: dict[str, int] = defaultdict(int)

    # Per-target-group breakdowns
    group_total: dict[str, int] = defaultdict(int)
    group_correct: dict[str, int] = defaultdict(int)

    for ex, p, g in zip(examples, preds, golds, strict=False):
        func = ex.functionality or "unknown"
        func_total[func] += 1
        if p == g:
            func_correct[func] += 1

        group = ex.target_group or "unspecified"
        group_total[group] += 1
        if p == g:
            group_correct[group] += 1

    per_func_accuracy = {f: round(func_correct[f] / func_total[f], 4) for f in sorted(func_total)}
    per_group_accuracy = {
        g: round(group_correct[g] / group_total[g], 4) for g in sorted(group_total)
    }

    report = {
        "model_name": model_name,
        "total_cases": total,
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_failure_rate": round(1.0 - overall_accuracy, 4),
        "per_functionality_accuracy": per_func_accuracy,
        "per_target_group_accuracy": per_group_accuracy,
    }
    logger.info(
        "HateCheck [%s]: accuracy=%.4f (failure_rate=%.4f across %d test cases)",
        model_name,
        overall_accuracy,
        1.0 - overall_accuracy,
        total,
    )
    return report
