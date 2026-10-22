"""Out-of-distribution evaluation (Phase 4).

Compares model performance in-domain (e.g. HateXplain test) versus
out-of-distribution (e.g. HateCheck or cross-lingual splits) and computes
the generalisation gap and macro-F1 drops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pj03.eval.metrics import classification_metrics
from pj03.utils.logging import get_logger

logger = get_logger("pj03.eval.ood")


def evaluate_ood(
    predict: Callable[[list[str]], list[int]],
    in_domain_texts: list[str],
    in_domain_labels: list[int],
    ood_texts: list[str],
    ood_labels: list[int],
    ood_name: str = "OOD",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Compare in-domain vs OOD macro-F1 for a model.

    Args:
        predict: Callable mapping texts -> predicted binary labels.
        in_domain_texts: Test texts from the in-domain dataset.
        in_domain_labels: Ground-truth binary labels for in-domain texts.
        ood_texts: Test texts from the out-of-distribution dataset.
        ood_labels: Ground-truth binary labels for OOD texts.
        ood_name: Identifier for the OOD dataset.
        batch_size: Evaluation batch size.

    Returns:
        Dict with in_domain_metrics, ood_metrics, macro_f1_drop, and accuracy_drop.
    """

    def _get_preds(texts: list[str]) -> list[int]:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            preds.extend([int(p) for p in predict(batch)])
        return preds

    in_preds = _get_preds(in_domain_texts)
    ood_preds = _get_preds(ood_texts)

    in_metrics = classification_metrics(in_domain_labels, in_preds)
    ood_metrics = classification_metrics(ood_labels, ood_preds)

    f1_drop = in_metrics["macro_f1"] - ood_metrics["macro_f1"]
    acc_drop = in_metrics["accuracy"] - ood_metrics["accuracy"]

    report = {
        "ood_dataset": ood_name,
        "in_domain_macro_f1": round(in_metrics["macro_f1"], 4),
        "ood_macro_f1": round(ood_metrics["macro_f1"], 4),
        "macro_f1_drop": round(f1_drop, 4),
        "in_domain_accuracy": round(in_metrics["accuracy"], 4),
        "ood_accuracy": round(ood_metrics["accuracy"], 4),
        "accuracy_drop": round(acc_drop, 4),
        "in_domain_metrics": in_metrics,
        "ood_metrics": ood_metrics,
    }
    logger.info(
        "OOD [%s]: in_domain_F1=%.4f ood_F1=%.4f (drop=%.4f)",
        ood_name,
        in_metrics["macro_f1"],
        ood_metrics["macro_f1"],
        f1_drop,
    )
    return report
