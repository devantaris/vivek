"""TF-IDF + Logistic Regression baseline (Phase 1, CPU-friendly)."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def build_pipeline(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 50_000,
    C: float = 1.0,
    class_weight: str | None = "balanced",
    seed: int = 42,
) -> Pipeline:
    """Build a TF-IDF -> LogisticRegression sklearn pipeline.

    Fully CPU-safe and fast enough to run on the university laptops.

    Args:
        ngram_range: Word n-gram range for TF-IDF.
        max_features: Vocabulary cap.
        C: Inverse L2 regularisation strength.
        class_weight: ``"balanced"`` recommended given class imbalance.
        seed: Random seed for the solver.

    Returns:
        An unfitted :class:`sklearn.pipeline.Pipeline`.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_and_evaluate(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    **pipeline_kwargs: Any,
) -> dict[str, float]:
    """Fit the pipeline and return test metrics (macro-F1, accuracy, etc.).

    TODO(Phase 1): wire to :func:`pj03.eval.metrics.classification_metrics`
    once loaders produce real splits. Kept minimal and honest for now.
    """
    pipe = build_pipeline(**pipeline_kwargs)
    pipe.fit(train_texts, train_labels)
    predictions = pipe.predict(test_texts)
    return _to_metrics(test_labels, predictions.tolist())


def _to_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute macro-F1 / accuracy without importing eval.metrics (no cycle)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
