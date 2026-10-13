"""Out-of-distribution evaluation (Phase 4).

TODO(Phase 4): define OOD pairs, e.g.:
    - train on HateXplain (English) -> test on HASOC code-mixed
    - train on HASOC -> test on HateXplain and HateCheck
    - train on all -> test on the Hinglish tweets corpus
Report the macro-F1 drop in-domain vs OOD (paper's generalisation table) and
per-language breakdowns.
"""

from __future__ import annotations

from typing import Any


def evaluate_ood(
    predict: Any,
    in_domain_loader: Any,
    ood_loader: Any,
) -> dict[str, float]:
    """Compare in-domain vs OOD macro-F1 for a model.

    Args:
        predict: Callable mapping texts -> predicted labels.
        in_domain_loader: Loader module of the training dataset.
        ood_loader: Loader module of the OOD dataset.

    Raises:
        NotImplementedError: Until Phase 4.
    """
    raise NotImplementedError(
        "TODO(Phase 4): implement in-domain vs OOD macro-F1 comparison."
    )
