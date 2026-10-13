"""HateCheck functional-test evaluation (Phase 4).

TODO(Phase 4): implement - load HateCheck via
:mod:`pj03.preprocess.loaders_hatecheck`, run the model over all cases, and
report per-functionality failure rates (the 29 functionalities) plus overall
pass rate, split by target identity. Report tables go to
``data/processed/hatecheck_report.json``.
"""

from __future__ import annotations

from typing import Any


def run_hatecheck(predict: Any, model_name: str) -> dict[str, float]:
    """Run HateCheck cases through the model and return failure rates.

    Args:
        predict: Callable mapping list[str] -> predicted binary labels.
        model_name: Checkpoint name for the report key.

    Raises:
        NotImplementedError: Until Phase 4 and HateCheck data download.
    """
    raise NotImplementedError(
        "TODO(Phase 4): implement per-functionality failure rates on HateCheck."
    )
