"""HateCheck loader (functional tests for hate-speech detectors).

HateCheck is an evaluation-only resource: it is used for Phase 4 functional
testing, not training. Record its licence in ``docs/dataset_notes.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

from pj03.preprocess.schema import DatasetStats, Example


def iter_examples() -> Iterator[Example]:
    """Yield every HateCheck case as an :class:`Example`.

    TODO(Phase 4): map HateCheck's ``functionality`` and ``target_ident``
    columns into ``target_group`` so per-functionality failure rates can be
    reported. All cases go to the TEST split - HateCheck must never be
    trained on.

    Raises:
        NotImplementedError: Until the data is downloaded.
    """
    raise NotImplementedError(
        "TODO(Phase 4): implement after downloading HateCheck; keep all cases "
        "in the TEST split."
    )


def build_processed(raw_dir: str, out_dir: str) -> DatasetStats:
    """Convert the raw HateCheck CSV to the unified schema (test split only).

    TODO(Phase 4): implement after download.
    """
    raise NotImplementedError("TODO(Phase 4): implement after download.")
