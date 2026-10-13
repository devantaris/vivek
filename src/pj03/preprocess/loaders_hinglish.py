"""Hinglish offensive-tweet dataset loader.

TODO(Phase 1): pick the exact dataset. Candidates (all require checking the
licence and often a form/gated download):
    - Hinglish Offensive Tweet dataset (e.g. from Fire et al.)
    - HOSID / HSD-Twitter Hinglish corpora used in shared tasks
    - any gated HF hub dataset the team is granted access to
Record the final choice and licence in ``docs/dataset_notes.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

from pj03.preprocess.schema import DatasetStats, Example, Split


def iter_examples(split: Split) -> Iterator[Example]:
    """Yield :class:`Example` objects for the given split.

    TODO(Phase 1): read from ``data/processed/hinglish_tweets/`` and map the
    source labels (e.g. ONG/NOT) to the binary schema.

    Raises:
        NotImplementedError: Until the dataset is chosen and downloaded.
    """
    raise NotImplementedError(
        "TODO(Phase 1): dataset not chosen yet. Candidates and licences are "
        "listed in docs/dataset_notes.md; expected processed files under "
        "data/processed/hinglish_tweets/."
    )


def build_processed(raw_dir: str, out_dir: str) -> DatasetStats:
    """Convert the raw Hinglish corpus to the unified schema.

    TODO(Phase 1): implement after the dataset is chosen.
    """
    raise NotImplementedError("TODO(Phase 1): choose dataset first.")
