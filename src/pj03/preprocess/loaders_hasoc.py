"""HASOC (Hindi/English/code-mixed) loader.

HASOC shared-task data requires registration / manual download; see
``scripts/download_data.py`` and ``docs/dataset_notes.md`` for instructions
and licence notes.
"""

from __future__ import annotations

from collections.abc import Iterator

from pj03.preprocess.schema import DatasetStats, Example, Split


def iter_examples(split: Split, year: str = "2021", language: str = "hi") -> Iterator[Example]:
    """Yield :class:`Example` objects for one HASOC track.

    TODO(Phase 1):
        1. Read the TSV from ``data/raw/hasoc/{year}/{language}/``.
        2. Map HASOC labels (HOF/NOT, and sublabels HATE/OFFN/PRFN) to binary;
           decide and document whether PRFN counts as hateful.
        3. Set ``language`` per track (hi / en / code-mixed).

    Raises:
        NotImplementedError: Until the manually downloaded TSVs exist.
    """
    raise NotImplementedError(
        "TODO(Phase 1): implement after manually downloading HASOC from the "
        "shared-task organisers (registration required). Expected files under "
        f"data/raw/hasoc/{year}/{language}/. See docs/dataset_notes.md."
    )


def build_processed(raw_dir: str, out_dir: str) -> DatasetStats:
    """Convert raw HASOC TSVs to the unified schema and write JSONL splits.

    TODO(Phase 1): implement after inspecting the TSV column layout.
    """
    raise NotImplementedError("TODO(Phase 1): inspect raw HASOC TSV columns first.")
