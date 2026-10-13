"""Dataset loaders: one module per source, all returning the unified schema.

Each loader raises NotImplementedError until the raw data has been downloaded
and inspected. See ``scripts/download_data.py`` and ``docs/dataset_notes.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

from pj03.preprocess.schema import DatasetStats, Example, Split


def iter_examples(split: Split) -> Iterator[Example]:
    """Yield :class:`Example` objects for the given split.

    TODO(Phase 1):
        1. Read ``data/processed/hatexplain/{split}.jsonl`` (produced by the
           build step below).
        2. Map HateXplain's 3-way classes (hatespeech, offensive, normal) and
           majority-vote annotator labels to our binary label
           (hatespeech=1, else 0 - decision to record in docs).
        3. Use the annotators' ``target`` majority vote for ``target_group``.
        4. Use the tokens flagged in the majority rationale for
           ``rationale_mask``.

    Raises:
        NotImplementedError: Until the processed files exist.
    """
    raise NotImplementedError(
        "TODO(Phase 1): implement after `make data` produces "
        "data/processed/hatexplain/. See module docstring for label mapping."
    )


def build_processed(raw_dir: str, out_dir: str) -> DatasetStats:
    """Convert raw HateXplain JSON to the unified schema and write JSONL splits.

    TODO(Phase 1): implement once the raw file layout has been inspected.

    Args:
        raw_dir: Directory containing the raw download.
        out_dir: Destination directory under ``data/processed/``.
    """
    raise NotImplementedError("TODO(Phase 1): inspect raw HateXplain JSON first.")
