"""Unit tests for the Hinglish dataset loader."""

from __future__ import annotations

from pj03.preprocess.loaders_hinglish import (
    BUILTIN_HINGLISH_DIAGNOSTICS,
    build_processed,
    iter_examples,
)
from pj03.preprocess.schema import Language, Split


def test_builtin_diagnostics_loader() -> None:
    examples = list(iter_examples(Split.TEST, processed_dir="non_existent_dir"))
    assert len(examples) == len(BUILTIN_HINGLISH_DIAGNOSTICS)
    assert all(e.language == Language.HINGLISH for e in examples)
    # Check that hateful and non-hateful are both represented
    assert any(e.label == 1 for e in examples)
    assert any(e.label == 0 for e in examples)


def test_build_processed_csv(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "hasoc_sample.csv"
    csv_file.write_text(
        "text,task_1\n"
        "tu pagal hai kya,HOF\n"
        "namaste dosto,NOT\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "processed"
    stats = build_processed(raw_dir, out_dir)

    assert stats.total == 2
    assert stats.n_positive == 1
    assert stats.n_negative == 1

    loaded = list(iter_examples(Split.TEST, processed_dir=out_dir))
    assert len(loaded) == 2
    assert loaded[0].label == 1
    assert loaded[1].label == 0
