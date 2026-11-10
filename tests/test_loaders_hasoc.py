"""Unit tests for the HASOC dataset loader."""

from __future__ import annotations

from pj03.preprocess.loaders_hasoc import (
    BUILTIN_HASOC_SAMPLES,
    build_processed,
    iter_examples,
    map_hasoc_label,
)
from pj03.preprocess.schema import Split


def test_map_hasoc_label() -> None:
    assert map_hasoc_label("HOF") == 1
    assert map_hasoc_label("HATE") == 1
    assert map_hasoc_label("OFFN") == 1
    assert map_hasoc_label("PRFN") == 1
    assert map_hasoc_label("NOT") == 0
    assert map_hasoc_label("NONE") == 0
    assert map_hasoc_label("unknown") == 0


def test_builtin_hasoc_loader() -> None:
    test_examples = list(iter_examples(Split.TEST))
    val_examples = list(iter_examples(Split.VALIDATION))
    train_examples = list(iter_examples(Split.TRAIN))

    total = len(test_examples) + len(val_examples) + len(train_examples)
    assert total == len(BUILTIN_HASOC_SAMPLES)
    assert len(test_examples) > 0
    assert any(e.label == 1 for e in test_examples)
    assert any(e.label == 0 for e in test_examples)


def test_build_processed_hasoc_tsv(tmp_path) -> None:
    raw_dir = tmp_path / "raw_hasoc"
    tsv_dir = raw_dir / "2021" / "hinglish"
    tsv_dir.mkdir(parents=True)
    tsv_file = tsv_dir / "test.tsv"
    tsv_file.write_text(
        "text_id\ttext\ttask_1\ttask_2\n"
        "h1\tkutte kamine mar jaa\tHOF\tHATE\n"
        "h2\taaj ka din bohot achha tha\tNOT\tNONE\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "processed_hasoc"
    stats = build_processed(raw_dir=raw_dir, out_dir=out_dir, year="2021", language="hinglish")

    assert stats.n_test == 2
    assert stats.n_positive >= 1
    assert stats.n_negative >= 1

    loaded = list(iter_examples(Split.TEST, year="2021", language="hinglish", data_dir=raw_dir))
    assert len(loaded) == 2
    assert loaded[0].label == 1
    assert loaded[1].label == 0
