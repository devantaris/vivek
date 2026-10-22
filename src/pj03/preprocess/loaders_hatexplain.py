"""Dataset loaders: one module per source, all returning the unified schema.

HateXplain raw layout (as downloaded by ``scripts/download_data.py``):
``data/raw/hatexplain/{train,validation,test}.jsonl`` with fields:
    - ``id``: original post id (suffixed ``_gab`` / ``_twitter``)
    - ``post_tokens``: pre-tokenised text (lowercased)
    - ``annotators.label``: 3 class ids; class map 0=hatespeech, 1=normal,
      2=offensive
    - ``annotators.target``: per-annotator list of targeted communities
    - ``rationales``: per-annotator 0/1 masks over ``post_tokens``
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pj03.preprocess.schema import DatasetStats, Example, Language, Split

SOURCE = "hatexplain"
# HateXplain class ids: 0 = hatespeech, 1 = normal, 2 = offensive.
CLASS_HATESPEECH, CLASS_NORMAL, CLASS_OFFENSIVE = 0, 1, 2


def _majority(labels: list[int]) -> int:
    """Majority vote over annotator labels; ties resolve toward hateful.

    With 3 annotators a tie means all three disagree. For moderation research
    we resolve ties toward the more harmful class (documented decision):
    hatespeech beats offensive beats normal.
    """
    counts = Counter(labels)
    top = counts.most_common()
    # Counter.most_common is insertion-ordered for equal counts; sort so
    # that in a tie the *more hateful* class wins regardless of insertion.
    priority = {CLASS_HATESPEECH: 0, CLASS_OFFENSIVE: 1, CLASS_NORMAL: 2}
    top = sorted(top, key=lambda kv: (-kv[1], priority.get(kv[0], 99)))
    return top[0][0]


def _majority_targets(targets: list[list[str]]) -> str | None:
    """Targets named by at least half the annotators, joined with '|'.

    Returns None when the majority is 'Other' or nothing is agreed on.
    """
    if not targets:
        return None
    counts = Counter(t for per_annotator in targets for t in per_annotator)
    n = len(targets)
    agreed = sorted(
        t.lower()
        for t, c in counts.items()
        if c >= max(2, (n + 1) // 2) and t.lower() != "other"
    )
    return "|".join(agreed) or None


def _majority_rationale_mask(rationales: list[list[int]]) -> list[int] | None:
    """Token indices flagged by at least 2 of 3 annotators (hateful posts only)."""
    if not rationales:
        return None
    n_tokens = len(rationales[0])
    per_token = [sum(r[i] for r in rationales) for i in range(n_tokens)]
    return [i for i, votes in enumerate(per_token) if votes >= 2]


def _to_binary(label: int, mode: str = "strict_hate") -> int:
    """Hatespeech -> 1; normal/offensive -> 0 under 'strict_hate'.

    If mode == 'toxic_abusive': both hatespeech and offensive are mapped to 1
    (toxic/abusive), while only normal is 0.
    """
    if mode == "toxic_abusive":
        return 1 if label in (CLASS_HATESPEECH, CLASS_OFFENSIVE) else 0
    return 1 if label == CLASS_HATESPEECH else 0


def build_processed(raw_dir: str, out_dir: str, mode: str = "strict_hate") -> DatasetStats:
    """Convert raw HateXplain JSONL to the unified schema and write splits.

    Writes ``{split}.jsonl`` files under ``out_dir`` and returns stats.
    """
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = DatasetStats(source_dataset=SOURCE)
    for split_name in ("train", "validation", "test"):
        split = Split(split_name)
        examples: list[Example] = []
        with (raw / f"{split_name}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                majority = _majority(row["annotators"]["label"])
                examples.append(
                    Example(
                        id=f"{SOURCE}:{row['id']}",
                        text=" ".join(row["post_tokens"]),
                        label=_to_binary(majority, mode=mode),
                        language=Language.EN,
                        source_dataset=SOURCE,
                        split=split,
                        rationale_mask=(
                            _majority_rationale_mask(row["rationales"])
                            if majority in (CLASS_HATESPEECH, CLASS_OFFENSIVE)
                            else None
                        ),
                        target_group=_majority_targets(row["annotators"]["target"]),
                    )
                )
        path = out / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(ex.model_dump_json() + "\n")
        stats.n_positive += sum(e.label for e in examples)
        stats.n_negative += sum(1 - e.label for e in examples)
        setattr(stats, f"n_{split_name}", len(examples))
    return stats


def iter_examples(split: Split, processed_dir: str = "data/processed/hatexplain") -> Iterator[Example]:
    """Yield :class:`Example` objects from the processed HateXplain splits."""
    path = Path(processed_dir) / f"{split.value}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found - run the build step first "
            "(scripts/build_hatexplain.py or make data)."
        )
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield Example.model_validate_json(line)
