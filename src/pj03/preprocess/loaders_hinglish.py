"""Hinglish offensive and hate speech dataset loader.

Supports:
    1. HASOC code-mixed track (Hindi-English Romanized).
    2. Bohra et al. (FIRE 2018) code-mixed hate speech Twitter corpus.
    3. Custom CSV/TSV/JSONL files with columns (text, label/task_1).
    4. Out-of-the-box curated Hinglish diagnostic test suite.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from pj03.preprocess.hinglish import normalise_hinglish
from pj03.preprocess.schema import DatasetStats, Example, Language, Split

SOURCE = "hinglish_code_mixed"

# Out-of-the-box curated Hinglish diagnostic cases for zero-shot / cross-lingual auditing.
BUILTIN_HINGLISH_DIAGNOSTICS: list[dict] = [
    # Explicit hate / severe abusive
    {"id": "h_001", "text": "tu bohot bada kutta hai saale", "label": 1, "target": "individual"},
    {"id": "h_002", "text": "chutiye log sab ke sab mar jao", "label": 1, "target": "general"},
    {
        "id": "h_003",
        "text": "ye mulla log desh ko barbad kar rahe hai",
        "label": 1,
        "target": "islam",
    },
    {
        "id": "h_004",
        "text": "aurat log ki koi aukaat nahi hoti kitchen me raho",
        "label": 1,
        "target": "women",
    },
    {
        "id": "h_005",
        "text": "saare dalit kutto ko yaha se bhagao",
        "label": 1,
        "target": "caste_oppressed",
    },
    {"id": "h_006", "text": "haramzada pagal hai kya dimag nahi hai", "label": 1, "target": "individual"},
    {"id": "h_007", "text": "you are such a disgusting kutta", "label": 1, "target": "individual"},
    {"id": "h_008", "text": "gandu kahi ka bakwas band kar", "label": 1, "target": "individual"},
    # Mild offensive / insults
    {"id": "h_009", "text": "bhai tu thoda sa pagal hai kya", "label": 1, "target": "individual"},
    {"id": "h_010", "text": "kya bakwas movie thi dimaag kharab ho gaya", "label": 1, "target": "none"},
    # Benign / Polite / Everyday conversational
    {"id": "h_011", "text": "namaste dosto kaise ho aap sab", "label": 0, "target": None},
    {"id": "h_012", "text": "kya mast mausam hai aaj maza aa gaya", "label": 0, "target": None},
    {
        "id": "h_013",
        "text": "bhai kya zabardast match tha maza aa gaya dekh ke",
        "label": 0,
        "target": None,
    },
    {"id": "h_014", "text": "hum sab milke party karenge weekend par", "label": 0, "target": None},
    {"id": "h_015", "text": "aaj office me bohot kaam tha thak gaya", "label": 0, "target": None},
    # Counterfactuals / Identity mentions in benign contexts
    {
        "id": "h_016",
        "text": "hamare desh me hindu muslim bhai bhai jaise rehte hai",
        "label": 0,
        "target": "islam",
    },
    {
        "id": "h_017",
        "text": "mera dost muslim hai aur hum sath me eid manate hai",
        "label": 0,
        "target": "islam",
    },
    {
        "id": "h_018",
        "text": "aurat log desh ka naam roshan kar rahi hai",
        "label": 0,
        "target": "women",
    },
    {"id": "h_019", "text": "apne dosto ki hamesha madad karni chahiye", "label": 0, "target": None},
    {"id": "h_020", "text": "chai peene chale kya bhai", "label": 0, "target": None},
]


def _map_label(val: object) -> int:
    """Normalize label string/int to binary 0 or 1."""
    if isinstance(val, (int, float)):
        return 1 if int(val) > 0 else 0
    s = str(val).strip().upper()
    if s in ("1", "HOF", "HATE", "OFFN", "PRFN", "TRUE", "YES", "TOXIC"):
        return 1
    return 0


def build_processed(raw_dir: str | Path, out_dir: str | Path) -> DatasetStats:
    """Convert raw Hinglish data (CSV, TSV, or JSONL) to unified schema.

    If raw_dir contains files, converts them; otherwise creates the diagnostic
    test split from BUILTIN_HINGLISH_DIAGNOSTICS.
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[Example] = []

    # Check for raw input files
    raw_files = list(raw_path.glob("*.csv")) + list(raw_path.glob("*.tsv")) + list(raw_path.glob("*.jsonl"))

    if not raw_files:
        # Use built-in diagnostic test suite
        for item in BUILTIN_HINGLISH_DIAGNOSTICS:
            records.append(
                Example(
                    id=f"{SOURCE}:{item['id']}",
                    text=item["text"].strip(),
                    label=item["label"],
                    language=Language.HINGLISH,
                    source_dataset=SOURCE,
                    split=Split.TEST,
                    target_group=item.get("target"),
                    meta={"normalised_text": normalise_hinglish(item["text"])},
                )
            )
    else:
        for file in raw_files:
            if file.suffix == ".jsonl":
                with file.open(encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        row = json.loads(line)
                        text = row.get("text") or row.get("tweet") or row.get("post", "")
                        label = _map_label(row.get("label", row.get("task_1", 0)))
                        records.append(
                            Example(
                                id=f"{SOURCE}:{file.stem}_{i}",
                                text=text.strip(),
                                label=label,
                                language=Language.HINGLISH,
                                source_dataset=SOURCE,
                                split=Split.TEST,
                                meta={"normalised_text": normalise_hinglish(text)},
                            )
                        )
            else:
                delimiter = "\t" if file.suffix == ".tsv" else ","
                with file.open(encoding="utf-8", errors="replace") as fh:
                    reader = csv.DictReader(fh, delimiter=delimiter)
                    for i, row in enumerate(reader):
                        text = (
                            row.get("text")
                            or row.get("tweet")
                            or row.get("sentence")
                            or row.get("post", "")
                        )
                        label = _map_label(row.get("label", row.get("task_1", row.get("task1", 0))))
                        records.append(
                            Example(
                                id=f"{SOURCE}:{file.stem}_{i}",
                                text=text.strip(),
                                label=label,
                                language=Language.HINGLISH,
                                source_dataset=SOURCE,
                                split=Split.TEST,
                                meta={"normalised_text": normalise_hinglish(text)},
                            )
                        )

    test_file = out_path / "test.jsonl"
    with test_file.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")

    return DatasetStats(
        source_dataset=SOURCE,
        n_test=len(records),
        n_positive=sum(r.label for r in records),
        n_negative=len(records) - sum(r.label for r in records),
    )


def iter_examples(
    split: Split = Split.TEST,
    processed_dir: str | Path = "data/processed/hinglish_tweets",
) -> Iterator[Example]:
    """Yield :class:`Example` objects for Hinglish evaluations."""
    path = Path(processed_dir) / f"{split.value}.jsonl"
    if not path.is_file():
        # Fall back to built-in diagnostic test suite
        for item in BUILTIN_HINGLISH_DIAGNOSTICS:
            yield Example(
                id=f"{SOURCE}:{item['id']}",
                text=item["text"].strip(),
                label=item["label"],
                language=Language.HINGLISH,
                source_dataset=SOURCE,
                split=Split.TEST,
                target_group=item.get("target"),
                meta={"normalised_text": normalise_hinglish(item["text"])},
            )
        return

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Example.model_validate_json(line)
