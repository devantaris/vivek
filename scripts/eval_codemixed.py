#!/usr/bin/env python
"""Evaluate cross-lingual performance across unified code-mixed datasets (HASOC + Hinglish).

Usage:
    python scripts/eval_codemixed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.demo.predictor import DemoPredictor  # noqa: E402
from pj03.eval.metrics import classification_metrics  # noqa: E402
from pj03.preprocess import loaders_hasoc, loaders_hinglish  # noqa: E402
from pj03.preprocess.schema import Split  # noqa: E402
from pj03.utils.logging import get_logger  # noqa: E402

logger = get_logger("pj03.eval_codemixed")


def main() -> None:
    logger.info("Loading unified code-mixed benchmarks (HASOC + Hinglish diagnostics) ...")
    hasoc_examples = list(loaders_hasoc.iter_examples(Split.TEST))
    hinglish_examples = list(loaders_hinglish.iter_examples(Split.TEST))

    all_examples = hasoc_examples + hinglish_examples
    logger.info("Total test examples: %d (HASOC: %d, Hinglish: %d)",
                len(all_examples), len(hasoc_examples), len(hinglish_examples))

    predictor = DemoPredictor()
    gold_labels = [ex.label for ex in all_examples]
    pred_labels = []
    pred_scores = []

    for ex in all_examples:
        res = predictor.predict(ex.text, model_type="auto")
        pred_labels.append(res.label)
        pred_scores.append(res.prob_hateful)

    metrics = classification_metrics(gold_labels, pred_labels, pred_scores)
    logger.info("Code-mixed benchmark metrics:\n%s", json.dumps(metrics, indent=2))

    # Calculate per-dataset breakdowns
    hasoc_len = len(hasoc_examples)
    hasoc_metrics = classification_metrics(
        gold_labels[:hasoc_len], pred_labels[:hasoc_len], pred_scores[:hasoc_len]
    )
    hinglish_metrics = classification_metrics(
        gold_labels[hasoc_len:], pred_labels[hasoc_len:], pred_scores[hasoc_len:]
    )

    report = {
        "overall": metrics,
        "hasoc_track": hasoc_metrics,
        "hinglish_track": hinglish_metrics,
        "n_samples": len(all_examples),
        "n_hasoc": hasoc_len,
        "n_hinglish": len(hinglish_examples),
    }

    out_dir = REPO_ROOT / "data" / "processed" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "codemixed_benchmark_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved code-mixed benchmark report -> %s", out_file)


if __name__ == "__main__":
    main()
