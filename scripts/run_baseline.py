#!/usr/bin/env python
"""Run Phase 1 baselines: TF-IDF+LogReg (implemented) and encoders (stub).

Usage:
    python scripts/run_baseline.py [--config configs/baselines.yaml] \
        [--model tfidf|encoder] [key=value ...]

Examples:
    python scripts/run_baseline.py --model tfidf
    python scripts/run_baseline.py --model tfidf baselines.tfidf_lr.C=3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.baselines import tfidf_lr  # noqa: E402
from pj03.eval.metrics import classification_metrics  # noqa: E402
from pj03.preprocess import loaders_hatexplain  # noqa: E402
from pj03.preprocess.schema import Split  # noqa: E402
from pj03.utils.config import load_config  # noqa: E402
from pj03.utils.logging import add_file_handler, get_logger  # noqa: E402
from pj03.utils.seed import set_seed  # noqa: E402

logger = get_logger("pj03.baseline")


def load_split(split: Split) -> tuple[list[str], list[int]]:
    """Load one processed HateXplain split as (texts, labels)."""
    texts: list[str] = []
    labels: list[int] = []
    for ex in loaders_hatexplain.iter_examples(split):
        texts.append(ex.text)
        labels.append(ex.label)
    return texts, labels


def run_tfidf(config: dict) -> dict[str, float]:
    """Train and evaluate the TF-IDF+LogReg baseline on HateXplain."""
    params = config["baselines"]["tfidf_lr"]
    logger.info("Loading processed HateXplain ...")
    train_texts, train_labels = load_split(Split.TRAIN)
    test_texts, test_labels = load_split(Split.TEST)
    logger.info("train=%d test=%d", len(train_texts), len(test_labels))

    pipe = tfidf_lr.build_pipeline(
        ngram_range=tuple(params["ngram_range"]),
        max_features=params["max_features"],
        C=params["C"],
        class_weight=params["class_weight"],
        seed=config.get("seed", 42),
    )
    pipe.fit(train_texts, train_labels)
    predictions = pipe.predict(test_texts)
    scores = pipe.predict_proba(test_texts)[:, 1]

    metrics = classification_metrics(test_labels, predictions.tolist(), scores.tolist())
    logger.info("TF-IDF+LogReg test metrics: %s", json.dumps(metrics, indent=2))

    results_dir = REPO_ROOT / "data" / "processed" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "tfidf_lr_hatexplain.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics -> %s", out_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--model", choices=["tfidf", "encoder"], default="tfidf")
    parser.add_argument("overrides", nargs="*", help="e.g. baselines.tfidf_lr.C=3.0")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    add_file_handler(logger, REPO_ROOT / "logs" / "baselines.log")

    if args.model == "tfidf":
        run_tfidf(config)
    else:
        raise NotImplementedError(
            "TODO(Phase 1): encoder fine-tuning (mBERT/XLM-R/MuRIL) - see "
            "src/pj03/baselines/encoder_finetune.py. Requires GPU."
        )


if __name__ == "__main__":
    main()
