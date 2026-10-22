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
from pj03.eval.fairness import bias_audit  # noqa: E402
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
    test_examples = list(loaders_hatexplain.iter_examples(Split.TEST))
    test_texts = [ex.text for ex in test_examples]
    test_labels = [ex.label for ex in test_examples]
    logger.info("train=%d test=%d", len(train_texts), len(test_labels))

    pipe = tfidf_lr.build_pipeline(
        ngram_range=tuple(params["ngram_range"]),
        max_features=params["max_features"],
        C=params["C"],
        class_weight=params["class_weight"],
        seed=config.get("seed", 42),
        char_features=params.get("char_features", False),
    )
    pipe.fit(train_texts, train_labels)
    predictions = pipe.predict(test_texts)
    scores = pipe.predict_proba(test_texts)[:, 1]

    metrics = classification_metrics(test_labels, predictions.tolist(), scores.tolist())
    logger.info("TF-IDF+LogReg test metrics: %s", json.dumps(metrics, indent=2))

    # Early bias signal: per-target-group AUCs (Borkan et al. definitions).
    terms = config.get("bias_audit", {}).get("terms", [])
    group_masks = {
        term: [ex.target_group is not None and term in ex.target_group for ex in test_examples]
        for term in terms
    }
    audit = bias_audit(test_labels, scores.tolist(), group_masks)
    logger.info("Bias audit (test): %s", json.dumps(audit, indent=2))

    results_dir = REPO_ROOT / "data" / "processed" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "tfidf_lr_hatexplain.json"
    out_path.write_text(
        json.dumps({"metrics": metrics, "bias_audit": audit}, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved metrics -> %s", out_path)
    return metrics


def run_encoder(config: dict) -> dict[str, float]:
    """Fine-tune encoder(s) on HateXplain; smoke mode validates the path on CPU."""
    from pj03.baselines.encoder_finetune import train_encoder

    enc = config["baselines"]["encoders"]
    smoke = enc.get("smoke", {})
    train_texts, train_labels = load_split(Split.TRAIN)
    eval_texts, eval_labels = load_split(Split.VALIDATION)

    if smoke.get("enabled", False):
        models = [smoke["model_name"]]
        epochs, batch_size = smoke.get("epochs", 1), smoke.get("batch_size", 16)
        n_train, n_eval = smoke.get("n_train", 200), smoke.get("n_eval", 100)
        logger.info(
            "SMOKE mode: %s on %d train / %d eval examples (CPU path validation)",
            models[0],
            n_train,
            n_eval,
        )
        # Stratified-ish trim: first half positives, half negatives.
        pos_t = [i for i, y in enumerate(train_labels) if y == 1][: n_train // 2]
        neg_t = [i for i, y in enumerate(train_labels) if y == 0][: n_train - n_train // 2]
        keep_t = pos_t + neg_t
        train_texts = [train_texts[i] for i in keep_t]
        train_labels = [train_labels[i] for i in keep_t]
        pos_e = [i for i, y in enumerate(eval_labels) if y == 1][: n_eval // 2]
        neg_e = [i for i, y in enumerate(eval_labels) if y == 0][: n_eval - n_eval // 2]
        keep_e = pos_e + neg_e
        eval_texts = [eval_texts[i] for i in keep_e]
        eval_labels = [eval_labels[i] for i in keep_e]
    else:
        models = enc["models"]
        epochs, batch_size = enc["epochs"], enc["batch_size"]

    all_metrics: dict[str, float] = {}
    results_dir = REPO_ROOT / "data" / "processed" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        logger.info("Fine-tuning %s ...", model_name)
        metrics = train_encoder(
            model_name=model_name,
            train_texts=train_texts,
            train_labels=train_labels,
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            output_dir=str(REPO_ROOT / "checkpoints" / model_name.replace("/", "_")),
            learning_rate=enc["learning_rate"],
            batch_size=batch_size,
            epochs=epochs,
            max_length=enc["max_length"],
            seed=config.get("seed", 42),
        )
        logger.info("%s eval metrics: %s", model_name, json.dumps(metrics, indent=2))
        all_metrics[model_name] = metrics
        suffix = "smoke" if smoke.get("enabled", False) else "full"
        out_path = results_dir / f"encoder_{model_name.replace('/', '_')}_{suffix}.json"
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Saved metrics -> %s", out_path)
    return all_metrics


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
        run_encoder(config)


if __name__ == "__main__":
    main()
