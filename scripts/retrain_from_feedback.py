#!/usr/bin/env python
"""Active Learning / Human-in-the-Loop Retraining Engine.

Extracts human moderator confirmations and overrides from ``logs/feedback.jsonl``
and ``logs/decisions.jsonl``, converts overrides into hard-negative training signals,
augments the training corpus, retrains the pipeline, and verifies test metrics.

Usage:
    python scripts/retrain_from_feedback.py [--feedback logs/feedback.jsonl] [--decisions logs/decisions.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib  # noqa: E402

from pj03.baselines.tfidf_lr import build_pipeline  # noqa: E402
from pj03.eval.metrics import classification_metrics  # noqa: E402
from pj03.preprocess import loaders_hatexplain  # noqa: E402
from pj03.preprocess.schema import Split  # noqa: E402
from pj03.utils.config import load_config  # noqa: E402
from pj03.utils.logging import get_logger  # noqa: E402
from pj03.utils.seed import set_seed  # noqa: E402

logger = get_logger("pj03.active_learning")


def extract_feedback_samples(
    feedback_path: Path, decisions_path: Path
) -> list[tuple[str, int, bool]]:
    """Extract (text, final_label, is_override) from feedback and decisions logs."""
    if not feedback_path.is_file() or not decisions_path.is_file():
        return []

    # Map decision_id -> input_scrubbed text
    decision_texts: dict[str, str] = {}
    with decisions_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entry = json.loads(line)
                d_id = entry.get("decision_id")
                txt = entry.get("input_scrubbed") or entry.get("text")
                if d_id and txt:
                    decision_texts[d_id] = txt

    feedback_samples: list[tuple[str, int, bool]] = []
    with feedback_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                fb = json.loads(line)
                d_id = fb.get("decision_id")
                final_label = fb.get("final_label")
                model_label = fb.get("model_label")
                if d_id in decision_texts and final_label in (0, 1):
                    is_override = bool(final_label != model_label)
                    feedback_samples.append((decision_texts[d_id], final_label, is_override))

    return feedback_samples


def run_active_learning_retrain(
    feedback_path: str | Path = "logs/feedback.jsonl",
    decisions_path: str | Path = "logs/decisions.jsonl",
    out_artifact: str | Path = "artifacts/retrained_feedback_model.joblib",
) -> dict:
    feedback_path = Path(feedback_path)
    decisions_path = Path(decisions_path)
    out_artifact = Path(out_artifact)

    config = load_config(REPO_ROOT / "configs" / "baselines.yaml")
    set_seed(config.get("seed", 42))
    params = config["baselines"]["tfidf_lr"]

    logger.info("Extracting moderator feedback...")
    samples = extract_feedback_samples(feedback_path, decisions_path)
    overrides = [s for s in samples if s[2]]
    confirmations = [s for s in samples if not s[2]]
    logger.info("Feedback summary: %d total (%d overrides, %d confirmations)",
                len(samples), len(overrides), len(confirmations))

    train = list(loaders_hatexplain.iter_examples(Split.TRAIN))
    test = list(loaders_hatexplain.iter_examples(Split.TEST))

    train_texts = [e.text for e in train]
    train_labels = [e.label for e in train]

    # Augment with feedback samples (overrides upweighted 3x as hard examples)
    for txt, lbl, is_ovr in samples:
        multiplier = 3 if is_ovr else 1
        for _ in range(multiplier):
            train_texts.append(txt)
            train_labels.append(lbl)

    logger.info("Training augmented pipeline on %d examples ...", len(train_texts))
    pipe = build_pipeline(
        ngram_range=tuple(params["ngram_range"]),
        max_features=params["max_features"],
        C=params["C"],
        class_weight=params["class_weight"],
        seed=config.get("seed", 42),
        char_features=params.get("char_features", False),
    )
    pipe.fit(train_texts, train_labels)

    preds = pipe.predict([e.text for e in test])
    scores = pipe.predict_proba([e.text for e in test])[:, 1]
    metrics = classification_metrics([e.label for e in test], preds.tolist(), scores.tolist())
    logger.info("Retrained model metrics: Macro-F1=%.4f, AUC=%.4f", metrics["macro_f1"], metrics["auc"])

    out_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "metadata": {"test_metrics": metrics, "augmented_samples": len(samples)}},
        out_artifact,
    )
    logger.info("Saved retrained active learning model -> %s", out_artifact)

    report = {
        "metrics": metrics,
        "n_feedback_samples": len(samples),
        "n_overrides": len(overrides),
        "n_confirmations": len(confirmations),
        "total_train_examples": len(train_texts),
        "artifact_path": str(out_artifact),
    }

    out_report = REPO_ROOT / "data" / "processed" / "results" / "active_learning_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved active learning report -> %s", out_report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", default="logs/feedback.jsonl")
    parser.add_argument("--decisions", default="logs/decisions.jsonl")
    parser.add_argument("--out", default="artifacts/retrained_feedback_model.joblib")
    args = parser.parse_args()

    report = run_active_learning_retrain(
        feedback_path=args.feedback,
        decisions_path=args.decisions,
        out_artifact=args.out,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
