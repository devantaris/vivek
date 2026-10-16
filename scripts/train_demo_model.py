#!/usr/bin/env python
"""Train and persist the demo model (TF-IDF+LogReg on HateXplain).

Usage:
    python scripts/train_demo_model.py

Produces ``artifacts/demo_tfidf.joblib`` (~a few MB) used by the Gradio app
(``make demo``). The artifact is gitignored; metrics JSON is committed.
"""

from __future__ import annotations

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

logger = get_logger("pj03.train_demo")


def main() -> None:
    config = load_config(REPO_ROOT / "configs" / "baselines.yaml")
    set_seed(config.get("seed", 42))
    params = config["baselines"]["tfidf_lr"]

    logger.info("Loading processed HateXplain ...")
    train = list(loaders_hatexplain.iter_examples(Split.TRAIN))
    test = list(loaders_hatexplain.iter_examples(Split.TEST))
    pipe = build_pipeline(
        ngram_range=tuple(params["ngram_range"]),
        max_features=params["max_features"],
        C=params["C"],
        class_weight=params["class_weight"],
        seed=config.get("seed", 42),
        char_features=params.get("char_features", False),
    )
    pipe.fit([e.text for e in train], [e.label for e in train])

    preds = pipe.predict([e.text for e in test])
    scores = pipe.predict_proba([e.text for e in test])[:, 1]
    metrics = classification_metrics(
        [e.label for e in test], preds.tolist(), scores.tolist()
    )
    logger.info("Demo model test metrics: %s", json.dumps(metrics))

    artifact_dir = REPO_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact = artifact_dir / "demo_tfidf.joblib"
    joblib.dump(
        {"pipeline": pipe, "metadata": {"test_metrics": metrics, "dataset": "hatexplain"}},
        artifact,
    )
    logger.info("Saved demo artifact -> %s", artifact)

    results = REPO_ROOT / "data" / "processed" / "results" / "demo_tfidf.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics -> %s", results)


if __name__ == "__main__":
    main()
