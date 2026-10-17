#!/usr/bin/env python
"""Release gate: fail (nonzero exit) if quality or fairness regresses.

Usage:
    python scripts/release_gate.py [--config configs/release.yaml]

Evaluates the current demo model (retrained fresh, ~30s CPU) on the HateXplain
test set, checks the gates in ``configs/release.yaml``, writes a gate report
JSON, and exits 0 only if every gate passes. Intended as a CI/CD hook: any
future model change (SFT, GRPO) must pass this before shipping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def evaluate_current_model() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Retrain the baseline fresh and return (metrics, per-group audit)."""
    from pj03.baselines.tfidf_lr import build_pipeline
    from pj03.eval.fairness import bias_audit
    from pj03.eval.metrics import classification_metrics
    from pj03.preprocess import loaders_hatexplain
    from pj03.preprocess.schema import Split
    from pj03.utils.config import load_config
    from pj03.utils.seed import set_seed

    config = load_config(REPO_ROOT / "configs" / "baselines.yaml")
    set_seed(config.get("seed", 42))
    params = config["baselines"]["tfidf_lr"]
    train = list(loaders_hatexplain.iter_examples(Split.TRAIN))
    test = list(loaders_hatexplain.iter_examples(Split.TEST))

    pipe = build_pipeline(
        ngram_range=tuple(params["ngram_range"]),
        max_features=params["max_features"],
        C=params["C"],
        class_weight=params["class_weight"],
        char_features=params.get("char_features", False),
        seed=config.get("seed", 42),
    )
    pipe.fit([e.text for e in train], [e.label for e in train])
    preds = pipe.predict([e.text for e in test])
    scores = pipe.predict_proba([e.text for e in test])[:, 1]
    metrics = classification_metrics(
        [e.label for e in test], preds.tolist(), scores.tolist()
    )

    terms = config.get("bias_audit", {}).get("terms", [])
    masks = {
        term: [
            ex.target_group is not None and term in ex.target_group for ex in test
        ]
        for term in terms
    }
    audit = bias_audit([e.label for e in test], scores.tolist(), masks)
    return metrics, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/release.yaml")
    args = parser.parse_args()

    from pj03.utils.config import load_config
    from pj03.utils.logging import get_logger

    logger = get_logger("pj03.release_gate")
    gates = load_config(REPO_ROOT / args.config)
    started = time.perf_counter()
    metrics, audit = evaluate_current_model()

    failures: list[str] = []
    report: dict = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gates": {},
        "metrics": metrics,
        "fairness_worst": {},
    }

    q = gates["quality"]
    for name, floor in (("macro_f1", q["macro_f1_min"]), ("auc", q["auc_min"])):
        value = metrics.get(name)
        ok = value is not None and value >= floor
        report["gates"][f"quality.{name}"] = {"value": value, "floor": floor, "pass": ok}
        if not ok:
            failures.append(f"quality.{name}: {value} < floor {floor}")

    f = gates["fairness"]
    worst = {
        key: min(v[key] for v in audit.values() if v[key] == v[key])  # ignore NaN
        for key in ("subgroup_auc", "bpsn_auc", "bnsp_auc")
        if any(v[key] == v[key] for v in audit.values())
    }
    floors = {
        "subgroup_auc": f["subgroup_auc_min"],
        "bpsn_auc": f["bpsn_auc_min"],
        "bnsp_auc": f["bnsp_auc_min"],
    }
    for key, floor in floors.items():
        value = worst.get(key)
        ok = value is not None and value >= floor
        report["gates"][f"fairness.{key}"] = {"value": value, "floor": floor, "pass": ok}
        if not ok:
            failures.append(f"fairness.{key}: worst group {value} < floor {floor}")
    report["fairness_worst"] = worst
    report["fairness_full"] = audit
    report["pass"] = not failures
    report["duration_s"] = round(time.perf_counter() - started, 1)

    report_path = REPO_ROOT / gates.get("report_path", "data/processed/results/release_gate.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, entry in report["gates"].items():
        logger.info(
            "GATE %-28s %-8s (value=%s floor=%s)",
            name, "PASS" if entry["pass"] else "FAIL", entry["value"], entry["floor"],
        )
    if failures:
        logger.error("RELEASE GATE FAILED: %s", "; ".join(failures))
        return 1
    logger.info("RELEASE GATE PASSED in %.1fs -> %s", report["duration_s"], report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
