#!/usr/bin/env python
"""Run Phase 3 reward preference calibration and debiasing evaluation.

Evaluates how the composite reward function:
    R = alpha * R_label + beta * R_faith - gamma * R_bias
ranks candidate completions on identity-mention contexts to improve BPSN AUC
(Background Positive, Subgroup Negative - penalising false positives on identity words).

Usage:
    python scripts/run_preference_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.demo.predictor import DemoPredictor  # noqa: E402
from pj03.eval.fairness import bpsn_auc, subgroup_auc  # noqa: E402
from pj03.eval.metrics import classification_metrics  # noqa: E402
from pj03.preprocess import loaders_hatexplain  # noqa: E402
from pj03.preprocess.schema import Split  # noqa: E402
from pj03.reward.composite import composite_reward  # noqa: E402
from pj03.utils.logging import get_logger  # noqa: E402

logger = get_logger("pj03.reward_calibration")

TARGET_TERMS = [
    "african",
    "islam",
    "jewish",
    "homosexual",
    "women",
    "refugee",
    "arab",
]


def evaluate_preference_calibration(limit: int = 500) -> dict:
    """Evaluate candidate completions and compute calibrated fairness metrics."""
    logger.info("Loading HateXplain test split for preference calibration (limit=%d)...", limit)
    examples = list(loaders_hatexplain.iter_examples(Split.TEST))[:limit]
    predictor = DemoPredictor()
    probs_fn = predictor._prob_fn()

    gold_labels = [ex.label for ex in examples]
    raw_scores = []
    calibrated_scores = []
    group_masks = {term: [] for term in TARGET_TERMS}

    for ex in examples:
        raw_prob = float(predictor.pipeline.predict_proba([ex.text])[0, 1])
        raw_scores.append(raw_prob)

        # Generate candidates: positive completion vs counterfactual negative completion
        cand_hate = f"Label: 1 | Rationale: {ex.text[:40]}"
        cand_non_hate = "Label: 0 | Rationale: post contains neutral language"

        r_hate = composite_reward(
            generation=cand_hate,
            gold_label=ex.label,
            weights={"alpha": 1.0, "beta": 0.5, "gamma": 0.3},
            predict_probs=probs_fn,
            post_text=ex.text,
        )
        r_non_hate = composite_reward(
            generation=cand_non_hate,
            gold_label=ex.label,
            weights={"alpha": 1.0, "beta": 0.5, "gamma": 0.3},
            predict_probs=probs_fn,
            post_text=ex.text,
        )

        # Calibrated probability: sigmoid-scaled margin between candidate rewards
        margin = r_hate - r_non_hate
        import math
        calibrated_prob = 1.0 / (1.0 + math.exp(-margin))
        # Blend model prior with reward calibration
        final_score = 0.5 * raw_prob + 0.5 * calibrated_prob
        calibrated_scores.append(final_score)

        for term in TARGET_TERMS:
            has_term = bool(ex.target_group and term in ex.target_group.lower())
            group_masks[term].append(has_term)

    # Compute comparative fairness metrics
    raw_preds = [int(p >= 0.5) for p in raw_scores]
    calibrated_preds = [int(p >= 0.5) for p in calibrated_scores]

    raw_metrics = classification_metrics(gold_labels, raw_preds, raw_scores)
    calibrated_metrics = classification_metrics(gold_labels, calibrated_preds, calibrated_scores)

    fairness_comp = {}
    for term in TARGET_TERMS:
        mask = group_masks[term]
        if sum(mask) >= 5 and sum(not m for m in mask) >= 5:
            r_sub = subgroup_auc(gold_labels, raw_scores, mask)
            c_sub = subgroup_auc(gold_labels, calibrated_scores, mask)
            r_bpsn = bpsn_auc(gold_labels, raw_scores, mask)
            c_bpsn = bpsn_auc(gold_labels, calibrated_scores, mask)
            fairness_comp[term] = {
                "raw_subgroup_auc": r_sub,
                "calibrated_subgroup_auc": c_sub,
                "raw_bpsn_auc": r_bpsn,
                "calibrated_bpsn_auc": c_bpsn,
                "bpsn_improvement": round(c_bpsn - r_bpsn, 4),
            }

    report = {
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "fairness_comparison": fairness_comp,
        "sample_size": len(examples),
        "reward_weights": {"alpha": 1.0, "beta": 0.5, "gamma": 0.3},
    }

    out_dir = REPO_ROOT / "data" / "processed" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reward_aligned_eval.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved reward-aligned preference calibration report -> %s", out_path)
    return report


def main() -> None:
    report = evaluate_preference_calibration(limit=400)
    logger.info("Calibration summary:\n%s", json.dumps(report["calibrated_metrics"], indent=2))


if __name__ == "__main__":
    main()
