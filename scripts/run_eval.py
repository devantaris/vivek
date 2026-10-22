#!/usr/bin/env python
"""Run Phase 4 evaluation: metrics, XAI faithfulness, bias audit, HateCheck, OOD.

Usage:
    python scripts/run_eval.py [--config configs/eval.yaml] [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.demo.predictor import DEFAULT_ARTIFACT, DemoPredictor  # noqa: E402
from pj03.eval.fairness import bias_audit  # noqa: E402
from pj03.eval.hatecheck import run_hatecheck  # noqa: E402
from pj03.eval.metrics import classification_metrics  # noqa: E402
from pj03.eval.ood import evaluate_ood  # noqa: E402
from pj03.preprocess import loaders_hatecheck, loaders_hatexplain  # noqa: E402
from pj03.preprocess.schema import Split  # noqa: E402
from pj03.utils.config import load_config  # noqa: E402
from pj03.utils.logging import get_logger  # noqa: E402
from pj03.utils.seed import set_seed  # noqa: E402

logger = get_logger("pj03.eval")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--model",
        choices=["tfidf", "llm"],
        default="tfidf",
        help="Which model architecture to evaluate (tfidf or llm).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on test examples (useful for fast CPU LLM evaluation).",
    )
    parser.add_argument("overrides", nargs="*", help="e.g. faithfulness.k=5")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))

    artifact_path = REPO_ROOT / DEFAULT_ARTIFACT
    if not artifact_path.is_file():
        logger.warning("Artifact not found, training demo model...")
        import subprocess

        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "train_demo_model.py")], check=True
        )

    predictor = DemoPredictor(artifact_path)
    model_name = "TF-IDF + LogReg Baseline"
    if args.model == "llm":
        logger.info("Initializing and loading Qwen2.5-1.5B LoRA model...")
        predictor.load_llm()
        model_name = "Qwen2.5-1.5B LoRA SFT"

    logger.info("Running Phase 4 comprehensive evaluation on: %s", model_name)

    # 1. In-domain evaluation on HateXplain test split
    test_examples = list(loaders_hatexplain.iter_examples(Split.TEST))
    if args.limit:
        test_examples = test_examples[: args.limit]

    test_texts = [ex.text for ex in test_examples]
    test_labels = [ex.label for ex in test_examples]

    if args.model == "llm":
        logger.info("Running LLM inference across %d test examples...", len(test_texts))
        preds = []
        scores = []
        for i, t in enumerate(test_texts):
            res = predictor.predict(t, model_type="llm")
            preds.append(res.label)
            scores.append(res.prob_hateful)
            if (i + 1) % 25 == 0:
                logger.info("Progress: %d/%d evaluated", i + 1, len(test_texts))
    else:
        preds = predictor.pipeline.predict(test_texts).tolist()
        scores = predictor.pipeline.predict_proba(test_texts)[:, 1].tolist()

    metrics = classification_metrics(test_labels, preds, scores)
    logger.info("In-domain HateXplain metrics: %s", metrics)

    # 2. Bias & Fairness audit across demographic groups
    terms = config.get("bias_audit", {}).get(
        "terms", ["african", "islam", "jewish", "homosexual", "women", "refugee", "arab"]
    )
    group_masks = {
        term: [ex.target_group is not None and term in ex.target_group for ex in test_examples]
        for term in terms
    }
    fairness_report = bias_audit(test_labels, scores, group_masks)
    logger.info("Fairness bias audit: %d groups audited", len(fairness_report))

    # 3. HateCheck functional suite
    if args.model == "llm":
        predict_fn = lambda batch: [predictor.predict(text, model_type="llm").label for text in batch]  # noqa: E731
    else:
        predict_fn = lambda batch: predictor.pipeline.predict(batch)  # noqa: E731

    hatecheck_report = run_hatecheck(
        predict=predict_fn,
        model_name=model_name,
    )
    logger.info("HateCheck overall accuracy: %.4f", hatecheck_report["overall_accuracy"])

    # 4. Out-of-Distribution (OOD) comparison
    hatecheck_examples = list(loaders_hatecheck.iter_examples())
    if args.limit:
        hatecheck_examples = hatecheck_examples[: args.limit]
    ood_texts = [ex.text for ex in hatecheck_examples]
    ood_labels = [ex.label for ex in hatecheck_examples]

    ood_report = evaluate_ood(
        predict=predict_fn,
        in_domain_texts=test_texts,
        in_domain_labels=test_labels,
        ood_texts=ood_texts,
        ood_labels=ood_labels,
        ood_name="HateCheck",
    )

    # Compile unified evaluation artifact
    final_report = {
        "model": model_name,
        "in_domain_metrics": metrics,
        "fairness_audit": fairness_report,
        "hatecheck_summary": {
            "overall_accuracy": hatecheck_report["overall_accuracy"],
            "overall_failure_rate": hatecheck_report["overall_failure_rate"],
            "total_cases": hatecheck_report["total_cases"],
            "top_failures": sorted(
                hatecheck_report["per_functionality_accuracy"].items(),
                key=lambda kv: kv[1],
            )[:5],
        },
        "ood_evaluation": ood_report,
    }

    results_dir = REPO_ROOT / "data" / "processed" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_filename = "eval_llm_report.json" if args.model == "llm" else "eval_phase4_report.json"
    out_path = results_dir / report_filename
    out_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    logger.info("Phase 4 evaluation report saved -> %s", out_path)

    # Print clean summary table
    print("\n" + "=" * 60)
    print("PHASE 4 EVALUATION EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"Model:                     {final_report['model']}")
    print(f"HateXplain Macro-F1:       {metrics['macro_f1']:.4f}")
    print(f"HateXplain ROC-AUC:        {metrics.get('auc', 0.0):.4f}")
    print(f"HateXplain Accuracy:       {metrics['accuracy']:.4f}")
    print(f"HateCheck Pass Rate:       {hatecheck_report['overall_accuracy']:.4f}")
    print(f"OOD Generalisation Drop:   {ood_report['macro_f1_drop']:+.4f} F1")
    print("Worst-3 Subgroup AUCs:")
    sorted_groups = sorted(fairness_report.items(), key=lambda kv: kv[1]["subgroup_auc"])
    for g, vals in sorted_groups[:3]:
        print(
            f"  - {g:12s} AUC={vals['subgroup_auc']:.3f} | BPSN={vals['bpsn_auc']:.3f} | BNSP={vals['bnsp_auc']:.3f}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
