#!/usr/bin/env python
"""Aggregate all experiment result JSONs into one clear research summary.

Usage:
    python scripts/summarize_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "data" / "processed" / "results"


def _fmt(val: object, precision: int = 4) -> str:
    if val is None:
        return "--"
    if isinstance(val, (int, float)):
        return f"{val:.{precision}f}"
    return str(val)


def main() -> None:
    if not RESULTS.is_dir():
        print("No results directory yet - run an evaluation or baseline first.")
        return

    rows: list[dict] = []
    phase4_data: dict | None = None

    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if path.name == "eval_phase4_report.json":
            phase4_data = data
            continue

        metrics = data.get("metrics", data)
        if "in_domain_metrics" in data:
            metrics = data["in_domain_metrics"]

        macro_f1 = metrics.get("macro_f1") or metrics.get("eval_macro_f1")
        auc = metrics.get("auc") or metrics.get("eval_auc")
        acc = metrics.get("accuracy") or metrics.get("eval_accuracy")
        loss = metrics.get("eval_loss") or metrics.get("loss")
        groups = len(data.get("bias_audit", {})) or (
            len(data.get("fairness_audit", {})) if "fairness_audit" in data else 0
        )

        rows.append(
            {
                "experiment": path.stem,
                "macro_f1": _fmt(macro_f1),
                "auc": _fmt(auc),
                "accuracy": _fmt(acc),
                "eval_loss": _fmt(loss),
                "bias_groups": str(groups) if groups > 0 else "--",
            }
        )

    if not rows:
        print("No valid experiment results found.")
        return

    print("=" * 95)
    print("                      PJ-03 BENCHMARK & EXPERIMENTAL SUMMARY")
    print("=" * 95)
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = " | ".join(k.ljust(widths[k]) for k in rows[0])
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[k]).ljust(widths[k]) for k in rows[0]))
    print("=" * 95)

    if phase4_data:
        print("\n=== Phase 4 Evaluation Highlights ===")
        in_dom = phase4_data.get("in_domain_metrics", {})
        print(
            f"In-Domain HateXplain Test: Macro-F1 = {_fmt(in_dom.get('macro_f1'))}, "
            f"AUC = {_fmt(in_dom.get('auc'))}, Accuracy = {_fmt(in_dom.get('accuracy'))}"
        )

        hc = phase4_data.get("hatecheck_summary", {})
        if hc:
            print(
                f"HateCheck Functional (3,728 cases): Overall Accuracy = {_fmt(hc.get('overall_accuracy'))} "
                f"(Failure Rate = {_fmt(hc.get('overall_failure_rate'))})"
            )

        ood = phase4_data.get("ood_evaluation", {})
        if ood:
            print(
                f"OOD Generalization Drop: In-Domain F1 {_fmt(ood.get('in_domain_macro_f1'))} -> "
                f"OOD F1 {_fmt(ood.get('ood_macro_f1'))} (Delta: {_fmt(ood.get('macro_f1_drop'))})"
            )

        fairness = phase4_data.get("fairness_audit", {})
        if fairness:
            print(f"\nFairness & Bias Audit ({len(fairness)} Demographic Groups):")
            for group, scores in sorted(fairness.items()):
                print(
                    f"  - {group.capitalize():12s} | Subgroup AUC: {_fmt(scores.get('subgroup_auc'))} "
                    f"| BPSN AUC: {_fmt(scores.get('bpsn_auc'))} | BNSP AUC: {_fmt(scores.get('bnsp_auc'))}"
                )
    print()


if __name__ == "__main__":
    sys.exit(main())
