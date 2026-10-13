#!/usr/bin/env python
"""Aggregate all experiment result JSONs into one summary table.

Usage:
    python scripts/summarize_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "data" / "processed" / "results"


def main() -> None:
    if not RESULTS.is_dir():
        print("No results directory yet - run a baseline first.")
        return
    rows: list[dict] = []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        # Baseline files wrap metrics under "metrics"; others are flat.
        metrics = data.get("metrics", data)
        rows.append(
            {
                "experiment": path.stem,
                "macro_f1": metrics.get("macro_f1"),
                "auc": metrics.get("auc"),
                "accuracy": metrics.get("accuracy"),
                "eval_loss": metrics.get("eval_loss"),
                "groups_audited": len(data.get("bias_audit", {})) or "",
            }
        )
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = " | ".join(k.ljust(widths[k]) for k in rows[0])
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[k]).ljust(widths[k]) for k in rows[0]))


if __name__ == "__main__":
    sys.exit(main())
