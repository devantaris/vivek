#!/usr/bin/env python
"""Drift report over the serving decision log.

Usage:
    python scripts/drift_report.py [--decisions logs/decisions.jsonl]

Writes ``data/processed/results/drift_report.json`` and exits 1 if any
alert threshold is crossed (CI-friendly).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", default="logs/decisions.jsonl")
    args = parser.parse_args()

    from pj03.service.monitor import MonitorConfig, analyse
    from pj03.utils.logging import get_logger

    logger = get_logger("pj03.monitor")
    report = analyse(REPO_ROOT / args.decisions, MonitorConfig())
    payload = report.model_dump()
    out = REPO_ROOT / "data" / "processed" / "results" / "drift_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(
        "decisions=%d hateful_rate=%s pii_rate=%s p95=%sms override_rate=%s",
        report.n_decisions,
        f"{report.hateful_rate:.2f}" if report.hateful_rate is not None else "—",
        f"{report.pii_rate:.2f}" if report.pii_rate is not None else "—",
        f"{report.latency_p95_ms:.0f}" if report.latency_p95_ms is not None else "—",
        f"{report.override_rate:.2f}" if report.override_rate is not None else "—",
    )
    for alert in report.alerts:
        logger.warning("ALERT: %s", alert)
    logger.info("Drift report -> %s (ok=%s)", out, report.ok)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
