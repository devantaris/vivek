"""Drift monitoring for the serving decision log.

Computes operational and model-behaviour statistics over the accumulated
decisions and raises alerts when configured thresholds are crossed. Pure
functions over JSONL so it can run in CI as well as from the script.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from pydantic import BaseModel, Field


class MonitorConfig(BaseModel):
    """Alert thresholds (None = check disabled)."""

    min_decisions: int = 10  # below this, skip behaviour checks
    hateful_rate_max: float | None = 0.9
    hateful_rate_min: float | None = 0.02
    pii_rate_max: float | None = 0.5
    latency_p95_ms_max: float | None = 2000.0
    override_rate_max: float | None = 0.4


class DriftReport(BaseModel):
    """One monitoring pass over the decision log."""

    n_decisions: int = 0
    hateful_rate: float | None = None
    pii_rate: float | None = None
    mean_prob: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    override_rate: float | None = None
    by_date: dict[str, dict[str, float]] = Field(default_factory=dict)
    alerts: list[str] = Field(default_factory=list)
    ok: bool = True


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))
    return ordered[idx]


def analyse(decisions_path: str | Path, config: MonitorConfig | None = None) -> DriftReport:
    """Compute the drift report for a decision log.

    Args:
        decisions_path: JSONL decision log written by the API.
        config: Alert thresholds; defaults used if omitted.

    Returns:
        :class:`DriftReport` with ``alerts`` populated and ``ok`` False if
        any threshold is crossed.
    """
    config = config or MonitorConfig()
    path = Path(decisions_path)
    report = DriftReport()
    if not path.is_file():
        report.alerts.append("no decision log found")
        report.ok = False
        return report

    entries: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    report.n_decisions = len(entries)
    if not entries:
        report.alerts.append("decision log is empty")
        report.ok = False
        return report

    probs = [e["prob_hateful"] for e in entries]
    labels = [e["label"] for e in entries]
    latencies = [e["latency_ms"] for e in entries if "latency_ms" in e]
    pii = [1.0 if e.get("pii_in_input") else 0.0 for e in entries]
    report.hateful_rate = sum(labels) / len(labels)
    report.pii_rate = sum(pii) / len(pii)
    report.mean_prob = statistics.fmean(probs)
    if latencies:
        report.latency_p50_ms = _percentile(latencies, 0.5)
        report.latency_p95_ms = _percentile(latencies, 0.95)

    by_date: dict[str, list[dict]] = {}
    for e in entries:
        by_date.setdefault(str(e.get("ts", ""))[:10], []).append(e)
    for day, day_entries in sorted(by_date.items()):
        report.by_date[day] = {
            "n": len(day_entries),
            "hateful_rate": sum(e["label"] for e in day_entries) / len(day_entries),
        }

    # Feedback store may live next to the decision log.
    feedback_path = path.with_name("feedback.jsonl")
    if feedback_path.is_file():
        records = [
            json.loads(line) for line in feedback_path.open(encoding="utf-8") if line.strip()
        ]
        if records:
            overrides = sum(1 for r in records if not r["agrees"])
            report.override_rate = overrides / len(records)

    if report.n_decisions >= config.min_decisions:
        if config.hateful_rate_max is not None and report.hateful_rate > config.hateful_rate_max:
            report.alerts.append(
                f"hateful rate {report.hateful_rate:.2f} > max {config.hateful_rate_max}"
            )
        if config.hateful_rate_min is not None and report.hateful_rate < config.hateful_rate_min:
            report.alerts.append(
                f"hateful rate {report.hateful_rate:.2f} < min {config.hateful_rate_min}"
            )
        if config.pii_rate_max is not None and report.pii_rate > config.pii_rate_max:
            report.alerts.append(f"PII rate {report.pii_rate:.2f} > max {config.pii_rate_max}")
    if (
        config.latency_p95_ms_max is not None
        and report.latency_p95_ms is not None
        and report.latency_p95_ms > config.latency_p95_ms_max
    ):
        report.alerts.append(
            f"latency p95 {report.latency_p95_ms:.0f}ms > max {config.latency_p95_ms_max}ms"
        )
    if config.override_rate_max is not None and report.override_rate is not None:
        if report.override_rate > config.override_rate_max:
            report.alerts.append(
                f"override rate {report.override_rate:.2f} > max {config.override_rate_max}"
            )
    report.ok = not report.alerts
    return report
