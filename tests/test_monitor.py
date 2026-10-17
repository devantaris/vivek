"""Tests for the drift monitor."""

from __future__ import annotations

import json

import pytest

from pj03.service.monitor import DriftReport, MonitorConfig, analyse


def write_log(path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def make_entries(n: int, prob: float = 0.9, label: int = 1, latency: float = 50.0) -> list[dict]:
    return [
        {
            "ts": "2026-10-17T10:00:00",
            "decision_id": f"d{i}",
            "input_scrubbed": "text",
            "label": label,
            "prob_hateful": prob,
            "latency_ms": latency,
            "pii_in_input": i % 4 == 0,
        }
        for i in range(n)
    ]


def test_basic_stats(tmp_path) -> None:
    log = tmp_path / "decisions.jsonl"
    entries = make_entries(20)
    for i, e in enumerate(entries):
        e["label"] = 1 if i % 5 == 0 else 0  # 20% hateful: inside default bounds
        e["prob_hateful"] = 0.3 if e["label"] == 0 else 0.9
    write_log(log, entries)
    report = analyse(log, MonitorConfig())
    assert report.n_decisions == 20
    assert report.hateful_rate == 0.2
    assert report.pii_rate == 0.25
    assert report.mean_prob == pytest.approx(0.4)
    assert report.latency_p50_ms == 50.0
    assert report.latency_p95_ms == 50.0
    assert report.by_date["2026-10-17"]["n"] == 20
    assert report.ok


def test_alert_on_hateful_rate_bounds(tmp_path) -> None:
    log = tmp_path / "decisions.jsonl"
    write_log(log, make_entries(20, label=0))
    cfg = MonitorConfig(hateful_rate_min=0.02, hateful_rate_max=0.9)
    report = analyse(log, cfg)
    assert not report.ok
    assert any("hateful rate" in a for a in report.alerts)

    write_log(log, make_entries(20, label=1))
    report = analyse(log, cfg)
    assert any("hateful rate" in a and ">" in a for a in report.alerts)


def test_alert_on_pii_and_latency(tmp_path) -> None:
    log = tmp_path / "decisions.jsonl"
    entries = make_entries(20, latency=5000.0)
    for e in entries:
        e["pii_in_input"] = True
    write_log(log, entries)
    report = analyse(log, MonitorConfig(pii_rate_max=0.5, latency_p95_ms_max=2000.0))
    assert any("PII rate" in a for a in report.alerts)
    assert any("latency p95" in a for a in report.alerts)
    assert not report.ok


def test_override_rate_alert(tmp_path) -> None:
    log = tmp_path / "decisions.jsonl"
    feedback = tmp_path / "feedback.jsonl"
    write_log(log, make_entries(20))
    with feedback.open("w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(
                json.dumps(
                    {"decision_id": str(i), "model_label": 1, "final_label": 0,
                     "agrees": i < 2, "moderator": "m", "note": "", "ts": "t"}
                )
                + "\n"
            )
    report = analyse(log, MonitorConfig(override_rate_max=0.4))
    # 8/10 = 0.8 override rate -> alert
    assert report.override_rate == 0.8
    assert any("override rate" in a for a in report.alerts)


def test_small_log_skips_behaviour_checks(tmp_path) -> None:
    log = tmp_path / "decisions.jsonl"
    write_log(log, make_entries(5, label=0))
    report = analyse(log, MonitorConfig(min_decisions=10, hateful_rate_min=0.02))
    assert report.ok and not report.alerts


def test_missing_log_fails_safe(tmp_path) -> None:
    report = analyse(tmp_path / "missing.jsonl")
    assert not report.ok
    assert report.n_decisions == 0
    assert isinstance(report, DriftReport)
