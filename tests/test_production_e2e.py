"""Test end-to-end production pipeline via FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path

from scripts.test_production_e2e import run_e2e_verification


def test_production_e2e_suite(tmp_path: Path) -> None:
    report = run_e2e_verification(tmp_path=tmp_path)
    assert report["status"] == "ALL_TESTS_PASSED"
    assert report["results"]["health_check"] is True
    assert report["results"]["english_accuracy"] is True
    assert report["results"]["hinglish_accuracy"] is True
    assert report["results"]["pii_scrubbing"] is True
    assert report["results"]["moderation_queue_cycle"] is True
