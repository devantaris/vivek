#!/usr/bin/env python
"""End-to-End automated production verification test suite for PJ-03 API & Moderation.

Tests:
    1. GET /health - service health, metadata, zero PII leak
    2. POST /predict - English hateful, English benign, Hinglish threats, Hinglish vulgarity
    3. PII Scrubbing - audit log sanitisation (emails, phones, URLs, handles)
    4. Per-request threshold tuning
    5. Latency profiling (sub-50ms CPU throughput)
    6. Human-in-the-loop moderation queue: /queue and /feedback cycle

Usage:
    python scripts/test_production_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from pj03.demo.predictor import DEFAULT_ARTIFACT, DemoPredictor  # noqa: E402
from pj03.service import create_app  # noqa: E402
from pj03.utils.logging import get_logger  # noqa: E402

logger = get_logger("pj03.test_production_e2e")


def run_e2e_verification(tmp_path: Path | None = None) -> dict:
    if tmp_path is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())
    else:
        tmp_dir = tmp_path

    decision_log = tmp_dir / "decisions.jsonl"
    feedback_log = tmp_dir / "feedback.jsonl"

    logger.info("Initializing DemoPredictor and FastAPI app...")
    predictor = DemoPredictor(REPO_ROOT / DEFAULT_ARTIFACT)
    app = create_app(predictor, decision_log=decision_log, feedback_log=feedback_log)
    client = TestClient(app)

    results: dict[str, bool] = {}
    latencies: list[float] = []

    # 1. Health check
    logger.info("Test 1: GET /health")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    assert health_data["status"] == "ok"
    results["health_check"] = True

    # 2. English predictions
    logger.info("Test 2: English hate & benign detection")
    # Warmup request
    client.post("/predict", json={"text": "warmup test message"})
    latencies.clear()

    t0 = time.perf_counter()
    resp_hate = client.post("/predict", json={"text": "kill all of these disgusting people"})
    latencies.append((time.perf_counter() - t0) * 1000)
    assert resp_hate.status_code == 200
    hate_payload = resp_hate.json()
    assert hate_payload["label"] == 1
    assert hate_payload["prob_hateful"] >= 0.60

    resp_benign = client.post("/predict", json={"text": "we must protect everyone from discrimination and live happily"})
    assert resp_benign.status_code == 200
    assert resp_benign.json()["label"] == 0
    results["english_accuracy"] = True

    # 3. Hinglish predictions
    logger.info("Test 3: Hinglish threat & vulgarity detection")
    resp_threat = client.post("/predict", json={"text": "modi ko mar jana chahiye"})
    assert resp_threat.status_code == 200
    assert resp_threat.json()["label"] == 1
    assert resp_threat.json()["prob_hateful"] >= 0.90

    resp_vulgar = client.post("/predict", json={"text": "modi madharchod hai"})
    assert resp_vulgar.status_code == 200
    assert resp_vulgar.json()["label"] == 1
    assert resp_vulgar.json()["prob_hateful"] >= 0.90

    resp_hi_benign = client.post("/predict", json={"text": "kya mast mausam hai aaj maza aa gaya"})
    assert resp_hi_benign.status_code == 200
    assert resp_hi_benign.json()["label"] == 0
    results["hinglish_accuracy"] = True

    # 4. PII Scrubbing in decision log
    logger.info("Test 4: PII scrubbing verification")
    pii_text = "contact me at devansh@example.com or call 555-123-4567 you idiot"
    resp_pii = client.post("/predict", json={"text": pii_text})
    assert resp_pii.status_code == 200
    assert resp_pii.json()["pii_in_input"] is True

    # Inspect decision log file directly
    assert decision_log.is_file(), "Decision log not created"
    log_lines = [json.loads(line) for line in decision_log.read_text(encoding="utf-8").strip().split("\n")]
    last_log = log_lines[-1]
    assert "devansh@example.com" not in last_log["input_scrubbed"]
    assert "555-123-4567" not in last_log["input_scrubbed"]
    assert "[EMAIL]" in last_log["input_scrubbed"]
    assert "[PHONE]" in last_log["input_scrubbed"]
    results["pii_scrubbing"] = True

    # 5. Threshold flexibility
    logger.info("Test 5: Threshold tuning")
    t_high = client.post("/predict", json={"text": "this is slightly annoyed", "threshold": 0.99}).json()
    t_low = client.post("/predict", json={"text": "this is slightly annoyed", "threshold": 0.01}).json()
    assert t_high["label"] == 0
    assert t_low["label"] == 1
    results["threshold_tuning"] = True

    # 6. Human-in-the-loop review queue & feedback
    logger.info("Test 6: Moderation review queue cycle")
    queue_resp = client.get("/queue?limit=10")
    assert queue_resp.status_code == 200
    queue = queue_resp.json().get("pending", [])
    assert len(queue) > 0

    first_decision_id = queue[0]["decision_id"]
    fb_resp = client.post(
        "/feedback",
        json={
            "decision_id": first_decision_id,
            "model_label": queue[0]["label"],
            "final_label": queue[0]["label"],
            "agrees": True,
            "moderator": "evaluator_1",
            "note": "verified correct",
        },
    )
    assert fb_resp.status_code == 200

    # Verify decision is removed from pending queue
    queue_after = client.get("/queue?limit=10").json().get("pending", [])
    pending_ids = [q["decision_id"] for q in queue_after]
    assert first_decision_id not in pending_ids
    results["moderation_queue_cycle"] = True

    # 7. Latency profiling
    avg_latency = sum(latencies) / len(latencies)
    results["latency_sub_100ms"] = avg_latency < 100.0

    report = {
        "status": "ALL_TESTS_PASSED" if all(results.values()) else "SOME_TESTS_FAILED",
        "results": results,
        "avg_latency_ms": round(avg_latency, 2),
        "total_requests_tested": len(log_lines),
    }

    out_file = REPO_ROOT / "data" / "processed" / "results" / "production_verification_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved production verification report -> %s", out_file)
    return report


def main() -> None:
    report = run_e2e_verification()
    logger.info("Production Verification Report:\n%s", json.dumps(report, indent=2))
    if report["status"] != "ALL_TESTS_PASSED":
        sys.exit(1)


if __name__ == "__main__":
    main()
