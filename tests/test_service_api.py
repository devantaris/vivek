"""Tests for the REST API layer (in-memory tiny model, TestClient)."""

from __future__ import annotations

import json

import pytest

from pj03.baselines.tfidf_lr import build_pipeline
from pj03.demo.predictor import DemoPredictor
from pj03.service import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    texts = [
        "kill all jews",
        "jews must go",
        "hate these jews",
        "good morning friends",
        "lovely day today",
        "so happy for you",
    ] * 4
    labels = [1, 1, 1, 0, 0, 0] * 4
    pipe = build_pipeline(char_features=True, max_features=500)
    pipe.fit(texts, labels)
    predictor = DemoPredictor.__new__(DemoPredictor)
    predictor.pipeline = pipe
    predictor.metadata = {"test_metrics": {"macro_f1": 0.99, "auc": 0.99}}

    log_file = tmp_path_factory.mktemp("logs") / "decisions.jsonl"
    app = create_app(predictor, decision_log=log_file)
    client = TestClient(app)
    client._log_file = log_file  # stash for assertions
    return client


def test_health(client) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["test_macro_f1"] == 0.99


def test_predict_hateful(client) -> None:
    res = client.post("/predict", json={"text": "kill all jews"})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == 1
    assert body["prob_hateful"] > 0.6
    assert body["decision_id"]
    assert any("jew" in r["ngram"] or "kill" in r["ngram"] for r in body["rationale"])
    assert body["disclaimer"]


def test_threshold_changes_label_not_prob(client) -> None:
    high = client.post("/predict", json={"text": "kill all jews", "threshold": 0.99})
    low = client.post("/predict", json={"text": "kill all jews", "threshold": 0.1})
    assert high.json()["label"] == 0  # 0.99 > P(hateful)
    assert low.json()["label"] == 1
    assert high.json()["prob_hateful"] == low.json()["prob_hateful"]


def test_pii_flagging_and_optional_scrub(client) -> None:
    text = "contact user@example.com about @moderator"
    plain = client.post("/predict", json={"text": text})
    scrubbed = client.post("/predict", json={"text": text, "scrub_pii": True})
    assert plain.json()["pii_in_input"] is True
    assert plain.json()["input_scrubbed_for_prediction"] is False
    assert scrubbed.json()["input_scrubbed_for_prediction"] is True


def test_decision_log_never_stores_raw_pii(client) -> None:
    client.post("/predict", json={"text": "email me at a@b.io you jerk"})
    entries = [json.loads(line) for line in client._log_file.read_text().splitlines()]
    assert entries, "decision log written"
    last = entries[-1]
    assert "a@b.io" not in json.dumps(last)  # raw PII never stored
    assert "[EMAIL]" in last["input_scrubbed"]
    assert last["decision_id"]
    assert "latency_ms" in last


def test_validation_errors(client) -> None:
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "hi", "threshold": 1.5}).status_code == 422
    assert client.post("/predict", json={"text": "x" * 30_000}).status_code == 422
