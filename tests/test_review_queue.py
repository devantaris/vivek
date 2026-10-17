"""Tests for the human-in-the-loop feedback store and review endpoints."""

from __future__ import annotations

import json

import pytest

from pj03.baselines.tfidf_lr import build_pipeline
from pj03.demo.predictor import DemoPredictor
from pj03.service import create_app
from pj03.service.feedback import FeedbackRecord, FeedbackStore


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    texts = ["kill all jews", "jews must go", "hate these jews",
             "good morning friends", "lovely day today", "so happy for you"] * 4
    labels = [1, 1, 1, 0, 0, 0] * 4
    pipe = build_pipeline(char_features=True, max_features=500)
    pipe.fit(texts, labels)
    predictor = DemoPredictor.__new__(DemoPredictor)
    predictor.pipeline = pipe
    predictor.metadata = {"test_metrics": {"macro_f1": 0.99}}

    tmp = tmp_path_factory.mktemp("review")
    app = create_app(
        predictor,
        decision_log=tmp / "decisions.jsonl",
        feedback_log=tmp / "feedback.jsonl",
    )
    client = TestClient(app)
    return client, tmp / "feedback.jsonl", tmp / "decisions.jsonl"


def test_full_review_cycle(env) -> None:
    client, feedback_path, _ = env
    # 1. Two predictions land in the log.
    d1 = client.post("/predict", json={"text": "kill all jews"}).json()["decision_id"]
    d2 = client.post("/predict", json={"text": "good morning friends"}).json()["decision_id"]
    # 2. Both appear in the queue.
    queue = client.get("/queue").json()["pending"]
    ids = {e["decision_id"] for e in queue}
    assert {d1, d2} <= ids
    # 3. Confirm one, override the other.
    model1 = client.get("/queue").json()["pending"][0]["label"]
    r1 = client.post(
        "/feedback",
        json={"decision_id": d1, "model_label": model1, "final_label": model1,
              "agrees": True, "moderator": "devansh"},
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/feedback",
        json={"decision_id": d2, "model_label": 0, "final_label": 1,
              "agrees": False, "moderator": "devansh", "note": "missed slur"},
    )
    assert r2.status_code == 200
    # 4. Queue is now empty; override rate 1/2.
    assert client.get("/queue").json()["pending"] == []
    store = FeedbackStore(feedback_path)
    assert store.override_rate() == pytest.approx(0.5)


def test_feedback_unknown_decision_404(env) -> None:
    client, _, _ = env
    res = client.post(
        "/feedback",
        json={"decision_id": "nope", "model_label": 0, "final_label": 0,
              "agrees": True, "moderator": "x"},
    )
    assert res.status_code == 404


def test_feedback_invalid_labels_422(env) -> None:
    client, _, _ = env
    # Unknown id would 404 first; use a real prediction then bad final_label
    # via direct store validation instead: pydantic model allows any int, so
    # the endpoint check is what matters. Get a real decision id:
    did = client.post("/predict", json={"text": "hello team"}).json()["decision_id"]
    res = client.post(
        "/feedback",
        json={"decision_id": did, "model_label": 0, "final_label": 7,
              "agrees": False, "moderator": "x"},
    )
    assert res.status_code == 422


def test_store_persists_and_pending_skips_reviewed(tmp_path) -> None:
    decisions = tmp_path / "d.jsonl"
    with decisions.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"decision_id": "a", "label": 1}) + "\n")
        fh.write(json.dumps({"decision_id": "b", "label": 0}) + "\n")
    store = FeedbackStore(tmp_path / "f.jsonl")
    store.append(
        FeedbackRecord(decision_id="b", model_label=0, final_label=1,
                       agrees=False, moderator="m")
    )
    pending = store.pending_queue(decisions)
    assert [e["decision_id"] for e in pending] == ["a"]  # newest first, b reviewed


def test_empty_logs_are_safe(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "f.jsonl")
    assert store.pending_queue(tmp_path / "missing.jsonl") == []
    assert store.override_rate() is None
    assert store.reviewed_ids() == set()
