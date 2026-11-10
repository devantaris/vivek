"""Unit tests for the active learning retraining engine."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.retrain_from_feedback import extract_feedback_samples


def test_extract_feedback_samples(tmp_path: Path) -> None:
    decisions_file = tmp_path / "decisions.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"

    decisions_file.write_text(
        json.dumps({"decision_id": "d1", "input_scrubbed": "hello world", "label": 0}) + "\n"
        + json.dumps({"decision_id": "d2", "input_scrubbed": "you are stupid", "label": 0}) + "\n",
        encoding="utf-8",
    )

    feedback_file.write_text(
        json.dumps({"decision_id": "d1", "model_label": 0, "final_label": 0, "agrees": True}) + "\n"
        + json.dumps({"decision_id": "d2", "model_label": 0, "final_label": 1, "agrees": False}) + "\n",
        encoding="utf-8",
    )

    samples = extract_feedback_samples(feedback_file, decisions_file)
    assert len(samples) == 2

    # First is confirmation
    assert samples[0] == ("hello world", 0, False)
    # Second is override (hard example)
    assert samples[1] == ("you are stupid", 1, True)
