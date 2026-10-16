"""Tests for the demo predictor (uses a tiny fitted pipeline, no artifact IO)."""

from __future__ import annotations

import pytest

from pj03.baselines.tfidf_lr import build_pipeline
from pj03.demo.predictor import DemoPredictor


@pytest.fixture(scope="module")
def predictor() -> DemoPredictor:
    texts = [
        "kill all jews",
        "jews should leave",
        "hate these jews",
        "good morning friends",
        "lovely day outside",
        "happy for you all",
    ] * 4
    labels = [1, 1, 1, 0, 0, 0] * 4
    pipe = build_pipeline(char_features=True, max_features=500)
    pipe.fit(texts, labels)
    p = DemoPredictor.__new__(DemoPredictor)
    p.pipeline = pipe
    p.metadata = {"test_metrics": {"macro_f1": 1.0}}
    return p


def test_hateful_prediction_with_rationale(predictor: DemoPredictor) -> None:
    result = predictor.predict("kill all jews")
    assert result.label == 1
    assert result.prob_hateful > 0.6
    top = [ngram for ngram, _ in result.rationale]
    assert any("jew" in n or "kill" in n for n in top)


def test_benign_prediction(predictor: DemoPredictor) -> None:
    result = predictor.predict("good morning friends")
    assert result.label == 0
    assert result.prob_hateful < 0.4


def test_faithfulness_scores_present(predictor: DemoPredictor) -> None:
    result = predictor.predict("kill all jews")
    assert result.comprehensiveness is not None
    assert result.sufficiency is not None
    # Removing the key words must move the probability (comp > 0).
    assert result.comprehensiveness > 0.0


def test_alignment_length_matches_tokens(predictor: DemoPredictor) -> None:
    tokens = "kill all jews now".split()
    scores = predictor._align_to_tokens(tokens, [("kill", 0.5), ("all jews", 0.3)])
    assert len(scores) == len(tokens)
    assert scores[0] == pytest.approx(0.5)  # 'kill'
    assert scores[2] != 0.0  # 'jews' via bigram 'all jews'
    assert scores[3] == 0.0  # 'now' unused by the model
