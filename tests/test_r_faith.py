"""Tests for the masking-based faithfulness reward."""

from __future__ import annotations

import pytest

from pj03.reward.composite import batch_rewards, composite_reward
from pj03.reward.r_faith import r_faith

POST = "you are all terrible subhuman trash and should leave"
FAITHFUL = "Label: 1\nRationale: the words subhuman trash attack the group"
UNFAITHFUL = "Label: 1\nRationale: the post contains hateful language"
NO_RATIONALE = "Label: 1"


def _probs(table: dict[str, float]):
    def predict(text: str) -> float:
        return table[text]
    return predict


class TestRFaith:
    def test_neutral_without_model_or_post(self) -> None:
        assert r_faith(FAITHFUL, 1) == 0.5
        assert r_faith(FAITHFUL, 1, predict_probs=lambda t: 0.9) == 0.5

    def test_faithful_rationale_scores_high(self) -> None:
        # Removing 'subhuman trash' drops P(hateful) 0.95 -> 0.10.
        masked = "you are all terrible and should leave"
        probs = _probs({POST: 0.95, masked: 0.10})
        assert r_faith(FAITHFUL, 1, probs, POST) == pytest.approx(0.85)

    def test_boilerplate_rationale_scores_zero(self) -> None:
        # Template rationale names no post tokens -> nothing masked -> 0.
        probs = _probs({POST: 0.9})
        assert r_faith(UNFAITHFUL, 1, probs, POST) == 0.0

    def test_missing_rationale_is_unfaithful(self) -> None:
        assert r_faith(NO_RATIONALE, 1, lambda t: 0.9, POST) == 0.0

    def test_irrelevant_cited_words_score_low(self) -> None:
        # Rationale cites a word that is in the post but non-predictive.
        gen = "Label: 1\nRationale: the word leave matters"
        masked = "you are all terrible subhuman trash and should"
        probs = _probs({POST: 0.9, masked: 0.88})
        assert r_faith(gen, 1, probs, POST) == pytest.approx(0.02)

    def test_output_in_unit_interval(self) -> None:
        probs = _probs({POST: 1.0, "you are all terrible and should leave": 0.0})
        assert 0.0 <= r_faith(FAITHFUL, 1, probs, POST) <= 1.0


class TestCompositeWithFaith:
    def test_post_text_changes_composite(self) -> None:
        masked = "you are all terrible and should leave"
        probs = _probs({POST: 0.95, masked: 0.05})
        without = composite_reward(FAITHFUL, 1)
        with_post = composite_reward(FAITHFUL, 1, predict_probs=probs, post_text=POST)
        assert with_post != without  # real faithfulness now moves the reward
        # beta=0.5, faith=0.9 -> +0.45 vs +0.25 for the neutral 0.5
        assert with_post > without

    def test_batch_rewards_accepts_post_texts(self) -> None:
        masked = "you are all terrible and should leave"
        probs = _probs({POST: 0.95, masked: 0.05})
        rewards = batch_rewards(
            [FAITHFUL, UNFAITHFUL], [1, 1], predict_probs=probs, post_texts=[POST, POST]
        )
        assert len(rewards) == 2
        assert rewards[0] > rewards[1]  # faithful beats boilerplate

        with pytest.raises(ValueError):
            batch_rewards([FAITHFUL], [1], post_texts=[POST, POST])
