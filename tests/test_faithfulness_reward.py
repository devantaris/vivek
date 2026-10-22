"""Smoke tests for faithfulness metrics and the composite reward."""

from __future__ import annotations

import pytest

from pj03.reward.composite import batch_rewards, composite_reward
from pj03.reward.r_bias import r_bias
from pj03.reward.r_label import r_label
from pj03.xai.faithfulness import comprehensiveness, sufficiency

GOOD = "Label: 1\nRationale: the post uses a slur against women"
BAD = "Label: 0\nRationale: friendly greeting only"
UNPARSEABLE = "I think this post is probably hateful."


class TestFaithfulness:
    def test_comprehensiveness_positive_when_important_removed(self) -> None:
        tokens = ["i", "hate", "all", "of", "them"]
        attribution = [0.0, 5.0, 4.0, 0.0, 0.1]
        probs = {" ".join(tokens): 0.9, "i of them": 0.1, "i all of them": 0.6}
        comp = comprehensiveness(lambda t: probs[t], tokens, attribution, k=2)
        assert comp == pytest.approx(0.9 - 0.1)

    def test_sufficiency_keeps_prediction(self) -> None:
        tokens = ["i", "hate", "all", "of", "them"]
        attribution = [0.0, 5.0, 4.0, 0.0, 0.1]
        probs = {" ".join(tokens): 0.9, "hate all": 0.85}
        suff = sufficiency(lambda t: probs[t], tokens, attribution, k=2)
        assert suff == pytest.approx(0.05)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            comprehensiveness(lambda t: 0.5, ["a"], [0.1, 0.2])
        with pytest.raises(ValueError):
            sufficiency(lambda t: 0.5, ["a", "b"], [0.1], k=5)


class TestCompositeReward:
    def test_components_in_unit_interval(self) -> None:
        for gen in (GOOD, BAD, UNPARSEABLE):
            assert 0.0 <= r_bias(gen) <= 1.0
        assert r_label(GOOD, 1) == 1.0
        assert r_label(GOOD, 0) == 0.0
        assert r_label(UNPARSEABLE, 1) == 0.0

    def test_composite_known_value(self) -> None:
        # default weights alpha=1, beta=0.5, gamma=0.3
        # r_label(GOOD,1)=1, r_faith placeholder=0.5; GOOD's rationale
        # ("the post uses a slur against women") has 7 tokens, 1 identity
        # term ("women") -> r_bias = 1 - 1/7 = 6/7.
        value = composite_reward(GOOD, 1)
        assert value == pytest.approx(1.0 + 0.5 * 0.5 - 0.3 * (6 / 7))

    def test_bias_penalty_kicks_in(self) -> None:
        gen = "Label: 1\nRationale: mentions muslim people"
        value_with_bias = composite_reward(gen, 1)
        value_no_gamma = composite_reward(gen, 1, weights={"gamma": 0.0})
        assert value_with_bias < value_no_gamma

    def test_batch_rewards(self) -> None:
        rewards = batch_rewards([GOOD, BAD], [1, 0])
        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)
        with pytest.raises(ValueError):
            batch_rewards([GOOD], [1, 0])
