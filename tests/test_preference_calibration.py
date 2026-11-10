"""Unit tests for Phase 3 reward preference calibration."""

from __future__ import annotations

from pj03.reward.composite import composite_reward


def test_preference_reward_margin() -> None:
    # A faithful hate completion vs a wrong non-hate completion on true hate post
    post = "kill all these disgusting vermin right now"
    cand_correct = "Label: 1 | Rationale: disgusting vermin kill"
    cand_wrong = "Label: 0 | Rationale: post is polite and harmless"

    r_correct = composite_reward(
        generation=cand_correct,
        gold_label=1,
        weights={"alpha": 1.0, "beta": 0.5, "gamma": 0.3},
        post_text=post,
    )
    r_wrong = composite_reward(
        generation=cand_wrong,
        gold_label=1,
        weights={"alpha": 1.0, "beta": 0.5, "gamma": 0.3},
        post_text=post,
    )

    # Correct candidate must have strictly higher reward than wrong candidate
    assert r_correct > r_wrong
    margin = r_correct - r_wrong
    assert margin > 0.5


def test_faithful_rationale_reward_higher_than_boilerplate() -> None:
    post = "kill all these disgusting vermin right now"
    cand_faithful = "Label: 1 | Rationale: disgusting vermin"
    cand_boilerplate = "Label: 1 | Rationale: offensive content detected"

    r_faithful = composite_reward(
        generation=cand_faithful,
        gold_label=1,
        weights={"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
        post_text=post,
    )
    r_boilerplate = composite_reward(
        generation=cand_boilerplate,
        gold_label=1,
        weights={"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
        post_text=post,
    )

    # Faithful rationale must yield strictly higher composite reward than boilerplate
    assert r_faithful >= r_boilerplate
