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


def test_bias_penalty_reduces_reward_on_mere_identity_mention() -> None:
    # A candidate rationale that merely cites an identity term ("muslim") vs actual toxic slur
    post = "that muslim is an idiot"
    cand_identity = "Label: 1 | Rationale: muslim"
    cand_toxic = "Label: 1 | Rationale: idiot"

    r_identity = composite_reward(
        generation=cand_identity,
        gold_label=1,
        weights={"alpha": 1.0, "beta": 0.0, "gamma": 0.8},
    )
    r_toxic = composite_reward(
        generation=cand_toxic,
        gold_label=1,
        weights={"alpha": 1.0, "beta": 0.0, "gamma": 0.8},
    )

    # Candidate with toxic rationale should be rewarded higher than candidate merely blaming identity
    assert r_toxic > r_identity
