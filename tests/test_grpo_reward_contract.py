"""CPU validation of the TRL GRPO reward-function contract.

TRL 1.15 calls reward functions as
``fn(prompts=..., completions=..., completion_ids=..., **reward_kwargs)``
where ``reward_kwargs`` contains every extra dataset column as a list PLUS
trainer utilities (``trainer_state``, ``log_extra``, ``log_metric``). These
tests simulate that exact call so the contract can be validated on a laptop
before any GPU session.
"""

from __future__ import annotations

import pytest

from pj03.reward.train_grpo import make_reward_fn

POST = "you are terrible subhuman trash"
FAITHFUL = "Label: 1\nRationale: subhuman trash is a slur"
BOILERPLATE = "Label: 1\nRationale: the post contains hateful language"
WRONG = "Label: 0\nRationale: nothing wrong here"


def _call_like_trl(reward_fn, prompts, completions, columns: dict) -> list[float]:
    """Mirror GRPOTrainer._calculate_rewards' invocation."""
    reward_kwargs = {key: list(values) for key, values in columns.items()}
    reward_kwargs["trainer_state"] = object()  # utility kwarg must be absorbed
    reward_kwargs["log_extra"] = None
    reward_kwargs["log_metric"] = None
    return reward_fn(
        prompts=prompts,
        completions=completions,
        completion_ids=[[]] * len(completions),
        **reward_kwargs,
    )


def _predictor():
    """Toy P(hateful): high only when the post's key slur is present."""
    return lambda text: 0.9 if "subhuman trash" in text else 0.3


class TestTRLContract:
    def test_signature_matches_trl_call(self) -> None:
        fn = make_reward_fn(predictor_builder=_predictor)
        rewards = _call_like_trl(
            fn,
            prompts=["prompt-a", "prompt-b", "prompt-c"],
            completions=[FAITHFUL, BOILERPLATE, WRONG],
            columns={"gold_label": [1, 1, 1], "post_text": [POST, POST, POST]},
        )
        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        # Correct label + faithful rationale beats boilerplate beats wrong label.
        assert rewards[0] > rewards[1] > rewards[2]

    def test_missing_columns_default_to_zero_gold(self) -> None:
        fn = make_reward_fn()
        # TRL forwards whatever the dataset has; a dataset without post_text
        # (e.g. during a dry run) must still yield finite rewards.
        rewards = _call_like_trl(fn, prompts=["p"], completions=[FAITHFUL], columns={})
        assert len(rewards) == 1
        assert rewards[0] == rewards[0]  # finite (not NaN)

    def test_rewards_can_be_negative(self) -> None:
        # Bias penalty can outweigh positives: rationale that is pure
        # identity-term citation with everything else wrong.
        fn = make_reward_fn({"alpha": 1.0, "beta": 0.0, "gamma": 1.0})
        gen = "Label: 1\nRationale: muslim"
        rewards = _call_like_trl(fn, prompts=["p"], completions=[gen], columns={"gold_label": [1]})
        # r_label=1, r_bias=1-1/1=0 -> R = 1 - 0 = 1; not negative here, but
        # boilerplate-only wrong-label cases go negative:
        rewards_bad = _call_like_trl(
            fn, prompts=["p"], completions=[WRONG], columns={"gold_label": [1]}
        )
        assert rewards_bad[0] < 0  # label 0 + r_bias 1.0 -> 0 - 1.0
        assert isinstance(rewards[0], float)

    def test_weights_change_ranking(self) -> None:
        cols = {"gold_label": [1, 1], "post_text": [POST, POST]}
        beta1 = _call_like_trl(
            make_reward_fn({"alpha": 1.0, "beta": 1.0, "gamma": 0.0}, _predictor),
            prompts=["p", "p"],
            completions=[FAITHFUL, BOILERPLATE],
            columns=cols,
        )
        beta_half = _call_like_trl(
            make_reward_fn({"alpha": 1.0, "beta": 0.5, "gamma": 0.0}, _predictor),
            prompts=["p", "p"],
            completions=[FAITHFUL, BOILERPLATE],
            columns=cols,
        )
        # The faithful-vs-boilerplate gap scales with beta.
        assert (beta1[0] - beta1[1]) == pytest.approx(2 * (beta_half[0] - beta_half[1]))


@pytest.mark.parametrize("weights", [None, {"alpha": 1.0, "beta": 0.5, "gamma": 0.3}])
def test_reward_fn_returns_one_float_per_completion(weights) -> None:
    fn = make_reward_fn(weights)
    out = fn(prompts=["p"] * 4, completions=[FAITHFUL] * 4, gold_label=[1] * 4)
    assert len(out) == 4 and all(isinstance(r, float) for r in out)


def test_resolve_model_and_adapter_directory(tmp_path) -> None:
    from pj03.reward.train_grpo import _resolve_model_and_adapter

    # Huggingface model ID string
    base, adapter = _resolve_model_and_adapter("Qwen/Qwen2.5-1.5B-Instruct")
    assert base == "Qwen/Qwen2.5-1.5B-Instruct"
    assert adapter is None

    # Mock adapter directory
    adapter_dir = tmp_path / "mock_adapter"
    adapter_dir.mkdir()
    cfg_file = adapter_dir / "adapter_config.json"
    cfg_file.write_text('{"base_model_name_or_path": "custom/base-model"}', encoding="utf-8")

    base, adapter = _resolve_model_and_adapter(str(adapter_dir))
    assert base == "custom/base-model"
    assert adapter == str(adapter_dir)
