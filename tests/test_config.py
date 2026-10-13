"""Smoke tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from pj03.utils.config import apply_override, deep_merge, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_all_configs_parse() -> None:
    for path in sorted(CONFIGS.glob("*.yaml")):
        config = load_config(path)
        assert isinstance(config, dict) and config, path


def test_reward_weights_present() -> None:
    config = load_config(CONFIGS / "reward.yaml")
    weights = config["reward"]["weights"]
    assert set(weights) == {"alpha", "beta", "gamma"}
    assert all(isinstance(w, (int, float)) for w in weights.values())


def test_overrides_scalar_and_nested() -> None:
    config = load_config(CONFIGS / "baselines.yaml", ["seed=7", "baselines.tfidf_lr.C=3.5"])
    assert config["seed"] == 7
    assert config["baselines"]["tfidf_lr"]["C"] == 3.5


def test_override_coerces_types() -> None:
    config: dict = {"a": {"b": 1}}
    apply_override(config, "a.b=2.5")
    apply_override(config, "a.c=true")
    apply_override(config, "a.d=null")
    assert config["a"]["b"] == 2.5
    assert config["a"]["c"] is True
    assert config["a"]["d"] is None


def test_bad_overrides_raise() -> None:
    config: dict = {"a": 1}
    with pytest.raises(ValueError, match="key=value"):
        apply_override(config, "noequals")
    with pytest.raises(ValueError, match="not a mapping"):
        apply_override(config, "a.b.c=1")
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_deep_merge() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    assert deep_merge(base, {"a": {"y": 9}, "c": 4}) == {
        "a": {"x": 1, "y": 9},
        "b": 3,
        "c": 4,
    }
