"""YAML config loading with dot-key overrides.

Configs live in ``configs/*.yaml``. Any setting can be overridden on the
command line with ``key=value`` pairs, where ``key`` uses dot notation for
nesting, e.g. ``model.name=xlm-roberta-base train.lr=2e-5``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML config file and apply ``key=value`` overrides.

    Args:
        path: Path to the YAML config file.
        overrides: List of ``dot.key=value`` strings. Values are parsed as
            YAML scalars, so ``1``, ``1e-4``, ``true``, ``null`` and quoted
            strings behave as expected.

    Returns:
        The merged config as a nested dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If an override is not ``key=value`` or names a key whose
            parent is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    for override in overrides or []:
        apply_override(config, override)
    return config


def apply_override(config: dict[str, Any], override: str) -> None:
    """Apply a single ``dot.key=value`` override to ``config`` in place."""
    if "=" not in override:
        raise ValueError(f"Invalid override {override!r}: expected 'dot.key=value' format")
    key, _, raw_value = override.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override {override!r}: empty key")

    value: Any = yaml.safe_load(raw_value)
    node: dict[str, Any] = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot override {override!r}: intermediate key {part!r} "
                "is not a mapping in the config"
            )
        node = child
    node[parts[-1]] = value


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into ``base`` and return a new dict.

    Values in ``extra`` win; nested mappings are merged key by key.
    """
    out: dict[str, Any] = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
