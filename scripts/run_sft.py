#!/usr/bin/env python
"""Run Phase 2: LoRA SFT of a small instruction-tuned LLM.

Usage:
    python scripts/run_sft.py [--config configs/sft.yaml] [key=value ...]
    python scripts/run_sft.py sft.smoke.enabled=true   # CPU path validation

Full runs need a GPU; the smoke config validates the peft/trl path on CPU
with a tiny GPT-2 on a few dozen SFT records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.utils.config import load_config  # noqa: E402
from pj03.utils.logging import add_file_handler, get_logger  # noqa: E402
from pj03.utils.seed import set_seed  # noqa: E402

logger = get_logger("pj03.sft")


def load_sft_split(name: str, n: int | None = None):
    """Load a prompt/completion SFT split as an HF Dataset (trimmed to n)."""
    from datasets import load_dataset

    path = REPO_ROOT / "data" / "processed" / "sft_hatexplain" / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found - run python scripts/build_sft.py first."
        )
    ds = load_dataset("json", data_files=str(path))["train"]
    return ds.select(range(min(n, len(ds)))) if n else ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. sft.epochs=5")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    add_file_handler(logger, REPO_ROOT / "logs" / "sft.log")

    from pj03.sft.lora_sft import run_lora_sft

    sft_cfg = config["sft"]
    smoke = sft_cfg.get("smoke", {})
    if smoke.get("enabled", False):
        model_name = smoke["model_name"]
        n_train, epochs = smoke.get("n_train", 64), smoke.get("epochs", 1)
        batch_size = smoke.get("batch_size", 8)
        max_length = smoke.get("max_length", 192)
        lora_r, lora_alpha = smoke.get("lora_r", 8), smoke.get("lora_alpha", 16)
        target_modules = smoke.get("target_modules", ["c_attn"])
        output_dir = REPO_ROOT / "checkpoints" / "sft_smoke"
        logger.info("SMOKE mode: %s on %d records (CPU path validation)", model_name, n_train)
    else:
        model_name = sft_cfg["model_name"]
        n_train, epochs = None, sft_cfg["epochs"]
        batch_size = sft_cfg["batch_size"]
        max_length = sft_cfg["max_length"]
        lora_r = sft_cfg["lora"]["r"]
        lora_alpha = sft_cfg["lora"]["alpha"]
        target_modules = sft_cfg["lora"]["target_modules"]
        output_dir = REPO_ROOT / sft_cfg.get("output_dir", "checkpoints/sft")
        logger.info("Full SFT: %s (%d epochs)", model_name, epochs)

    train_ds = load_sft_split("train", n_train)
    eval_ds = load_sft_split("validation", 32 if smoke.get("enabled", False) else None)
    logger.info("SFT records: train=%d eval=%d", len(train_ds), len(eval_ds))

    metrics = run_lora_sft(
        model_name=model_name,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=str(output_dir),
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        learning_rate=sft_cfg["learning_rate"],
        batch_size=batch_size,
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation", 2),
        epochs=epochs,
        max_length=max_length,
        seed=config.get("seed", 42),
        save_adapter=True,
    )
    results_dir = REPO_ROOT / "data" / "processed" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if smoke.get("enabled", False) else "full"
    out_path = results_dir / f"sft_{model_name.replace('/', '_')}_{suffix}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved SFT metrics -> %s", out_path)


if __name__ == "__main__":
    main()
