#!/usr/bin/env python
"""Run Phase 3: GRPO post-training with the composite reward.

Usage:
    python scripts/run_reward.py [--config configs/reward.yaml] [key=value ...]
    e.g. run_reward.py reward.weights.beta=0.7 reward.weights.gamma=0.5

Requires a CUDA GPU (Linux: Colab/Kaggle/lab machine). On the laptop the
script validates the config and exits with a clear message.
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

logger = get_logger("pj03.reward")


def load_grpo_dataset():
    """Load the SFT prompts joined with post_text/gold_label for the rewards."""
    from datasets import load_dataset

    path = REPO_ROOT / "data" / "processed" / "sft_hatexplain" / "train.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run python scripts/build_sft.py first.")
    ds = load_dataset("json", data_files=str(path))["train"]
    # gold_label is inside the completion; recover it from the completion text
    # to avoid a join with the schema here.
    def add_gold(rec: dict) -> dict:
        rec["gold_label"] = int(rec["completion"].split("Label:")[1].strip()[0])
        rec["post_text"] = rec["prompt"].split("Post:", 1)[1].strip()
        return rec

    ds = ds.map(add_gold)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reward.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. reward.weights.beta=0.7")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    set_seed(config.get("seed", 42))
    add_file_handler(logger, REPO_ROOT / "logs" / "reward.log")

    import torch

    if not torch.cuda.is_available():
        logger.error("No CUDA device. GRPO runs on a Linux GPU box - see docs/gpu_runbook.md.")
        raise SystemExit(1)

    from pj03.reward.train_grpo import run_grpo

    train_cfg = config["train"]
    weights = config.get("reward", {}).get("weights", {})
    logger.info("GRPO: model=%s weights=%s", train_cfg["model_name"], weights)
    dataset = load_grpo_dataset()
    logger.info("GRPO prompts: %d", len(dataset))

    metrics = run_grpo(
        model_path=train_cfg["model_name"],
        train_dataset=dataset,
        output_dir=str(REPO_ROOT / train_cfg.get("output_dir", "checkpoints/reward")),
        reward_config=config,
        num_generations=train_cfg["num_generations"],
        learning_rate=train_cfg["learning_rate"],
        kl_coef=train_cfg["kl_coef"],
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        seed=config.get("seed", 42),
    )
    results_dir = REPO_ROOT / "data" / "processed" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "grpo_final.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("Saved GRPO metrics -> %s", out_path)


if __name__ == "__main__":
    main()
