#!/usr/bin/env python
"""Colab-optimized True Online GRPO training script for PJ-03.

Designed specifically to run on Google Colab (T4 or A100 GPU) with memory-safe
quantization and small group sizes (G=4), optimizing policy directly against
the composite multi-objective reward.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.utils.logging import get_logger  # noqa: E402

logger = get_logger("pj03.train_grpo_gpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab GPU GRPO Training")
    parser.add_argument("--steps", type=int, default=500, help="Total GRPO optimization steps")
    parser.add_argument("--group-size", type=int, default=4, help="Group rollout size (G)")
    parser.add_argument("--lr", type=float, default=1e-6, help="Policy learning rate")
    parser.add_argument("--beta", type=float, default=0.04, help="KL divergence penalty coefficient")
    parser.add_argument("--output-dir", default="checkpoints/grpo_policy")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is required for online GRPO. Please enable GPU in Colab (Runtime -> Change runtime type -> T4 GPU).")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    logger.info("Starting online GRPO on GPU: %s", device_name)

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from pj03.reward.train_grpo import make_reward_fn

    sft_data_path = REPO_ROOT / "data" / "processed" / "sft_hatexplain" / "train.jsonl"
    if not sft_data_path.is_file():
        logger.info("Building SFT dataset first...")
        import subprocess
        subprocess.run([sys.executable, "scripts/build_sft.py"], check=True)

    ds = load_dataset("json", data_files=str(sft_data_path))["train"]

    def add_columns(rec: dict) -> dict:
        rec["gold_label"] = int(rec["completion"].split("Label:")[1].strip()[0])
        rec["post_text"] = rec["prompt"].split("Post:", 1)[1].strip()
        return rec

    ds = ds.map(add_columns)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_fn = make_reward_fn(weights={"alpha": 1.0, "beta": 0.5, "gamma": 0.4})

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=32,
        num_generations=args.group_size,
        max_steps=args.steps,
        logging_steps=10,
        save_steps=100,
        beta=args.beta,
        use_vllm=False,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config,
    )

    logger.info("Launching GRPOTrainer for %d steps...", args.steps)
    trainer.train()

    logger.info("Saving trained policy adapter to %s...", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("GRPO post-training complete!")


if __name__ == "__main__":
    main()
