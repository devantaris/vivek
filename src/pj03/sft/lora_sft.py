"""LoRA supervised fine-tuning of a small instruction-tuned LLM (Phase 2).

Implemented with HF ``peft`` + a plain ``Trainer`` loop with completion-only
label masking (the prompt tokens are excluded from the loss):

- Base model: a 1-3B instruction-tuned model (candidates: Qwen2.5-1.5B-Instruct,
  Llama-3.2-1B/3B-Instruct, Gemma-2-2B-it) - requires a human/licence decision.
- Training data: the prompt/completion JSONL built by ``scripts/build_sft.py``
  (``data/processed/sft_hatexplain/``); completions use the fixed
  ``Label:/Rationale:`` format the Phase-3 rewards parse.

Why not ``trl.SFTTrainer``: as of trl 1.15 it patches a fused LM-head forward
whose Triton kernel ships with PyTorch on Linux only, so it crashes on
Windows/CPython before the first step. The manual loop below is ~40 lines,
platform-independent, and gives exact control over the completion-only loss.
TRL is still used for GRPO in Phase 3, which runs on Linux GPUs.
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset

from pj03.utils.logging import get_logger  # noqa: E402
from pj03.utils.seed import set_seed  # noqa: E402

logger = get_logger("pj03.sft")

# Attention-projection defaults (GPT-2-family Conv1D modules use "c_attn").
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class PromptCompletionDataset(TorchDataset):
    """Tokenised prompt+completion pairs with completion-only labels."""

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.records = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        rec = self.records[idx]
        prompt_ids = self.tokenizer(rec["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(
            rec["completion"] + self.tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        input_ids = (prompt_ids + completion_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self.max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def _collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch to the longest sequence (labels padded with -100)."""
    max_len = max(len(b["input_ids"]) for b in batch)
    out: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = max_len - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def run_lora_sft(
    model_name: str,
    train_dataset: Any,
    eval_dataset: Any | None,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    learning_rate: float = 2e-4,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    epochs: int = 3,
    max_length: int = 512,
    seed: int = 42,
    save_adapter: bool = True,
) -> dict[str, float]:
    """LoRA-fine-tune ``model_name`` on prompt/completion records.

    Args:
        model_name: HF id of the base instruction-tuned LLM.
        train_dataset: HF ``Dataset`` with ``prompt`` and ``completion``
            string columns (extra columns are ignored).
        eval_dataset: Optional eval ``Dataset`` in the same format.
        output_dir: Where the LoRA adapter is saved (gitignored).
        lora_r / lora_alpha / lora_dropout: LoRA hyperparameters.
        target_modules: Modules to adapt; defaults to attention projections
            (pass ``["c_attn"]`` for GPT-2-family models).
        learning_rate: AdamW learning rate.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        epochs: Training epochs.
        max_length: Max tokenised length of prompt+completion.
        seed: Random seed.
        save_adapter: If True, save the trained adapter to ``output_dir``.

    Returns:
        Eval metrics dict (completion loss), empty if no eval dataset.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_wrapped = PromptCompletionDataset(train_dataset, tokenizer, max_length)
    eval_wrapped = (
        PromptCompletionDataset(eval_dataset, tokenizer, max_length)
        if eval_dataset is not None
        else None
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        eval_strategy="epoch" if eval_wrapped is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
    )

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return _collate(batch, tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_wrapped,
        eval_dataset=eval_wrapped,
        data_collator=collate,
    )
    trainer.train()
    metrics: dict[str, float] = {}
    if eval_wrapped is not None:
        metrics = {k: float(v) for k, v in trainer.evaluate().items()}
        logger.info("SFT eval metrics: %s", metrics)
    if save_adapter:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved LoRA adapter + tokenizer -> %s", output_dir)
    return metrics
