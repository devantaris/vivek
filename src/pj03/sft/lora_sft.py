"""LoRA supervised fine-tuning of a small instruction-tuned LLM (Phase 2).

TODO(Phase 2): implement with HF ``peft`` + ``trl.SFTTrainer``:

- Base model: a 1-3B instruction-tuned model (candidates: Qwen2.5-1.5B-Instruct,
  Llama-3.2-1B/3B-Instruct, Gemma-2-2B-it) - requires a human/licence decision.
- LoRA config: r=16, alpha=32, dropout=0.05, targets = q/k/v/o projections.
- Targets: the unified schema examples rendered via
  :func:`pj03.sft.prompt_templates.format_prompt`, completions in the fixed
  ``Label:/Rationale:`` format (gold rationale = dataset rationale text where
  available, else a template over ``rationale_mask`` tokens).
- CPU note: SFT will NOT run on CPU; requires a GPU session. Import must
  still be CPU-safe, hence lazy imports inside the function.
"""

from __future__ import annotations

from typing import Any


def run_lora_sft(
    model_name: str,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    seed: int = 42,
) -> Any:
    """Run LoRA SFT; returns the trained (PEFT-wrapped) model.

    Args:
        model_name: HF id of the base instruction-tuned LLM.
        train_dataset: HF ``Dataset`` of prompt/completion pairs.
        eval_dataset: Held-out prompt/completion pairs.
        output_dir: Output dir for the adapter (gitignored).
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling.
        epochs: Training epochs.
        seed: Random seed.

    Raises:
        NotImplementedError: Until Phase 2 is implemented.
    """
    raise NotImplementedError(
        "TODO(Phase 2): implement peft/trl.SFTTrainer LoRA SFT "
        f"for {model_name}. Requires GPU; scaffold only."
    )
