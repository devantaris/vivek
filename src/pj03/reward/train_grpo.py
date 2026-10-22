"""GRPO post-training with the composite reward (Phase 3).

Implemented with ``trl.GRPOTrainer``. Runs ONLY on a Linux GPU box
(Colab/Kaggle/lab machine): the composite reward needs the SFT model in the
loop, and TRL's generation kernels are Linux-only. Everything below imports
lazily so the module stays importable on CPU for tests/config checks.

Data flow:
    1. Prompts come from the SFT dataset (``prompt`` column) plus the extra
       columns (``post_text``, ``gold_label``) that TRL forwards to the
       reward functions as keyword lists.
    2. Per generated completion we compute
       R = alpha*R_label + beta*R_faith - gamma*R_bias with weights from
       ``configs/reward.yaml`` (overridable on the command line).
    3. GRPO normalises rewards within each prompt's generation group.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from collections.abc import Callable

from pj03.utils.logging import get_logger  # noqa: E402
from pj03.utils.seed import set_seed  # noqa: E402

logger = get_logger("pj03.grpo")


def make_reward_fn(
    weights: dict[str, float] | None = None,
    predictor_builder: Callable[[], Callable[[str], float]] | None = None,
) -> Callable[..., list[float]]:
    """Build a TRL-compatible composite reward function.

    TRL calls reward functions as ``fn(prompts, completions, **columns)`` and
    expects one float per completion. The SFT dataset's extra columns
    (``gold_label``, ``post_text``) arrive as lists aligned with completions.

    Args:
        weights: Composite weights (alpha/beta/gamma); None -> defaults.
        predictor_builder: Optional factory returning a P(hateful) callable
            (e.g. the SFT model's classifier head). Built lazily ONCE so the
            model loads a single time for the whole training run; None means
            R_faith falls back to its neutral 0.5.

    Returns:
        A function with TRL's expected signature.
    """
    from pj03.reward.composite import composite_reward

    predictor: Callable[[str], float] | None = (
        predictor_builder() if predictor_builder is not None else None
    )

    def reward_fn(
        prompts: Any, completions: list[str], *, gold_label=None, post_text=None, **_: Any
    ) -> list[float]:
        del prompts  # prompts are implicit in the completions' context
        n = len(completions)
        golds = gold_label if gold_label is not None else [0] * n
        posts = post_text if post_text is not None else [None] * n
        return [
            composite_reward(
                completion,
                int(gold),
                weights=weights,
                predict_probs=predictor,
                post_text=post,
            )
            for completion, gold, post in zip(completions, golds, posts, strict=False)
        ]

    return reward_fn


def _resolve_model_and_adapter(model_path: str) -> tuple[str, str | None]:
    """Given a model path or HuggingFace ID, resolve base model and adapter dir."""
    import json
    path = Path(model_path)
    adapter_cfg = path / "adapter_config.json"
    if path.is_dir() and adapter_cfg.is_file():
        try:
            cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            base_name = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
            return base_name, str(path)
        except Exception as e:
            logger.warning("Could not parse adapter_config.json (%s), assuming %s", e, model_path)
    return model_path, None


def build_predictor(model_path: str, device: str = "cuda") -> Callable[[str], float]:
    """Build a P(hateful) callable from the SFT model (for R_faith).

    Loads the model ONCE and scores a post by the P(token '1' | prompt).
    Cheap generation-free proxy for the classifier head, suitable as a
    faithfulness signal inside the reward.

    Args:
        model_path: HF id or checkpoint dir of the SFT model.
        device: Torch device string.

    Returns:
        Callable mapping post text -> P(hateful) in [0, 1].
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pj03.sft.prompt_templates import format_prompt

    base_name, adapter_path = _resolve_model_and_adapter(model_path)
    tok_source = adapter_path or base_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device).eval()

    hateful_id = tokenizer("1", add_special_tokens=False)["input_ids"][0]

    @torch.no_grad()
    def predict(post: str) -> float:
        inputs = tokenizer(
            format_prompt(post), return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        logits = model(**inputs).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, hateful_id].item())

    return predict


def run_grpo(
    model_path: str,
    train_dataset: Any,
    output_dir: str,
    reward_config: dict[str, Any] | None = None,
    num_generations: int = 8,
    learning_rate: float = 2e-6,
    kl_coef: float = 0.04,
    batch_size: int = 4,
    epochs: int = 1,
    max_completion_length: int = 128,
    max_prompt_length: int = 384,
    seed: int = 42,
    use_faith_predictor: bool = True,
    save_model: bool = True,
) -> dict[str, Any]:
    """Run GRPO post-training with the composite reward.

    Args:
        model_path: HF id or checkpoint of the SFT model (base + adapter dir
            from Phase 2; the adapter is loaded and merged).
        train_dataset: HF ``Dataset`` with columns ``prompt``, ``post_text``,
            ``gold_label`` (from ``scripts/build_sft.py`` output + schema).
        output_dir: Where the trained model/adapter is saved.
        reward_config: Loaded PJ-03 config; weights at ``reward.weights``.
        num_generations: Completions sampled per prompt (GRPO group size).
        learning_rate: AdamW learning rate.
        kl_coef: KL penalty coefficient against the reference policy
            (GRPOConfig ``beta``).
        batch_size: Prompts per optimisation step (per device).
        epochs: Passes over the prompt set.
        max_completion_length: Generation cap (rationales are short).
        max_prompt_length: Prompt truncation cap.
        seed: Random seed.
        use_faith_predictor: Whether to load the model for the R_faith
            predictor (adds one forward pass per post; disable for speed).
        save_model: Save the final model to ``output_dir``.

    Returns:
        Final trainer state metrics dict.

    Raises:
        RuntimeError: If no CUDA device is available (GRPO needs a GPU).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GRPO requires a CUDA GPU (Linux box: Colab/Kaggle/lab machine). "
            "The laptop is for data prep, smoke tests and code validation."
        )

    set_seed(seed)
    weights = (reward_config or {}).get("reward", {}).get("weights", None)
    logger.info("GRPO reward weights: %s", weights)

    # Load the base model and merge the LoRA adapter if one is present.
    base_name, adapter_path = _resolve_model_and_adapter(model_path)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info("Loading base model %s for GRPO with dtype=%s", base_name, dtype)
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    if adapter_path:
        logger.info("Loading and merging LoRA adapter from %s into %s", adapter_path, base_name)
        base = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    predictor_builder = (lambda: build_predictor(model_path)) if use_faith_predictor else None
    reward_fn = make_reward_fn(weights, predictor_builder)

    required = {"prompt", "post_text", "gold_label"}
    missing = required - set(train_dataset.column_names)
    if missing:
        raise ValueError(
            f"train_dataset missing columns {sorted(missing)}; build it from "
            "the SFT JSONL joined with the unified schema (see gpu_runbook)."
        )

    config = GRPOConfig(
        output_dir=output_dir,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        learning_rate=learning_rate,
        beta=kl_coef,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        report_to="none",
        seed=seed,
        use_cpu=False,
    )
    trainer = GRPOTrainer(
        model=base,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=train_dataset,
    )
    trainer.train()
    metrics: dict[str, Any] = trainer.state.log_history[-1] if trainer.state.log_history else {}
    if save_model:
        trainer.save_model(output_dir)
        AutoTokenizer.from_pretrained(model_path).save_pretrained(output_dir)
        logger.info("Saved GRPO model -> %s", output_dir)
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
