"""Attribution methods: SHAP and Integrated Gradients (Phase 4).

TODO(Phase 4): implement two backends behind a common interface:

- Integrated Gradients via ``captum.attr.IntegratedGradients`` on the
  classifier's embedding layer (works for mBERT/XLM-R/MuRIL checkpoints).
- SHAP via ``shap.Explainer`` / ``shap.KernelExplainer`` on the model's
  predict function for token-level attributions (model-agnostic; needed for
  the LoRA-LLM where Captum hooks are less convenient).

Output contract: a list of per-token floats (one per token of the input,
including special tokens), normalised so the caller can threshold them into
a rationale mask.
"""

from __future__ import annotations

from typing import Any


def integrated_gradients_attribution(
    model: Any, tokenizer: Any, text: str, n_steps: int = 50, seed: int = 42
) -> list[float]:
    """Per-token Integrated Gradients attribution for P(hateful).

    Args:
        model: HF classification model.
        tokenizer: Matching HF tokenizer.
        text: Input text to attribute.
        n_steps: Number of IG interpolation steps.
        seed: Baseline/zero-token random seed (kept for determinism).

    Raises:
        NotImplementedError: Until Phase 4.
    """
    raise NotImplementedError(
        "TODO(Phase 4): implement Captum IntegratedGradients on embeddings."
    )


def shap_attribution(model_predict: Any, texts: list[str]) -> list[list[float]]:
    """Per-token SHAP values for a batch of texts.

    Args:
        model_predict: Callable mapping list[str] -> list[P(hateful)].
        texts: Input texts.

    Raises:
        NotImplementedError: Until Phase 4.
    """
    raise NotImplementedError("TODO(Phase 4): implement SHAP token attributions.")
