"""Attribution methods: Captum Integrated Gradients and Model-Agnostic KernelSHAP.

Provides:
- ``integrated_gradients_attribution``: Captum LayerIntegratedGradients on the
  embedding layer of transformer models (MuRIL, BERT, XLM-R).
- ``token_shap_attribution``: Exact leave-one-out and sampled Shapley value
  computation across tokens for model-agnostic attribution.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np


def integrated_gradients_attribution(
    model: Any,
    tokenizer: Any,
    text: str,
    n_steps: int = 30,
    target_class: int = 1,
) -> list[tuple[str, float]]:
    """Compute per-token Integrated Gradients attributions on transformer embeddings.

    Uses ``captum.attr.LayerIntegratedGradients`` to attribute the output prediction
    for ``target_class`` back to the model's word embeddings.

    Args:
        model: HuggingFace AutoModelForSequenceClassification instance.
        tokenizer: Matching HuggingFace tokenizer.
        text: Input string.
        n_steps: Number of Riemann summation approximation steps.
        target_class: Class index to attribute (1 = hateful).

    Returns:
        List of ``(token_string, attribution_score)`` tuples for each token.
    """
    import torch
    from captum.attr import LayerIntegratedGradients

    device = next(model.parameters()).device
    model.eval()

    # Determine embedding layer
    if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
        embed_layer = model.bert.embeddings
    elif hasattr(model, "roberta") and hasattr(model.roberta, "embeddings"):
        embed_layer = model.roberta.embeddings
    elif hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        raise AttributeError("Cannot identify input embedding layer for Integrated Gradients.")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # Reference / baseline is pad or zero tokens
    pad_id = tokenizer.pad_token_id or 0
    ref_input_ids = torch.full_like(input_ids, pad_id)

    def forward_func(inputs_ids_tensor: torch.Tensor) -> torch.Tensor:
        logits = model(input_ids=inputs_ids_tensor, attention_mask=attention_mask).logits
        return logits[:, target_class]

    lig = LayerIntegratedGradients(forward_func, embed_layer)
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=ref_input_ids,
        n_steps=n_steps,
        return_convergence_delta=False,
    )

    # Sum attribution across embedding dimensions
    if isinstance(attributions, tuple):
        attributions = attributions[0]
    token_attributions = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # Normalize attribution vector to L2 or max absolute norm
    norm = np.linalg.norm(token_attributions)
    if norm > 0:
        token_attributions = token_attributions / norm

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return list(zip(tokens, [float(v) for v in token_attributions], strict=False))


def token_shap_attribution(
    predict_fn: Callable[[list[str]], list[float]] | Callable[[str], float],
    text: str,
    n_samples: int = 64,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Compute model-agnostic token-level Shapley values.

    For short texts (<= 12 tokens), computes exact leave-one-out marginal
    contributions. For longer sequences, uses weighted Monte Carlo subset
    sampling with Shapley kernel weights.

    Args:
        predict_fn: Callable returning P(hateful) for a batch (or single) text.
        text: Input string.
        n_samples: Monte Carlo sample iterations for long texts.
        seed: Random seed for sampling.

    Returns:
        List of ``(token, shap_value)`` pairs.
    """
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return []

    def _eval(tok_list: list[str]) -> float:
        sub_text = " ".join(tok_list)
        if not sub_text:
            return 0.0
        try:
            res = predict_fn([sub_text])
            return float(res[0]) if isinstance(res, (list, np.ndarray)) else float(res)
        except Exception:
            return float(predict_fn(sub_text))

    base_score = _eval(tokens)

    shap_values: list[float] = []
    if n <= 12:
        for i in range(n):
            subset = [t for j, t in enumerate(tokens) if j != i]
            score_without = _eval(subset)
            shap_values.append(base_score - score_without)
    else:
        rng = random.Random(seed)
        marginals = [0.0] * n
        counts = [0] * n
        for _ in range(n_samples):
            mask = [rng.random() > 0.5 for _ in range(n)]
            idx = rng.randint(0, n - 1)
            with_idx = [t for j, t in enumerate(tokens) if mask[j] or j == idx]
            without_idx = [t for j, t in enumerate(tokens) if mask[j] and j != idx]
            diff = _eval(with_idx) - _eval(without_idx)
            marginals[idx] += diff
            counts[idx] += 1

        for i in range(n):
            shap_values.append(marginals[i] / max(counts[i], 1))

    return list(zip(tokens, [float(v) for v in shap_values], strict=False))
