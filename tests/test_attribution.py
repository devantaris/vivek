"""Unit tests for Captum Integrated Gradients and token SHAP attributions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pj03.xai.attribution import integrated_gradients_attribution, token_shap_attribution


class DummyEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 100, embed_dim: int = 16) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class DummyTransformerModel(nn.Module):
    def __init__(self, vocab_size: int = 100, embed_dim: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        self.embeddings = DummyEmbedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Any:
        emb = self.embeddings(input_ids)
        pooled = emb.mean(dim=1)
        logits = self.classifier(pooled)

        class Output:
            def __init__(self, logits: torch.Tensor) -> None:
                self.logits = logits

        return Output(logits)


class DummyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 128,
    ) -> dict[str, torch.Tensor]:
        tokens = text.split()
        ids = [min(abs(hash(t)) % 90 + 1, 99) for t in tokens]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"tok_{i}" for i in ids]


def test_token_shap_attribution() -> None:
    # A simple predictor where 'hate' gives 0.9 and other words give 0.1
    def dummy_predict(texts: list[str]) -> list[float]:
        return [0.9 if "hate" in t else 0.1 for t in texts]

    text = "i really hate you"
    attributions = token_shap_attribution(dummy_predict, text)
    assert len(attributions) == 4
    # 'hate' token should have the largest positive marginal contribution
    attr_dict = dict(attributions)
    assert attr_dict["hate"] > 0.5
    assert attr_dict["hate"] > attr_dict["really"]


def test_integrated_gradients_dummy_model() -> None:
    model = DummyTransformerModel()
    tokenizer = DummyTokenizer()
    text = "kill all enemies"
    attrs = integrated_gradients_attribution(model, tokenizer, text, n_steps=5, target_class=1)
    assert len(attrs) == 3
    assert all(isinstance(val, float) for _, val in attrs)
