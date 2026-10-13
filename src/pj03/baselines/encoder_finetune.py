"""Encoder fine-tuning baselines: mBERT, XLM-R, MuRIL (Phase 1).

TODO(Phase 1): implement the Hugging Face ``Trainer`` loop:

- mBERT:           ``bert-base-multilingual-cased``
- XLM-RoBERTa:     ``xlm-roberta-base``
- MuRIL:           ``google/muril-base-cased`` (best Hindi/Hinglish coverage)

Plan: load model via ``AutoModelForSequenceClassification`` (num_labels=2),
tokenise with ``AutoTokenizer``, train with HF ``Trainer`` +
``TrainingArguments`` (fp16 off on CPU), evaluate with macro-F1 computed by
:mod:`pj03.eval.metrics`. Log per-language (en / hi / hinglish) macro-F1 too -
that is the headline table of the paper.
"""

from __future__ import annotations

from typing import Any


def train_encoder(
    model_name: str,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: str,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    seed: int = 42,
) -> Any:
    """Fine-tune a pretrained encoder for binary hate-speech classification.

    Args:
        model_name: HF model id (e.g. ``xlm-roberta-base``).
        train_dataset: Tokenised HF ``Dataset`` with a ``labels`` column.
        eval_dataset: Tokenised eval ``Dataset``.
        output_dir: Where checkpoints/metrics go (under ``data/`` or
            ``checkpoints/``, gitignored).
        learning_rate: AdamW learning rate.
        batch_size: Per-device batch size.
        epochs: Training epochs.
        seed: Random seed.

    Raises:
        NotImplementedError: Until Phase 1 training is implemented.
    """
    raise NotImplementedError(
        "TODO(Phase 1): implement HF Trainer loop for encoders "
        f"({model_name}). No training happens in the scaffold."
    )
