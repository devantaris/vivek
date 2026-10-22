"""Encoder fine-tuning baselines: mBERT, XLM-R, MuRIL (Phase 1).

Models (config ``baselines.encoders.models``):
    - mBERT:       ``bert-base-multilingual-cased``
    - XLM-RoBERTa: ``xlm-roberta-base``
    - MuRIL:       ``google/muril-base-cased`` (best Hindi/Hinglish coverage)

Full runs need a GPU. The same code path is validated on CPU via the smoke
config (``baselines.encoders.smoke``): a tiny model (prajjwal1/bert-tiny,
~17MB) on a few hundred examples, finishing in seconds.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

from pj03.utils.logging import get_logger  # noqa: E402
from pj03.utils.seed import set_seed  # noqa: E402

logger = get_logger("pj03.encoder")


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_encoder(
    model_name: str,
    train_texts: list[str],
    train_labels: list[int],
    eval_texts: list[str],
    eval_labels: list[int],
    output_dir: str,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    max_length: int = 256,
    seed: int = 42,
    save_model: bool = False,
) -> dict[str, float]:
    """Fine-tune a pretrained encoder for binary hate-speech classification.

    Args:
        model_name: HF model id (e.g. ``xlm-roberta-base``).
        train_texts / train_labels: Training posts and binary labels.
        eval_texts / eval_labels: Held-out posts and labels.
        output_dir: Where Trainer logs/metrics go (checkpoints only if
            ``save_model``).
        learning_rate: AdamW learning rate.
        batch_size: Per-device batch size.
        epochs: Training epochs.
        max_length: Tokeniser max length (HateXplain max is ~165 tokens).
        seed: Random seed.
        save_model: If True, save the final model + tokenizer (disk-heavy for
            full-size encoders; off by default).

    Returns:
        Eval metrics dict (loss, macro-F1, AUC).
    """
    from datasets import Dataset
    from sklearn.metrics import roc_auc_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from pj03.eval.metrics import macro_f1

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch: dict[str, list]) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels}).map(
        tokenize, batched=True
    )
    eval_ds = Dataset.from_dict({"text": eval_texts, "labels": eval_labels}).map(
        tokenize, batched=True
    )

    def compute_metrics(pred: Any) -> dict[str, float]:
        probs = _softmax(pred.predictions)
        y_pred = probs.argmax(axis=1).tolist()
        y_true = pred.label_ids.tolist()
        metrics = {"macro_f1": macro_f1(y_true, y_pred)}
        if len(set(y_true)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        return metrics

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="no",  # no checkpoint bloat; save_model controls final save
        logging_steps=20,
        report_to="none",  # no wandb unless explicitly configured
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = {k: float(v) for k, v in trainer.evaluate().items()}
    if save_model:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved model + tokenizer -> %s", output_dir)
    return metrics
