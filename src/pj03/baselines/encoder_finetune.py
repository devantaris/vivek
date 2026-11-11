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
    learning_rate: float = 1.5e-5,
    batch_size: int = 16,
    epochs: int = 4,
    max_length: int = 256,
    seed: int = 42,
    save_model: bool = False,
    weighted_loss: bool = True,
    warmup_ratio: float = 0.1,
) -> dict[str, float]:
    """Fine-tune a pretrained encoder for binary hate-speech classification.

    Args:
        model_name: HF model id (e.g. ``google/muril-base-cased``, ``xlm-roberta-base``).
        train_texts / train_labels: Training posts and binary labels.
        eval_texts / eval_labels: Held-out posts and labels.
        output_dir: Where Trainer logs/metrics go.
        learning_rate: AdamW learning rate (default 1.5e-5).
        batch_size: Per-device batch size.
        epochs: Training epochs (default 4).
        max_length: Tokeniser max length.
        seed: Random seed.
        save_model: If True, save final model + tokenizer.
        weighted_loss: Balance class weights in CrossEntropyLoss.
        warmup_ratio: Linear learning-rate warmup fraction.

    Returns:
        Eval metrics dict (loss, macro-F1, AUC).
    """
    import torch
    import torch.nn as nn
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

    # Compute inverse class frequencies
    weights_tensor = None
    if weighted_loss and len(train_labels) > 0:
        n_pos = max(sum(train_labels), 1)
        n_neg = max(len(train_labels) - n_pos, 1)
        total = len(train_labels)
        w0 = total / (2.0 * n_neg)
        w1 = total / (2.0 * n_pos)
        weights_tensor = torch.tensor([w0, w1], dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if weights_tensor is not None and labels is not None:
                loss_fct = nn.CrossEntropyLoss(weight=weights_tensor.to(logits.device))
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            else:
                loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

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
        save_strategy="no",
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=20,
        report_to="none",
        seed=seed,
    )
    trainer = WeightedTrainer(
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
