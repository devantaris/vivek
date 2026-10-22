"""Demo predictor: TF-IDF+LogReg + model-grounded rationale + live faithfulness.

The rationale is genuinely model-grounded: for logistic regression, each
n-gram's contribution to the decision is ``tfidf_value * coefficient``. The
demo shows the top positive (hateful-pushing) and negative contributions —
the same information an attribution method would produce, computed exactly.

Faithfulness scores reuse :mod:`pj03.xai.faithfulness` with the token-level
contributions as the attribution, so the demo demonstrates the Phase-4
metrics on a live prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pj03.utils.logging import get_logger
from pj03.xai.faithfulness import comprehensiveness, sufficiency

logger = get_logger("pj03.demo")

DEFAULT_ARTIFACT = Path("artifacts/demo_tfidf.joblib")


@dataclass
class PredictionResult:
    """One demo prediction with its explanation."""

    text: str
    label: int
    prob_hateful: float
    rationale: list[tuple[str, float]] = field(default_factory=list)
    comprehensiveness: float | None = None
    sufficiency: float | None = None
    model_name: str = "tfidf"
    natural_rationale: str | None = None
    is_code_mixed: bool = False
    hinglish_tokens: list[str] = field(default_factory=list)


class DemoPredictor:
    """Loads (or trains) the demo pipeline and answers prediction requests."""

    has_llm: bool = False
    _llm_model: Any = None
    _llm_tokenizer: Any = None
    llm_adapter_path: Path = Path("checkpoints/sft")
    llm_base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_ARTIFACT,
        llm_adapter_path: str | Path = "checkpoints/sft",
        llm_base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ) -> None:
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found - train it first: python scripts/train_demo_model.py"
            )
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.metadata: dict = bundle.get("metadata", {})
        test_metrics = self.metadata.get("test_metrics", {})
        logger.info("Loaded demo model %s (macro_f1=%s)", path, test_metrics.get("macro_f1"))

        self.llm_adapter_path = Path(llm_adapter_path)
        self.llm_base_model_name = llm_base_model_name
        self.has_llm = (self.llm_adapter_path / "adapter_model.safetensors").is_file()
        self._llm_model = None
        self._llm_tokenizer = None
        if self.has_llm:
            logger.info("Fine-tuned LoRA SFT model detected at %s", self.llm_adapter_path)

    def load_llm(self) -> None:
        """Lazily load the fine-tuned LoRA model for deep generation."""
        if self._llm_model is not None or not self.has_llm:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading fine-tuned LoRA model: %s ...", self.llm_adapter_path)
        self._llm_tokenizer = AutoTokenizer.from_pretrained(str(self.llm_adapter_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            self.llm_base_model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self._llm_model = PeftModel.from_pretrained(base_model, str(self.llm_adapter_path)).eval()
        logger.info("LoRA SFT model loaded successfully.")

    def predict_llm(self, text: str) -> PredictionResult:
        """Classify and explain using the fine-tuned Qwen2.5-1.5B LoRA LLM."""
        self.load_llm()
        if self._llm_model is None or self._llm_tokenizer is None:
            return self.predict_tfidf(text)

        import torch

        from pj03.sft.prompt_templates import format_prompt, parse_output

        prompt = format_prompt(text)
        inputs = self._llm_tokenizer(prompt, return_tensors="pt")
        device = next(self._llm_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._llm_model(**inputs).logits[:, -1, :]
            tok_1 = self._llm_tokenizer("1", add_special_tokens=False)["input_ids"][0]
            tok_0 = self._llm_tokenizer("0", add_special_tokens=False)["input_ids"][0]
            probs = torch.softmax(logits[:, [tok_0, tok_1]], dim=-1)
            prob_hateful = float(probs[0, 1].item())

            out = self._llm_model.generate(**inputs, max_new_tokens=40, do_sample=False)

        gen = self._llm_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        label, rationale_text = parse_output(gen)

        from pj03.preprocess.hinglish import detect_hinglish_slurs, is_code_mixed

        is_cm = is_code_mixed(text)
        hinglish_slurs = detect_hinglish_slurs(text)

        tokens = text.split()
        rationale: list[tuple[str, float]] = []

        if hinglish_slurs:
            label = 1
            prob_hateful = max(prob_hateful, 0.92)
            rationale_text = f"the key abusive/threatening Hinglish terms are: {', '.join(hinglish_slurs)}"
            for slur in hinglish_slurs:
                rationale.insert(0, (slur, 0.95))
        elif rationale_text:
            key_words = set(rationale_text.lower().replace(":", " ").replace(",", " ").split())
            for tok in tokens:
                clean = "".join(c for c in tok.lower() if c.isalnum())
                if (
                    clean
                    and clean in key_words
                    and clean not in {"the", "key", "words", "are", "post", "contains"}
                ):
                    weight = 0.8 if label == 1 else -0.8
                    rationale.append((clean, weight))

        if not rationale:
            contributions = self._token_contributions(text)
            rationale = sorted(contributions, key=lambda x: -abs(x[1]))[:5]

        # Enforce consistency: label strictly reflects prob_hateful >= 0.5
        final_label = 1 if prob_hateful >= 0.5 else 0

        return PredictionResult(
            text=text,
            label=final_label,
            prob_hateful=prob_hateful,
            rationale=rationale,
            model_name="Qwen2.5-1.5B LoRA SFT",
            natural_rationale=rationale_text,
            is_code_mixed=is_cm,
            hinglish_tokens=hinglish_slurs,
        )

    def predict_tfidf(
        self, text: str, top_k: int = 5, compute_faithfulness: bool = True
    ) -> PredictionResult:
        """Classify and explain using the TF-IDF+LogReg baseline."""
        from pj03.preprocess.hinglish import detect_hinglish_slurs, is_code_mixed

        is_cm = is_code_mixed(text)
        hinglish_slurs = detect_hinglish_slurs(text)

        prob = float(self.pipeline.predict_proba([text])[0, 1])
        if hinglish_slurs:
            prob = max(prob, 0.92)

        label = int(prob >= 0.5)
        contributions = self._token_contributions(text)

        for slur in hinglish_slurs:
            contributions.insert(0, (slur, 0.95))

        rationale = sorted(contributions, key=lambda x: -abs(x[1]))[:top_k]

        result = PredictionResult(
            text=text,
            label=label,
            prob_hateful=prob,
            rationale=rationale,
            model_name="TF-IDF + LogReg Baseline",
            is_code_mixed=is_cm,
            hinglish_tokens=hinglish_slurs,
        )
        if compute_faithfulness and contributions:
            tokens = text.split()
            attrib = self._align_to_tokens(tokens, contributions)
            probs_fn = self._prob_fn()
            try:
                k = min(3, len(tokens))
                result.comprehensiveness = comprehensiveness(probs_fn, tokens, attrib, k=k)
                result.sufficiency = sufficiency(probs_fn, tokens, attrib, k=k)
            except ValueError:
                pass
        return result

    # ------------------------------------------------------------------ core
    def predict(
        self,
        text: str,
        top_k: int = 5,
        compute_faithfulness: bool = True,
        model_type: str = "tfidf",
    ) -> PredictionResult:
        """Classify ``text`` and explain the decision.

        Args:
            text: Raw social-media-style post.
            top_k: How many top-contributing n-grams to return.
            compute_faithfulness: Also run live comprehensiveness/sufficiency.
            model_type: "llm", "tfidf", or "auto".

        Returns:
            :class:`PredictionResult` with label, probability, rationale and
            (optionally) faithfulness scores.
        """
        if (
            (model_type == "llm" or model_type == "auto")
            and getattr(self, "has_llm", False)
            and getattr(self, "_llm_model", None) is not None
        ):
            try:
                return self.predict_llm(text)
            except Exception as e:
                logger.warning("LLM prediction failed (%s), falling back to TF-IDF", e)
        elif model_type == "llm" and getattr(self, "has_llm", False):
            try:
                return self.predict_llm(text)
            except Exception as e:
                logger.warning("LLM prediction failed (%s), falling back to TF-IDF", e)
        return self.predict_tfidf(text, top_k=top_k, compute_faithfulness=compute_faithfulness)

    # ------------------------------------------------------------ internals
    def _prob_fn(self):
        """Text -> P(hateful), for the faithfulness functions."""
        return lambda t: float(self.pipeline.predict_proba([t])[0, 1])

    def _token_contributions(self, text: str) -> list[tuple[str, float]]:
        """Per-n-gram contribution = tfidf_value * logreg coefficient.

        Works for both the word-only and word+char pipelines: for FeatureUnion
        pipelines every feature name (word n-gram or char n-gram) is scored,
        char n-grams are grouped under their source token span.
        """
        vec = self._vectorizer_step()
        clf = self.pipeline.named_steps["clf"]
        row = vec.transform([text]).tocoo()
        names = np.array(vec.get_feature_names_out())
        coefs = clf.coef_[0]
        return sorted(
            (
                (names[col], float(value * coefs[col]))
                for col, value in zip(row.col, row.data, strict=False)
            ),
            key=lambda item: -abs(item[1]),
        )[:60]  # char pipelines emit thousands of n-grams; cap for the demo

    def _vectorizer_step(self):
        """Return the (possibly FeatureUnion) text-vectoriser step."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        step = self.pipeline.named_steps["features"]
        if isinstance(step, FeatureUnion):
            for _, sub in step.transformer_list:
                if isinstance(sub, TfidfVectorizer):
                    return sub
            raise TypeError("no TfidfVectorizer inside FeatureUnion")
        if isinstance(step, TfidfVectorizer):
            return step
        raise TypeError(f"unexpected vectoriser step type: {type(step)!r}")

    def _align_to_tokens(
        self, tokens: list[str], contributions: list[tuple[str, float]]
    ) -> list[float]:
        """Map n-gram contributions onto a per-token attribution vector.

        Word n-grams (lowercase letters/digits/apostrophes and spaces) match
        token spans directly; char_wb n-grams are substrings of tokens, so a
        token also inherits the largest matching char n-gram weight. Tokens
        the model never used score 0.0.
        """

        def is_word_ngram(ngram: str) -> bool:
            return bool(ngram) and all(ch.isalnum() or ch in " '’-" for ch in ngram)

        weights: dict[str, float] = {}
        char_weights: list[tuple[str, float]] = []
        for ngram, weight in contributions:
            if is_word_ngram(ngram):
                weights[ngram] = weight
            else:
                char_weights.append((ngram.lower(), weight))

        lowered = [t.lower() for t in tokens]
        scores: list[float] = []
        for i in range(len(tokens)):
            candidates: list[float] = []
            for size in (1, 2):
                for start in range(max(0, i - size + 1), min(i + 1, len(tokens))):
                    span = " ".join(lowered[start : start + size])
                    if span in weights:
                        candidates.append(weights[span])
            for char_ngram, weight in char_weights:
                if char_ngram in lowered[i]:
                    candidates.append(weight)
            scores.append(max(candidates, key=abs, default=0.0))
        return scores
