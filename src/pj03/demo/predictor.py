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


class DemoPredictor:
    """Loads (or trains) the demo pipeline and answers prediction requests."""

    def __init__(self, artifact_path: str | Path = DEFAULT_ARTIFACT) -> None:
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found - train it first: "
                "python scripts/train_demo_model.py"
            )
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.metadata: dict = bundle.get("metadata", {})
        test_metrics = self.metadata.get("test_metrics", {})
        logger.info(
            "Loaded demo model %s (macro_f1=%s)", path, test_metrics.get("macro_f1")
        )

    # ------------------------------------------------------------------ core
    def predict(self, text: str, top_k: int = 5, compute_faithfulness: bool = True) -> PredictionResult:
        """Classify ``text`` and explain the decision.

        Args:
            text: Raw social-media-style post.
            top_k: How many top-contributing n-grams to return.
            compute_faithfulness: Also run live comprehensiveness/sufficiency.

        Returns:
            :class:`PredictionResult` with label, probability, rationale and
            (optionally) faithfulness scores.
        """
        prob = float(self.pipeline.predict_proba([text])[0, 1])
        label = int(prob >= 0.5)
        contributions = self._token_contributions(text)
        rationale = sorted(contributions, key=lambda x: -abs(x[1]))[:top_k]

        result = PredictionResult(
            text=text, label=label, prob_hateful=prob, rationale=rationale
        )
        if compute_faithfulness and contributions:
            tokens = text.split()
            attrib = self._align_to_tokens(tokens, contributions)
            probs_fn = self._prob_fn()
            try:
                k = min(3, len(tokens))
                result.comprehensiveness = comprehensiveness(probs_fn, tokens, attrib, k=k)
                result.sufficiency = sufficiency(probs_fn, tokens, attrib, k=k)
            except ValueError:  # degenerate input (single token etc.)
                pass
        return result

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
            return bool(ngram) and all(
                ch.isalnum() or ch in " '’-" for ch in ngram
            )

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
