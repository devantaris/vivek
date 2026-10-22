"""Faithfulness reward: does the stated rationale actually support the decision?

Masking-based definition (comprehensiveness-style): remove the rationale's
content words from the original post and measure how much the model's own
P(hateful) changes. If the rationale names the words the model actually relies
on, removing them should move the prediction; if the rationale is boilerplate
or cites irrelevant words, the prediction barely moves and the reward is low.

Sign convention: we reward movement *away from the predicted label* when the
rationale words are removed — i.e. the score is
``|p_full − p_masked|``, clipped to [0, 1]. Without a probability callable or
the post text the neutral 0.5 is returned (documented placeholder used by
tests and dry runs).
"""

from __future__ import annotations

from collections.abc import Callable

from pj03.preprocess.cleaning import _WHITESPACE_RE  # noqa: PLC2701 - reuse splitter
from pj03.sft.prompt_templates import parse_output

PredictProbs = Callable[[str], float]

# Words that appear in almost every rationale template but never carry signal.
_STOPWORDS = frozenset(
    """a an the this that these those is are was were be been being it its
    post contains does not contain hateful language targeting and or of to
    in on at for with about against people person they them their he she
    i you we my your our us as by from have has had word words key""".split()
)

MAX_RATIONALE_TOKENS = 20


def _rationale_content_tokens(rationale: str) -> list[str]:
    """Lowercased, punctuation-stripped rationale tokens minus stopwords."""
    tokens = [tok.strip(".,!?;:\"'()[]").lower() for tok in rationale.split()]
    return [tok for tok in tokens if tok and tok not in _STOPWORDS and not tok.startswith("key")][
        :MAX_RATIONALE_TOKENS
    ]


def r_faith(
    generation: str,
    gold_label: int,
    predict_probs: PredictProbs | None = None,
    post_text: str | None = None,
) -> float:
    """Reward for rationale faithfulness, in [0, 1].

    Args:
        generation: Raw model output text (``Label:``/``Rationale:`` format).
        gold_label: Ground-truth label (unused for the symmetric score; kept
            for a uniform component signature).
        predict_probs: Callable mapping a text to the model's P(hateful) in
            [0, 1]. If None, returns the neutral 0.5.
        post_text: The original post the generation explains. If None,
            returns the neutral 0.5 (nothing to mask).

    Returns:
        ``clip(|p_full − p_masked|, 0, 1)`` where ``p_masked`` is the
        prediction on the post with the rationale's content words removed.
        0.5 when the score cannot be computed (no model or no post).
    """
    del gold_label  # symmetric definition; kept for component-signature parity
    if predict_probs is None or not post_text:
        return 0.5

    _, rationale = parse_output(generation)
    if not rationale:
        return 0.0  # a missing rationale is maximally unfaithful

    mask_tokens = set(_rationale_content_tokens(rationale))
    if not mask_tokens:
        return 0.0

    post_tokens = post_text.split()
    masked_tokens = [t for t in post_tokens if t.strip(".,!?;:\"'").lower() not in mask_tokens]
    if len(masked_tokens) == len(post_tokens):
        # Rationale named nothing present in the post: unfaithful citation.
        return 0.0
    masked_text = _WHITESPACE_RE.sub(" ", " ".join(masked_tokens)).strip()

    p_full = min(max(predict_probs(post_text), 0.0), 1.0)
    p_masked = min(max(predict_probs(masked_text), 0.0), 1.0) if masked_text else 0.0
    return min(abs(p_full - p_masked), 1.0)
