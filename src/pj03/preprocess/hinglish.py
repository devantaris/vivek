"""Hinglish (Hindi-English code-mixed) normalisation.

TODO(Phase 1): Hinglish tweets are written in many transliteration schemes
("kyu/kyun/kyon", "nahi/nahin"). Plan:

1. Build a small transliteration-variant dictionary (Devanagari-in-Roman ->
   canonical Roman spelling) for high-frequency function words.
2. Optionally use langdetect / a code-mix language ID model (e.g. LinCE
   pretrained tags) to tag per-token language, which downstream bias analysis
   can use.
3. Keep the original text alongside the normalised one - never overwrite.
"""

from __future__ import annotations

# Starter variant map (extend after inspecting the actual data).
_VARIANT_MAP: dict[str, str] = {
    "kyun": "kyu",
    "kyon": "kyu",
    "nahin": "nahi",
    "hai": "hai",
}


def normalise_hinglish(text: str) -> str:
    """Normalise common transliteration variants in Hinglish text.

    TODO(Phase 1): this is a placeholder word-level mapping; replace with the
    dictionary built from the actual datasets once downloaded.

    Args:
        text: Input Hinglish (Roman-script) text.

    Returns:
        Text with known variants mapped to canonical spellings.
    """
    if not text:
        return ""
    tokens = text.split()
    return " ".join(_VARIANT_MAP.get(tok.lower(), tok) for tok in tokens)


def is_code_mixed(text: str) -> bool:
    """Heuristic check for Roman-script Hindi function words in the text.

    TODO(Phase 1): replace with a proper code-mix / language-ID model.
    """
    if not text:
        return False
    tokens = set(text.lower().split())
    hinglish_markers = {
        "hai", "hain", "nahi", "kyu", "kya", "aur", "tum", "tuh", "mera",
        "tera", "usko", "isko", "yeh", "woh", "bhai", "kaiko", "matlab",
    }
    return bool(tokens & hinglish_markers)
