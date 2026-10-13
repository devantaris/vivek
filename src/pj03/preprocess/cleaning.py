"""Text cleaning for social media hate-speech data.

Kept deliberately light and lossless-by-default: aggressive cleaning (e.g.
stripping emoji or hashtags) can destroy the very cues models and XAI methods
need. Phase 1 experiments should compare raw vs cleaned variants.
"""

from __future__ import annotations

import html
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
# Repeated-character elongations common in tweets: "sooooo" -> "sooo"
_ELONGATION_RE = re.compile(r"(.)\1{2,}")


def clean_text(text: str, strip_mentions: bool = False) -> str:
    """Normalise whitespace, unescape HTML entities, and optionally drop @mentions.

    URLs are removed (they carry no hate-speech signal and inflate vocab),
    mentions are kept by default because targets can matter, and elongations
    are truncated to 3 repeats so tokenisers see fewer OOV variants.

    Args:
        text: Raw input text.
        strip_mentions: If True, remove ``@user`` mentions.

    Returns:
        The cleaned text (never None; empty string in -> empty string out).
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    if strip_mentions:
        text = _MENTION_RE.sub(" ", text)
    text = _ELONGATION_RE.sub(r"\1\1\1", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
