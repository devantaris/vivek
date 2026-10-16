"""PII scrubbing for serving-time inputs and logs.

Redacts emails, URLs, @mentions and phone numbers with deterministic
placeholders so raw user data never lands in stored decision logs.

Scope note: this is a *serving/retention* utility, deliberately NOT part of
the training pipeline - HateXplain rationale indices point into the original
token sequence, and the models were trained on that distribution. Scrubbing
at inference is also available (``scrub_pii=True`` in the API) but defaults
to off so predictions match training conditions; logs are always scrubbed.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s-]{7,14}\d)(?!\w)")

PLACEHOLDERS = {
    "email": "[EMAIL]",
    "url": "[URL]",
    "mention": "[USER]",
    "phone": "[PHONE]",
}


def scrub(text: str) -> str:
    """Redact common PII patterns with deterministic placeholders.

    Order matters: emails and URLs first (so @handles inside them are not
    double-masked), then mentions, then phones.

    Args:
        text: Raw input text.

    Returns:
        Text with PII replaced; empty string in -> empty string out.
    """
    if not text:
        return ""
    text = _EMAIL_RE.sub(PLACEHOLDERS["email"], text)
    text = _URL_RE.sub(PLACEHOLDERS["url"], text)
    text = _MENTION_RE.sub(PLACEHOLDERS["mention"], text)
    text = _PHONE_RE.sub(PLACEHOLDERS["phone"], text)
    return text


def has_pii(text: str) -> bool:
    """True if any scrubber pattern matches (used to flag log entries)."""
    if not text:
        return False
    return any(
        pattern.search(text)
        for pattern in (_EMAIL_RE, _URL_RE, _MENTION_RE, _PHONE_RE)
    )
