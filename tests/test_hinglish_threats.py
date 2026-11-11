"""Unit tests for extended Hinglish threat pattern detection.

Covers both canonical forms and common transliteration variants (e.g.
*chahliye* / *chahiye*, *maaro* / *mar*) that were observed as failure modes
in the Oct-2026 demo session.
"""

from __future__ import annotations

import pytest

from pj03.preprocess.hinglish import detect_hinglish_slurs, is_code_mixed

# ---------------------------------------------------------------------------
# Threat pattern variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "modi ko mar jana chahiye",
        "modi ko mar jana chahliye",  # common mispelling variant
        "unhe mar jana chahiye",
        "sab ko mar jana chahiye",
    ],
)
def test_mar_jana_chahiye_variants(text: str) -> None:
    """All *mar jana chahiye* transliteration variants must be detected."""
    slurs = detect_hinglish_slurs(text)
    assert any("mar jana" in s for s in slurs), (
        f"Expected mar-jana threat in {text!r}, got {slurs!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        "yeh log mar jao",
        "sab mar jaye",
        "inhe mar do",
        "unhe khatam karo",
        "goli maar do unhe",
    ],
)
def test_violent_threat_patterns(text: str) -> None:
    """Other violent Hinglish threat phrases must be detected."""
    slurs = detect_hinglish_slurs(text)
    assert slurs, f"Expected threat detection in {text!r}, got {slurs!r}"


# ---------------------------------------------------------------------------
# Toxic term normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "modi madharchod hai",   # variant -> madarchod (canonical)
        "woh behenchod hai",
        "chutiye log",
        "haramzade",
    ],
)
def test_toxic_term_detection(text: str) -> None:
    """Toxic Hinglish terms (and their spelling variants) must be detected."""
    slurs = detect_hinglish_slurs(text)
    assert slurs, f"Expected toxic term detection in {text!r}, got {slurs!r}"


# ---------------------------------------------------------------------------
# Clean texts must NOT trigger false positives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "aaj bahut acha din hai",
        "kya haal hai bhai",
        "namaste dosto",
        "good morning everyone",
    ],
)
def test_clean_texts_no_false_positive(text: str) -> None:
    """Benign Hinglish text should not be flagged as a slur."""
    slurs = detect_hinglish_slurs(text)
    assert not slurs, f"False positive for {text!r}: got {slurs!r}"


# ---------------------------------------------------------------------------
# Code-mix detection
# ---------------------------------------------------------------------------


def test_code_mixed_positives() -> None:
    for text in ["modi ko mar jana chahliye", "modi madharchod hai"]:
        assert is_code_mixed(text), f"Should be code-mixed: {text!r}"


def test_code_mixed_negative() -> None:
    assert not is_code_mixed("hello good morning"), "English-only should not be code-mixed"
