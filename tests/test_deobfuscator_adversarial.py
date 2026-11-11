"""Unit tests for adversarial de-obfuscation and perturbation generation."""

from __future__ import annotations

import random

from pj03.preprocess.adversarial import (
    generate_adversarial_variants,
    perturb_char_del,
    perturb_char_swap,
    perturb_leetspeak,
    perturb_question_framing,
)
from pj03.preprocess.deobfuscator import (
    collapse_punct_spelled_words,
    decode_leetspeak_word,
    deobfuscate_text,
    normalize_homoglyphs,
    strip_zero_width,
)


def test_strip_zero_width() -> None:
    text_with_zw = "k\u200bi\u200cl\u200dl"
    assert strip_zero_width(text_with_zw) == "kill"


def test_normalize_homoglyphs() -> None:
    # Cyrillic 'а', 'е', 'о' look identical to Latin 'a', 'e', 'o'
    cyrillic_word = "\u0441\u0430\u0442"  # Cyrillic 'с', 'а', 'т'
    assert normalize_homoglyphs(cyrillic_word) == "cat"


def test_collapse_punct_spelled_words() -> None:
    assert collapse_punct_spelled_words("you are a s.l.u.t") == "you are a slut"
    assert collapse_punct_spelled_words("f_u_c_k you") == "fuck you"


def test_decode_leetspeak_word() -> None:
    assert decode_leetspeak_word("s1ut") == "slut"
    assert decode_leetspeak_word("f@ggot") == "faggot"
    assert decode_leetspeak_word("k1ll") == "kill"
    assert decode_leetspeak_word("clean") == "clean"


def test_deobfuscate_full_pipeline() -> None:
    evasive = "you dirty s.l.u.t go d1e in a f1re"
    cleaned = deobfuscate_text(evasive)
    assert "slut" in cleaned
    assert "die" in cleaned


def test_perturb_char_swap() -> None:
    rng = random.Random(42)
    original = "people are wonderful"
    perturbed = perturb_char_swap(original, rng=rng)
    assert perturbed != original
    # Word count preserved
    assert len(perturbed.split()) == len(original.split())


def test_perturb_char_del() -> None:
    rng = random.Random(42)
    original = "women are leaders"
    perturbed = perturb_char_del(original, rng=rng)
    assert perturbed != original


def test_perturb_leetspeak() -> None:
    rng = random.Random(42)
    original = "hate this community"
    perturbed = perturb_leetspeak(original, rng=rng)
    assert any(c in "@431!0$57" for c in perturbed)


def test_perturb_question_framing() -> None:
    statement = "immigrants destroy jobs"
    framed = perturb_question_framing(statement)
    assert "?" in framed
    assert "immigrants destroy jobs" in framed


def test_generate_adversarial_variants() -> None:
    variants = generate_adversarial_variants("all women are terrible", label=1, num_variants=3, seed=123)
    assert len(variants) >= 1
    assert all(v[1] == 1 for v in variants)
