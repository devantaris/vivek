"""Synthetic adversarial perturbation engine.

Generates adversarial variations of training examples modeling HateCheck
functional test categories:
  - spell_char_swap_h: Transposing adjacent characters inside words.
  - spell_char_del_h: Dropping single characters.
  - spell_leetspeak: Substituting letters with leetspeak symbols / numbers.
  - phrase_question_h: Wrapping toxic statements in rhetorical inquiry frames.
"""

from __future__ import annotations

import random
import re

_LEET_MAP: dict[str, list[str]] = {
    "a": ["@", "4"],
    "e": ["3"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["$", "5"],
    "t": ["7"],
}

_QUESTION_TEMPLATES: list[str] = [
    "Why do people say {text}?",
    "Is it true that {text}?",
    "Can someone explain why {text}?",
    "Do you actually believe {text}?",
    "Why would anyone think {text}?",
]


def perturb_char_swap(text: str, rng: random.Random | None = None) -> str:
    """Swap two adjacent internal characters in a word with length >= 4."""
    if rng is None:
        rng = random.Random()
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not candidates:
        return text
    idx = rng.choice(candidates)
    word = list(words[idx])
    # Swap two internal characters (avoid first and last letters if possible)
    swap_pos = rng.randint(1, len(word) - 2) if len(word) > 3 else 0
    word[swap_pos], word[swap_pos + 1] = word[swap_pos + 1], word[swap_pos]
    words[idx] = "".join(word)
    return " ".join(words)


def perturb_char_del(text: str, rng: random.Random | None = None) -> str:
    """Delete one internal character in a word with length >= 4."""
    if rng is None:
        rng = random.Random()
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not candidates:
        return text
    idx = rng.choice(candidates)
    word = list(words[idx])
    del_pos = rng.randint(1, len(word) - 2) if len(word) > 3 else 0
    del word[del_pos]
    words[idx] = "".join(word)
    return " ".join(words)


def perturb_leetspeak(text: str, rng: random.Random | None = None) -> str:
    """Substitute vowel/sibilant characters with leetspeak equivalents."""
    if rng is None:
        rng = random.Random()
    chars = []
    for c in text:
        lower_c = c.lower()
        if lower_c in _LEET_MAP and rng.random() < 0.4:
            replacement = rng.choice(_LEET_MAP[lower_c])
            chars.append(replacement)
        else:
            chars.append(c)
    return "".join(chars)


def perturb_question_framing(text: str, rng: random.Random | None = None) -> str:
    """Wrap a statement inside an evasive / rhetorical inquiry frame."""
    if rng is None:
        rng = random.Random()
    # Strip terminal period or exclamation
    clean = re.sub(r"[\.!\?]+$", "", text.strip())
    template = rng.choice(_QUESTION_TEMPLATES)
    return template.format(text=clean)


def generate_adversarial_variants(
    text: str,
    label: int,
    num_variants: int = 2,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Produce adversarially perturbed variants preserving original label."""
    rng = random.Random(seed)
    variants: list[tuple[str, int]] = []
    perturb_fns = [perturb_char_swap, perturb_char_del, perturb_leetspeak]
    if label == 1:
        # Question framing is only valid for toxic assertions
        perturb_fns.append(perturb_question_framing)

    for _ in range(num_variants):
        fn = rng.choice(perturb_fns)
        perturbed = fn(text, rng=rng)
        if perturbed and perturbed != text:
            variants.append((perturbed, label))
    return variants
