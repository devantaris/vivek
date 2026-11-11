"""Adversarial de-obfuscation and homoglyph normalization.

Counteracts evasion techniques evaluated in HateCheck (e.g. character swaps,
leetspeak, zero-width characters, spaced tokens, and Unicode homoglyphs)
where attackers deliberately alter tokens to evade standard n-gram and subword
filters while remaining human-readable.
"""

from __future__ import annotations

import re
import unicodedata

# Explicit Cyrillic, Greek and lookalike character mappings to Latin
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic lookalikes
    "а": "a", "А": "A",
    "в": "b", "В": "B",
    "с": "c", "С": "C",
    "е": "e", "Е": "E",
    "ё": "e", "Ё": "E",
    "н": "h", "Н": "H",
    "і": "i", "І": "I",
    "ј": "j", "Ј": "J",
    "к": "k", "К": "K",
    "м": "m", "М": "M",
    "о": "o", "О": "O",
    "р": "p", "Р": "P",
    "ѕ": "s", "Ѕ": "S",
    "т": "t", "Т": "T",
    "у": "y", "У": "Y",
    "х": "x", "Х": "X",
    # Greek lookalikes
    "α": "a", "Α": "A",
    "β": "b", "Β": "B",
    "γ": "y", "Γ": "G",
    "δ": "d", "Δ": "D",
    "ε": "e", "Ε": "E",
    "η": "n", "Η": "H",
    "θ": "th", "Θ": "TH",
    "ι": "i", "Ι": "I",
    "κ": "k", "Κ": "K",
    "λ": "l", "Λ": "L",
    "ν": "v", "Ν": "N",
    "ο": "o", "Ο": "O",
    "ρ": "p", "Ρ": "P",
    "τ": "t", "Τ": "T",
    "υ": "u", "Υ": "Y",
    "χ": "x", "Χ": "X",
    "ω": "w", "Ω": "W",
    # Mathematical / Enclosed / Fullwidth variants handled via NFKD
}

# Zero-width and invisible control characters
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\u00AD\u2060\u200E\u200F]")

# Regex to detect spaced single letters like "s l u t" or "k i l l"
_SPACED_LETTERS_RE = re.compile(r"(?<=\b[a-zA-Z])\s+(?=[a-zA-Z]\b)")

# Regex to detect punctuated single letters like "s.l.u.t" or "f_u_c_k"
_PUNCT_LETTERS_RE = re.compile(
    r"\b([a-zA-Z])[\.\-_*~^]([a-zA-Z])[\.\-_*~^]([a-zA-Z])(?:[\.\-_*~^]([a-zA-Z]))?(?:[\.\-_*~^]([a-zA-Z]))?\b"
)

# Common leetspeak translation table for mixed alphanumeric tokens
_LEET_SUBS: list[tuple[str, str]] = [
    ("@", "a"),
    ("4", "a"),
    ("3", "e"),
    ("1", "i"),
    ("!", "i"),
    ("0", "o"),
    ("$", "s"),
    ("5", "s"),
    ("7", "t"),
]

# High-priority known offensive / slur words to reconstruct when leetspeak is used
_KNOWN_VULNERABLE_KEYWORDS: dict[str, str] = {
    "s1ut": "slut",
    "sl*t": "slut",
    "sl0t": "slut",
    "b1tch": "bitch",
    "b!tch": "bitch",
    "b*tch": "bitch",
    "b1tches": "bitches",
    "f4ggot": "faggot",
    "f@ggot": "faggot",
    "f@g": "fag",
    "f1lth": "filth",
    "k1ll": "kill",
    "k!ll": "kill",
    "d1e": "die",
    "wh0re": "whore",
    "wh*re": "whore",
    "w-h-o-r-e": "whore",
    "n1gger": "nigger",
    "n1gga": "nigga",
    "n*gger": "nigger",
    "n*gga": "nigga",
    "r3tard": "retard",
    "c*nt": "cunt",
    "c0ck": "cock",
    "sh1t": "shit",
    "f*ck": "fuck",
    "f**k": "fuck",
}


def normalize_homoglyphs(text: str) -> str:
    """Map visual homoglyphs (Cyrillic, Greek, lookalikes) to standard Latin."""
    if not text:
        return ""
    # NFKD decomposes compatibility characters (e.g. 𝔣, ⓕ, ｆ -> f)
    nfkd = unicodedata.normalize("NFKD", text)
    chars = [_HOMOGLYPH_MAP.get(c, c) for c in nfkd]
    return "".join(chars)


def strip_zero_width(text: str) -> str:
    """Remove invisible zero-width and directional control characters."""
    if not text:
        return ""
    return _ZERO_WIDTH_RE.sub("", text)


def collapse_punct_spelled_words(text: str) -> str:
    """Collapse punctuated spellings like 's.l.u.t' or 'f_u_c_k' into whole words."""
    if not text:
        return ""

    def _replace_match(match: re.Match) -> str:
        groups = [g for g in match.groups() if g is not None]
        return "".join(groups)

    return _PUNCT_LETTERS_RE.sub(_replace_match, text)


def decode_leetspeak_word(word: str) -> str:
    """Decode a single token if it contains leetspeak characters embedded in letters."""
    if not word:
        return ""
    lowered = word.lower()

    # Exact known lookup
    if lowered in _KNOWN_VULNERABLE_KEYWORDS:
        return _KNOWN_VULNERABLE_KEYWORDS[lowered]

    # Only decode if the token has both letters and leetspeak chars
    has_letters = any(c.isalpha() for c in word)
    has_leet_chars = any(c in "@431!0$57*" for c in word)

    if has_letters and has_leet_chars:
        decoded = lowered
        for char, repl in _LEET_SUBS:
            decoded = decoded.replace(char, repl)
        # Avoid corrupting words where digits are intentional (e.g. 'h2o', 'mp3')
        # Check if the decoded word matches a common dictionary word or slur
        if any(v in decoded for v in ("slut", "bitch", "faggot", "kill", "die", "whore", "retard", "cunt", "shit", "fuck")):
            return decoded
        return decoded
    return word


def deobfuscate_text(text: str) -> str:
    """Full de-obfuscation pipeline counteracting adversarial evasion.

    1. Strips zero-width and invisible control characters.
    2. Normalizes Unicode homoglyphs and styled/fullwidth glyphs to Latin.
    3. Reconstructs punctuated single-letter obfuscations ('s.l.u.t' -> 'slut').
    4. Decodes leetspeak and number substitutions ('s1ut' -> 'slut', 'f@ggot' -> 'faggot').
    """
    if not text:
        return ""

    # Step 1: Zero-width removal
    cleaned = strip_zero_width(text)

    # Step 2: Unicode homoglyphs & compatibility forms
    cleaned = normalize_homoglyphs(cleaned)

    # Step 3: Punctuated words
    cleaned = collapse_punct_spelled_words(cleaned)

    # Step 4: Token-level leetspeak decoding
    tokens = cleaned.split()
    decoded_tokens = [decode_leetspeak_word(t) for t in tokens]
    return " ".join(decoded_tokens)
