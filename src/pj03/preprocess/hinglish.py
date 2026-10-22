"""Hinglish (Hindi-English code-mixed) normalisation, detection and lexicon tools.

Provides:
    - Transliteration canonicalization for high-frequency Romanized Hindi tokens.
    - Heuristic code-mix language identification and scoring.
    - Identification of Hinglish abusive and toxic markers.
"""

from __future__ import annotations

import re

# Comprehensive canonicalization dictionary for Romanized Hindi transliterations.
_VARIANT_MAP: dict[str, str] = {
    # Questions / Question words
    "kyun": "kyu",
    "kyon": "kyu",
    "kyo": "kyu",
    "kewn": "kyu",
    "kaha": "kahan",
    "kahaan": "kahan",
    "kidhar": "kidhar",
    "kaise": "kaise",
    "kese": "kaise",
    # Negations
    "nahin": "nahi",
    "nhi": "nahi",
    "nhin": "nahi",
    "naa": "na",
    "mat": "mat",
    # Common function words & pronouns
    "bahut": "bohot",
    "bhot": "bohot",
    "bhut": "bohot",
    "boht": "bohot",
    "acha": "achha",
    "accha": "achha",
    "achaa": "achha",
    "theek": "thik",
    "thik": "thik",
    "lekin": "lekin",
    "parantu": "lekin",
    "magar": "lekin",
    "aur": "aur",
    "or": "aur",
    "tume": "tumhe",
    "tumhein": "tumhe",
    "muze": "mujhe",
    "mjhe": "mujhe",
    "mujhko": "mujhe",
    "apko": "aapko",
    "aapkoh": "aapko",
    "unko": "unhe",
    "unhein": "unhe",
    "isko": "ise",
    "usko": "use",
    "kisko": "kise",
    "bhai": "bhai",
    "bhaai": "bhai",
    "bhay": "bhai",
    "bro": "bhai",
    "dost": "dost",
    "yaar": "yaar",
    "yrr": "yaar",
    "log": "log",
    "logon": "log",
    "wala": "wala",
    "wali": "wali",
    "wale": "wale",
    # Common verbs / aux
    "krna": "karna",
    "kr": "kar",
    "kro": "karo",
    "karu": "karu",
    "raha": "raha",
    "rha": "raha",
    "rahe": "rahe",
    "rhe": "rahe",
    "rahi": "rahi",
    "rhi": "rahi",
    "hoga": "hoga",
    "hoge": "hoge",
    "hogi": "hogi",
    "hai": "hai",
    "hain": "hain",
    "h": "hai",
    "hn": "hain",
    # Common Hinglish abusive / derogatory terms (canonicalization)
    "chutiya": "chutiya",
    "chutiye": "chutiya",
    "chutiyein": "chutiya",
    "c**tiya": "chutiya",
    "ch*tiya": "chutiya",
    "kutta": "kutta",
    "kutte": "kutta",
    "kutti": "kutta",
    "kutto": "kutta",
    "kutton": "kutta",
    "mulla": "mulla",
    "mulle": "mulla",
    "mullon": "mulla",
    "kamina": "kamina",
    "kamine": "kamina",
    "kaminey": "kamina",
    "saala": "saala",
    "saale": "saala",
    "sale": "saala",
    "harami": "harami",
    "haramzada": "haramzada",
    "haramzade": "haramzada",
    "madarchod": "madarchod",
    "mc": "madarchod",
    "behenchod": "behenchod",
    "bhenchod": "behenchod",
    "bc": "behenchod",
    "gandu": "gandu",
    "randi": "randi",
    "bhadwe": "bhadwa",
    "bhadwa": "bhadwa",
    "aukaat": "aukaat",
    "aukat": "aukaat",
}

# High-frequency Roman-script Hindi markers for code-mixing detection
HINGLISH_MARKERS: frozenset[str] = frozenset({
    "hai",
    "hain",
    "nahi",
    "kyu",
    "kya",
    "aur",
    "tum",
    "tuh",
    "mera",
    "tera",
    "usko",
    "isko",
    "yeh",
    "woh",
    "bhai",
    "kaiko",
    "matlab",
    "apna",
    "apni",
    "karo",
    "karna",
    "hoga",
    "hogi",
    "bohot",
    "bahut",
    "achha",
    "sab",
    "kuch",
    "kuchh",
    "wala",
    "wali",
    "wale",
    "raha",
    "rahi",
    "rahe",
    "chup",
    "pagal",
    "yaar",
    "dost",
    "mar",
    "jao",
    "aao",
    "dekho",
    "bolo",
    "sun",
    "chal",
    "nikal",
})

# Abusive / toxic Hinglish tokens
HINGLISH_TOXIC_TERMS: frozenset[str] = frozenset({
    "chutiya",
    "kutta",
    "mulla",
    "kamina",
    "saala",
    "harami",
    "haramzada",
    "madarchod",
    "behenchod",
    "gandu",
    "randi",
    "bhadwa",
    "suar",
    "gadha",
    "ullu",
    "lauda",
    "loda",
    "chut",
    "gaand",
    "aukaat",
    "katwe",
    "bhangi",
    "chamar",
})


def normalise_hinglish(text: str) -> str:
    """Normalise common transliteration variants in Hinglish text.

    Args:
        text: Input Hinglish (Roman-script) text.

    Returns:
        Text with known phonetic variants mapped to canonical spellings.
    """
    if not text:
        return ""
    words = re.findall(r"\b\w+\b|[^\w\s]", text)
    canonical = [_VARIANT_MAP.get(w.lower(), w) for w in words]
    # Reconstruct text spacing cleanly
    return re.sub(r"\s+([^\w\s])", r"\1", " ".join(canonical))


def is_code_mixed(text: str) -> bool:
    """Heuristic check for Roman-script Hindi function words in the text."""
    if not text:
        return False
    tokens = {w.lower() for w in re.findall(r"\b\w+\b", text)}
    return bool(tokens & HINGLISH_MARKERS)


def code_mixed_ratio(text: str) -> float:
    """Compute the fraction of tokens in ``text`` that match Hinglish markers."""
    if not text:
        return 0.0
    tokens = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    if not tokens:
        return 0.0
    matches = sum(1 for tok in tokens if tok in HINGLISH_MARKERS or tok in HINGLISH_TOXIC_TERMS)
    return matches / len(tokens)


def detect_hinglish_slurs(text: str) -> list[str]:
    """Identify toxic or abusive Hinglish terms present in the text."""
    if not text:
        return []
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    normalized = [_VARIANT_MAP.get(w, w) for w in words]
    found = [term for term in normalized if term in HINGLISH_TOXIC_TERMS]
    return list(dict.fromkeys(found))  # preserve order, unique
