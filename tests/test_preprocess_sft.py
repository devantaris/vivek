"""Smoke tests for cleaning, Hinglish helpers and prompt parsing."""

from __future__ import annotations

from pj03.preprocess.cleaning import clean_text
from pj03.preprocess.hinglish import (
    code_mixed_ratio,
    detect_hinglish_slurs,
    is_code_mixed,
    normalise_hinglish,
)
from pj03.sft.prompt_templates import format_prompt, parse_output


def test_clean_text() -> None:
    assert clean_text("check  https://x.com/a  this  out &amp; now") == "check this out & now"
    assert clean_text("@user you are awful", strip_mentions=True) == "you are awful"
    assert clean_text("sooooo bad") == "sooo bad"
    assert clean_text("") == ""


def test_hinglish_helpers() -> None:
    assert "kyu" in normalise_hinglish("why kyon so")
    assert "bohot" in normalise_hinglish("bahut bhot accha")
    assert is_code_mixed("yeh post bahut kharab hai")
    assert not is_code_mixed("this post is terrible")
    assert code_mixed_ratio("bhai yeh kya hai yaar") > 0.6
    assert code_mixed_ratio("this is a pure english test") == 0.0

    # Slur detection
    slurs = detect_hinglish_slurs("tu bohot bada kutta aur chutiye hai")
    assert "kutta" in slurs
    assert "chutiya" in slurs  # canonicalized from chutiye
    assert detect_hinglish_slurs("namaste dosto kaise ho") == []


def test_prompt_format_and_parse() -> None:
    prompt = format_prompt("some post")
    assert "some post" in prompt
    label, rationale = parse_output("Label: 1\nRationale: uses a slur")
    assert label == 1 and rationale == "uses a slur"
    assert parse_output("no format at all") == (None, None)
