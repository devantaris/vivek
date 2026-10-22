"""Tests for serving-time PII scrubbing."""

from __future__ import annotations

from pj03.preprocess.pii import PLACEHOLDERS, has_pii, scrub


def test_each_pattern() -> None:
    assert scrub("mail me at john.doe@example.com now") == (
        f"mail me at {PLACEHOLDERS['email']} now"
    )
    assert scrub("see https://x.com/post?a=1 please") == (f"see {PLACEHOLDERS['url']} please")
    assert scrub("hey @devansh look") == f"hey {PLACEHOLDERS['mention']} look"
    assert scrub("call +91 98765 43210 ok") == (f"call {PLACEHOLDERS['phone']} ok")


def test_email_inside_url_not_double_masked() -> None:
    out = scrub("https://mail.example.com/u=user@example.com")
    assert out.count("[") == 1  # the whole URL went as one [URL]
    assert "[EMAIL]" not in out


def test_mention_inside_email_preserved_order() -> None:
    out = scrub("contact user@example.com or @handle")
    assert out == f"contact {PLACEHOLDERS['email']} or {PLACEHOLDERS['mention']}"


def test_clean_text_untouched() -> None:
    text = "kill all jews and i hate this"
    assert scrub(text) == text
    assert not has_pii(text)


def test_has_pii_flags() -> None:
    assert has_pii("write to a@b.io")
    assert has_pii("@user hi")
    assert has_pii("go to www.site.com")
    assert has_pii("9876543210")


def test_empty() -> None:
    assert scrub("") == ""
    assert not has_pii("")


def test_numbers_not_overmasked() -> None:
    # Short numbers (years, counts) must survive.
    assert scrub("in 2024 there were 45 cases") == "in 2024 there were 45 cases"
