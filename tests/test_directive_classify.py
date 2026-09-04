"""Pure classifier recognizes standing-order phrasings, including scoped
always/never forms."""

from __future__ import annotations

from iai_mcp.directive_classify import classify_is_directive


def test_classifier_recognizes_standing_order_phrasings():
    assert classify_is_directive("from now on reply in English") is True
    assert classify_is_directive(
        "remember this across all sessions: I go by Alex"
    ) is True
    assert classify_is_directive("save this for all sessions") is True
    assert classify_is_directive("always reply in English") is True
    assert classify_is_directive("never use markdown in commit messages") is True


def test_classifier_ignores_bare_descriptive_always_never():
    assert classify_is_directive("the build always takes twenty minutes") is False
    assert classify_is_directive("she never uses the staging branch") is False
    assert classify_is_directive("that test always flakes on arm") is False
    assert classify_is_directive("we never got a reply from the vendor") is False


def test_classifier_ignores_plain_chit_chat():
    assert classify_is_directive("the weather is nice today") is False
    assert classify_is_directive("the meeting is scheduled for next Monday") is False


def test_classifier_fails_safe_on_non_string_or_empty():
    assert classify_is_directive(None) is False  # type: ignore[arg-type]
    assert classify_is_directive("") is False
    assert classify_is_directive(123) is False  # type: ignore[arg-type]
    assert classify_is_directive("   ") is False


def test_classifier_scan_cap_ignores_signal_beyond_2000_chars():
    padding = "x" * 2000
    text = padding + " always reply in English"
    assert classify_is_directive(text) is False
