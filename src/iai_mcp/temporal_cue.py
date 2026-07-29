"""Explicit date mentions in a recall cue, matched against record dates.

Temporal binding lives at the rank layer: encoding the capture date into
the stored vector necessarily inflates same-day unrelated-pair cosine
past the similarity floors, so the vector stays pure text and a cue that
names a date instead earns matching-day records a bounded rank boost.

Precision over recall by construction: only explicit forms parse (ISO
dates, month names with optional day/year). A cue with no date mention
produces no mentions and the rank term never fires — non-temporal recall
is byte-identical.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
_MONTH_ABBREV = {name[:3]: num for name, num in _MONTHS.items()}

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")
_MONTH_NAME_RE = re.compile(
    r"\b(?:(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?)?"
    r"(january|february|march|april|may|june|july|august|september|october"
    r"|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"\.?(?:\s+(\d{1,2})(?:st|nd|rd|th)?)?"
    r"(?:,?\s+(\d{4}))?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DateMention:
    year: "int | None" = None
    month: "int | None" = None
    day: "int | None" = None


def _month_number(token: str) -> "int | None":
    t = token.lower().rstrip(".")
    if t in _MONTHS:
        return _MONTHS[t]
    if t == "sept":
        return 9
    return _MONTH_ABBREV.get(t[:3]) if len(t) == 3 else None


def parse_date_mentions(cue: str) -> "list[DateMention]":
    if not cue:
        return []
    mentions: list[DateMention] = []
    seen: set[DateMention] = set()

    for m in _ISO_RE.finditer(cue):
        year, month = int(m.group(1)), int(m.group(2))
        day = int(m.group(3)) if m.group(3) else None
        if not (1900 <= year <= 2200 and 1 <= month <= 12):
            continue
        if day is not None and not (1 <= day <= 31):
            continue
        mention = DateMention(year=year, month=month, day=day)
        if mention not in seen:
            seen.add(mention)
            mentions.append(mention)

    for m in _MONTH_NAME_RE.finditer(cue):
        token = m.group(2)
        month = _month_number(token)
        if month is None:
            continue
        day_raw = m.group(1) or m.group(3)
        day = int(day_raw) if day_raw else None
        if day is not None and not (1 <= day <= 31):
            day = None
        year = int(m.group(4)) if m.group(4) else None
        if year is not None and not (1900 <= year <= 2200):
            year = None
        # Ambiguous English words ("may I ask", "they march") and bare
        # abbreviations only count with a day or year attached.
        bare = day is None and year is None
        t_low = token.lower().rstrip(".")
        ambiguous = t_low in ("may", "march") or t_low not in _MONTHS
        if bare and ambiguous:
            continue
        mention = DateMention(year=year, month=month, day=day)
        if mention not in seen:
            seen.add(mention)
            mentions.append(mention)

    return mentions


def matches_mentions(created_at: "datetime | None", mentions) -> bool:
    """True when any mention's SPECIFIED components all equal the record's
    (unspecified components are wildcards: bare "July" spans every July)."""
    if created_at is None or not mentions:
        return False
    for mention in mentions:
        if mention.year is not None and created_at.year != mention.year:
            continue
        if mention.month is not None and created_at.month != mention.month:
            continue
        if mention.day is not None and created_at.day != mention.day:
            continue
        if (mention.year, mention.month, mention.day) != (None, None, None):
            return True
    return False
