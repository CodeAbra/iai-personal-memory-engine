from __future__ import annotations

import re

EN_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("quoted-phrase",  re.compile(r'"[^"]+"')),
    ("european-quote", re.compile(r'«[^»]+»')),
    ("word-marker",    re.compile(r'\b(verbatim|exact|quote|quoted|said|wrote)\b', re.IGNORECASE)),
    ("day-N",          re.compile(r'\bday\s+\d+\b', re.IGNORECASE)),
]


EN_HISTORICAL_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("historical-en-original",   re.compile(r'\b(original|originally)\b', re.IGNORECASE)),
    ("historical-en-before",     re.compile(r'\bbefore\b', re.IGNORECASE)),
    ("historical-en-first",      re.compile(r'\b(first|initial|initially)\b', re.IGNORECASE)),
    ("historical-en-earlier",    re.compile(r'\bearlier\b', re.IGNORECASE)),
    ("historical-en-previously", re.compile(r'\b(previously|previous)\b', re.IGNORECASE)),
]



def _classify_cue(text: str) -> tuple[str, str | None, str | None]:
    if not text:
        return "concept", None, None

    mode = "concept"
    label: str | None = None
    for lbl, pat in EN_TRIGGERS:
        if pat.search(text):
            mode = "verbatim"
            label = lbl
            break

    intent: str | None = None
    for _lbl, pat in EN_HISTORICAL_TRIGGERS:
        if pat.search(text):
            intent = "historical_verbatim"
            break

    return mode, intent, label
