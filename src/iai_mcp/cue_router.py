from __future__ import annotations

import re

EN_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("quoted-phrase",  re.compile(r'"[^"]+"')),
    ("european-quote", re.compile(r'«[^»]+»')),
    ("word-marker",    re.compile(r'\b(verbatim|exact|quote|quoted|said|wrote)\b', re.IGNORECASE)),
    ("day-N",          re.compile(r'\bday\s+\d+\b', re.IGNORECASE)),
]

RU_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("ru-start-найди-дословно",  re.compile(r'^найди дословно', re.IGNORECASE)),
    ("ru-start-точная-цитата",   re.compile(r'^точная цитата',  re.IGNORECASE)),
    ("ru-start-что-я-сказал",    re.compile(r'^что я сказал',    re.IGNORECASE)),
    ("ru-start-что-я-писал",     re.compile(r'^что я писал',     re.IGNORECASE)),
]

EN_HISTORICAL_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("historical-en-original",   re.compile(r'\b(original|originally)\b', re.IGNORECASE)),
    ("historical-en-before",     re.compile(r'\bbefore\b', re.IGNORECASE)),
    ("historical-en-first",      re.compile(r'\b(first|initial|initially)\b', re.IGNORECASE)),
    ("historical-en-earlier",    re.compile(r'\bearlier\b', re.IGNORECASE)),
    ("historical-en-previously", re.compile(r'\b(previously|previous)\b', re.IGNORECASE)),
]

RU_HISTORICAL_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("historical-ru-original",   re.compile(r'\b(оригинал|оригинальн\w*)\b', re.IGNORECASE)),
    ("historical-ru-snachala",   re.compile(r'\bсначала\b', re.IGNORECASE)),
    ("historical-ru-iznachal",   re.compile(r'\bизначальн\w*\b', re.IGNORECASE)),
    ("historical-ru-ranee",      re.compile(r'\bранее\b', re.IGNORECASE)),
]

# Correction-seeking span: 'вспомни' alone qualifies (fires even alongside a
# historical word like 'ранее'); 'напомни' requires an immediate current-
# practice companion ('как мы'/'правильн*') or it stays historical-eligible.
RU_CORRECTION_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("ru-correction-vspomni", re.compile(
        r'\bвспомни(?:ть)?\b\s*(?:мне\s*[,:]?\s*)?(?:как(?:\s+мы)?\s+)?(?:правильн\w*\s*)?',
        re.IGNORECASE,
    )),
    ("ru-correction-napomni-companion", re.compile(
        r'\bнапомни(?:ть)?\b\s*(?:мне\s*[,:]?\s*)?(?:как\s+мы\s+|правильн\w*\s*)',
        re.IGNORECASE,
    )),
    ("ru-correction-kakoy-pravilno", re.compile(
        r'\bкакой\s+правильн\w*\b\s*', re.IGNORECASE,
    )),
    ("ru-correction-ne-tak-a-vot-tak", re.compile(
        r'\bне\s+так[,]?\s+а\s+вот\s+так\b\s*', re.IGNORECASE,
    )),
    ("ru-correction-na-samom-dele", re.compile(
        r'\bна\s+самом\s+деле\b\s*', re.IGNORECASE,
    )),
]

EN_CORRECTION_TRIGGERS: list[tuple[str, re.Pattern]] = [
    ("en-correction-remind-me",     re.compile(r'\bremind me\b\s*', re.IGNORECASE)),
    ("en-correction-whats-correct", re.compile(r"\bwhat'?s the correct\b\s*", re.IGNORECASE)),
    ("en-correction-how-do-we",     re.compile(r'\bhow do we\b\s*', re.IGNORECASE)),
]


def strip_correction_trigger(text: str) -> str:
    """Remove the matched correction-seeking trigger span, leaving the
    topical remainder as the cue text to embed."""
    if not text:
        return text
    stripped = text
    for _lbl, pat in RU_CORRECTION_TRIGGERS:
        stripped = pat.sub("", stripped, count=1)
    for _lbl, pat in EN_CORRECTION_TRIGGERS:
        stripped = pat.sub("", stripped, count=1)
    return stripped.strip()


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
    if mode != "verbatim":
        for lbl, pat in RU_TRIGGERS:
            if pat.search(text):
                mode = "verbatim"
                label = lbl
                break

    is_correction = (
        any(pat.search(text) for _lbl, pat in EN_CORRECTION_TRIGGERS)
        or any(pat.search(text) for _lbl, pat in RU_CORRECTION_TRIGGERS)
    )

    # A correction-seeking cue must stay on the default (None) intent path:
    # that path is what carries apply_stale_downweight/apply_supersede_cap.
    # Flipping it to historical_verbatim would disable supersession for
    # exactly the turn class this trigger targets.
    intent: str | None = None
    if not is_correction:
        for _lbl, pat in EN_HISTORICAL_TRIGGERS:
            if pat.search(text):
                intent = "historical_verbatim"
                break
        if intent is None:
            for _lbl, pat in RU_HISTORICAL_TRIGGERS:
                if pat.search(text):
                    intent = "historical_verbatim"
                    break

    return mode, intent, label
