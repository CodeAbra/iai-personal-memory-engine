"""Deterministic entity-anchor extraction for capture-time tagging.

Anchors become ``entity:`` tags and flow into the AAAK E: field; the
rank's anchor-overlap term matches cues against them. Runs on the
capture hot path: pure string processing, input capped, every regex
free of nested-quantifier ambiguity (a pathological token must never
stall capture). Anchor values never contain ``/``, ``:`` or ``,`` —
those are AAAK structural separators and would corrupt the index.

Anchors are stored in plaintext columns (tags_json, aaak_index) by
decision: the bank layers already hold plaintext summaries under the
same local-only threat model, and the lexical index requires cleartext.
"""
from __future__ import annotations

import re

MAX_ENTITIES = 16
_MIN_LEN = 3
_MAX_LEN = 40
_SCAN_CAP = 12000

_BACKTICKED = re.compile(r"`([^`\n]{2,60})`")
_HANDLE = re.compile(r"(?<!\w)@([A-Za-z0-9_]{3,32})")
# Head has NO underscore so the separator split is unambiguous — the
# engine never backtracks across `_` partitions.
_IDENTIFIER = re.compile(
    r"(?<![\w./-])([A-Za-zЀ-ӿ][A-Za-z0-9Ѐ-ӿ]*"
    r"(?:[._-][A-Za-z0-9Ѐ-ӿ]+)+)(?![\w/-])"
)
_CAMEL = re.compile(
    r"(?<![\w])([a-z]+[A-Z][A-Za-z]+|[A-Z][a-z]+(?:[A-Z][a-z]+)+)(?![\w])"
)
_CAPITALIZED = re.compile(
    r"(?<![A-Za-zА-ЯЁа-яёЀ-ӿ])([A-ZА-ЯЁ][a-zа-яё]{2,})(?![A-Za-zА-ЯЁа-яёЀ-ӿ])"
)
_SENTENCE_BOUNDARY = re.compile(r"^|[.!?…]\s+|\n\s*")

_CAP_DENYLIST = frozenset({
    # English function/furniture words that survive the sentence-start
    # filter via quoting, dashes or mid-sentence capitalization.
    "the", "this", "that", "these", "those", "there", "here", "when",
    "what", "which", "with", "from", "then", "have", "been", "will",
    "would", "should", "could", "about", "after", "before", "into",
    "over", "under", "other", "another", "some", "such", "than", "them",
    "they", "your", "please", "thanks", "using", "request",
    # Path furniture: basename extraction turns /Users/... segments into
    # capitalized tokens.
    "users", "desktop", "library", "applications", "downloads", "home",
    # Harness-envelope vocabulary: quoted notifications must never anchor.
    "task-notification", "task-id", "tool-use-id", "output-file",
    "session-id", "record-id", "caveat", "local-command-stdout",
    "local-command-caveat", "command-name", "command-message",
    "system-reminder",
    # Russian function words, incl. the hyphen-joined forms the
    # identifier lane would otherwise treat as names.
    "они", "это", "этот", "эта", "эти", "этого", "этой", "этом", "этих",
    "если", "когда", "чтобы", "чтоб", "потому", "поэтому", "также",
    "теперь", "что", "как", "для", "или", "оно", "она", "его", "нее",
    "неё", "все", "всё", "вот", "так", "уже", "еще", "ещё", "был",
    "была", "было", "были", "быть", "есть", "нет", "при", "под", "над",
    "без", "про", "после", "перед", "через", "между", "потом", "тоже",
    "только", "очень", "надо", "нужно", "можно", "кто", "где", "куда",
    "зачем", "почему", "пока", "сейчас", "того", "тогда", "здесь",
    "там", "даже", "меня", "тебя", "себя", "него",
    # Generic nouns/adjectives/numerals that carry no naming power.
    "итог", "итоги", "стадия", "одно", "один", "одна", "два", "две",
    "три", "результат", "результаты", "дальше", "состояние", "живой",
    "лучшая", "лучший", "важная", "важный", "хочешь", "хочу", "серия",
    "вопрос", "ответ", "правки", "готово", "план", "шаг", "шаги",
    "true", "false", "none", "null", "code", "phase", "wave", "step",
    "note", "next", "done", "test", "tests", "fixed", "ready",
    "чего", "чему", "чем", "кого", "кому", "кем", "ком", "тому",
    "этому", "этим", "мне", "мной", "тебе", "тобой", "нам", "нами",
    "вам", "вами", "ней", "ним", "ними", "себе", "собой", "весь",
    "вся", "всей", "всех", "всем", "всеми",
    "по-прежнему", "во-первых", "во-вторых", "в-третьих",
    "что-нибудь", "кто-нибудь", "как-нибудь", "где-нибудь",
})

# Short hyphen-joined pure-Cyrillic tokens are overwhelmingly function
# words in case forms (composed of particles), not names.
_CYRILLIC_PARTICLE = re.compile(r"^[а-яёѐ-ӿ]{1,6}-[а-яёѐ-ӿ]{1,6}$")

# Anchors are lowercase names: letters, digits, dots, underscores,
# hyphens, apostrophes. Anything else is a code or formula fragment,
# not a name.
_ANCHOR_SHAPE = re.compile(r"^[a-z0-9Ѐ-ӿ._'\-]+$")


def _clean(token: str) -> str:
    tok = token.strip(" \t\r\n.,;:!?()[]{}\"'«»").lower()
    if "/" in tok:
        tok = tok.rsplit("/", 1)[-1]
    return tok


def extract_entities(text: str, max_n: int = MAX_ENTITIES) -> list[str]:
    """Return up to ``max_n`` normalized anchors, first-seen order."""
    if not text:
        return []
    text = text[:_SCAN_CAP]
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        tok = _clean(raw)
        if not (_MIN_LEN <= len(tok) <= _MAX_LEN):
            return
        if not any(ch.isalpha() for ch in tok):
            return
        if ":" in tok or "," in tok or "/" in tok:
            return
        if not _ANCHOR_SHAPE.match(tok):
            return
        if _CYRILLIC_PARTICLE.match(tok):
            return
        if tok in _CAP_DENYLIST or tok in seen:
            return
        seen.add(tok)
        ordered.append(tok)

    for m in _BACKTICKED.finditer(text):
        inner = m.group(1)
        # A backticked phrase is an anchor only when it is a single token —
        # code snippets and command lines are not names.
        if " " not in inner.strip():
            _add(inner)
    for m in _HANDLE.finditer(text):
        _add(m.group(1))
    for m in _IDENTIFIER.finditer(text):
        _add(m.group(1))
    for m in _CAMEL.finditer(text):
        _add(m.group(1))
    boundaries = {m.end() for m in _SENTENCE_BOUNDARY.finditer(text)}
    for m in _CAPITALIZED.finditer(text):
        if m.start() in boundaries:
            continue
        _add(m.group(1))

    return ordered[:max_n]


def entity_tags(text: str, max_n: int = MAX_ENTITIES) -> list[str]:
    return [f"entity:{e}" for e in extract_entities(text, max_n=max_n)]
