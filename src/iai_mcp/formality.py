from __future__ import annotations

import logging
import math
import re
import warnings
from typing import Iterable


LEX_MARKERS: dict[str, list[str]] = {
    "en": [
        "therefore", "however", "accordingly", "nonetheless", "furthermore",
        "hence", "thus", "consequently", "moreover", "notwithstanding",
        "whereas", "hereby", "herein", "thereof", "pursuant", "aforementioned",
        "shall", "aforesaid",
    ],
}

HEDGE_MARKERS: dict[str, list[str]] = {
    "en": [
        "possibly", "perhaps", "might", "may", "could", "seemingly",
        "appears to", "seems", "somewhat", "apparently", "presumably",
    ],
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "lex": 0.45,
    "complexity": 0.20,
    "hedge": 0.15,
    "punct": 0.20,
}

_SENTENCE_COMPLEXITY_CENTER: float = 40.0
_SENTENCE_COMPLEXITY_SCALE: float = 25.0
_CLAUSE_COUNT_CENTER: float = 0.5
_CLAUSE_COUNT_SCALE: float = 0.5

_LEX_DENSITY_CENTER: float = 1.5
_LEX_DENSITY_SCALE: float = 1.2
_HEDGE_DENSITY_CENTER: float = 1.0
_HEDGE_DENSITY_SCALE: float = 0.8
_PUNCT_DENSITY_CENTER: float = 1.5
_PUNCT_DENSITY_SCALE: float = 1.3

_NEUTRAL_SCORE: float = 0.5

_logger = logging.getLogger(__name__)


def _tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    return [t.lower() for t in cleaned.split() if t]


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"[.!?;]+", text)
    return [p.strip() for p in parts if p.strip()]


def _sigmoid(x: float) -> float:
    if x >= 0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    ez = math.exp(x)
    return ez / (1.0 + ez)


def _count_phrase_occurrences(text_lower: str, phrases: Iterable[str]) -> int:
    count = 0
    for p in phrases:
        if not p:
            continue
        if " " in p or "-" in p:
            count += text_lower.count(p)
        else:
            count += len(re.findall(rf"\b{re.escape(p)}\b", text_lower, flags=re.UNICODE))
    return count


def _lex_score(text: str, lang: str) -> float:
    markers = LEX_MARKERS.get(lang, [])
    if not markers:
        return _NEUTRAL_SCORE
    toks = _tokens(text)
    if not toks:
        return 0.0
    hits = _count_phrase_occurrences(text.lower(), markers)
    density = hits * 100.0 / max(len(toks), 1)
    return _sigmoid((density - _LEX_DENSITY_CENTER) / _LEX_DENSITY_SCALE)


def _complexity_score(text: str) -> float:
    sents = _sentence_split(text)
    if not sents:
        return 0.0
    avg_len = sum(len(s) for s in sents) / len(sents)
    avg_clauses = sum(s.count(",") for s in sents) / len(sents)
    len_score = _sigmoid(
        (avg_len - _SENTENCE_COMPLEXITY_CENTER) / _SENTENCE_COMPLEXITY_SCALE
    )
    cl_score = _sigmoid((avg_clauses - _CLAUSE_COUNT_CENTER) / _CLAUSE_COUNT_SCALE)
    return 0.5 * len_score + 0.5 * cl_score


def _hedge_score(text: str, lang: str) -> float:
    markers = HEDGE_MARKERS.get(lang, [])
    if not markers:
        return _NEUTRAL_SCORE
    toks = _tokens(text)
    if not toks:
        return 0.0
    hits = _count_phrase_occurrences(text.lower(), markers)
    density = hits * 100.0 / max(len(toks), 1)
    return _sigmoid((density - _HEDGE_DENSITY_CENTER) / _HEDGE_DENSITY_SCALE)


def _punct_score(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    semi = text.count(";")
    em = text.count("—") + text.count("–")
    fq = (
        text.count('"')
        + text.count("“")
        + text.count("”")
        + text.count("«")
        + text.count("»")
    )
    hits = semi + em + fq
    density = hits * 100.0 / max(len(toks), 1)
    return _sigmoid((density - _PUNCT_DENSITY_CENTER) / _PUNCT_DENSITY_SCALE)


def formality_score(
    text: str,
    lang: str,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0

    if lang not in LEX_MARKERS:
        warnings.warn(
            f"formality_score: lang={lang!r} outside the English baseline; "
            "returning neutral 0.5 (graceful degradation for unsupported languages)",
            stacklevel=2,
        )
        _logger.debug("formality_score unknown lang=%s text_len=%d", lang, len(text))
        return _NEUTRAL_SCORE

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    total_w = sum(w.values()) or 1.0

    lex = _lex_score(text, lang)
    complexity = _complexity_score(text)
    hedge = _hedge_score(text, lang)
    punct = _punct_score(text)

    weighted = (
        w["lex"] * lex
        + w["complexity"] * complexity
        + w["hedge"] * hedge
        + w["punct"] * punct
    ) / total_w
    return max(0.0, min(1.0, weighted))
