from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from iai_mcp.quiet_window import BUCKET_MINUTES


WIND_DOWN_GATE_MINUTES_BEFORE: int = 30


WIND_DOWN_EN: list[str] = [
    r"\bgood\s*night\b",
    r"\bgoodnight\b",
    r"\bnight[,!.]?\s*$",
    r"\bI'?m\s+(heading|going)\s+to\s+bed\b",
    r"\b(time\s+(to|for)\s+bed|bedtime)\b",
    r"\bI'?m\s+(tired|exhausted|sleepy)\b",
    r"\b(catch\s+you\s+tomorrow|see\s+you\s+tomorrow)\b",
    r"\blet'?s\s+(continue|pick\s+up)\s+tomorrow\b",
    r"\bgoing\s+to\s+sleep\b",
]

WIND_DOWN_RU: list[str] = [
    r"спокойной\s+ночи",
    r"пойду\s+(спать|в\s+постель)",
    r"(я\s+)?(устал|устала|вымотан[аы]?|засыпаю)",
    r"пора\s+(спать|ложиться)",
    r"до\s+завтра",
    r"давай\s+завтра",
    r"ухожу\s+спать",
    r"(окей|ок|ладно),?\s+сплю",
    r"ложусь",
]

WIND_DOWN_JA: list[str] = [
    r"お\s*や\s*す\s*み(なさい)?",
    r"寝\s*ます",
    r"(眠|ねむ)い",
    r"(寝る|ねる)(ね|よ|わ)?",
    r"また\s*(明日|あした)",
    r"(疲|つか)れた",
    r"ベッド\s*に\s*(入る|はいる)",
]

WIND_DOWN_AR: list[str] = [
    r"تصبح\s+على\s+خير",
    r"ليلة\s+سعيدة",
    r"أنا\s+(ذاهب|ذاهبة)\s+(للنوم|إلى\s+النوم)",
    r"أنا\s+(متعب|متعبة|تعبان[ةه]?)",
    r"سأنام",
    r"وقت\s+النوم",
    r"إلى\s+(الغد|اللقاء\s+غدا)",
]

WIND_DOWN_DE: list[str] = [
    r"\bgute\s+nacht\b",
    r"\bgn8\b",
    r"\bich\s+gehe\s+(jetzt\s+)?(ins\s+bett|schlafen)\b",
    r"\b(ich\s+bin\s+)?(müde|kaputt|fertig)\b",
    r"\bschlafenszeit\b",
    r"\bbis\s+morgen\b",
    r"\blass\s+uns\s+morgen\s+weitermachen\b",
]

WIND_DOWN_FR: list[str] = [
    r"\bbonne\s+nuit\b",
    r"\bje\s+(vais|pars)\s+(me\s+coucher|dormir)\b",
    r"\b(je\s+suis\s+)?(fatigu[ée]|[ée]puis[ée])\b",
    r"\b(il\s+est\s+)?l'?heure\s+de\s+(dormir|me\s+coucher)\b",
    r"\b[aà]\s+demain\b",
    r"\bon\s+reprend\s+demain\b",
]

WIND_DOWN_ES: list[str] = [
    r"\bbuenas\s+noches\b",
    r"\bme\s+voy\s+a\s+(dormir|la\s+cama|descansar)\b",
    r"\b(estoy\s+)?(cansad[oa]|agotad[oa])\b",
    r"\bhora\s+de\s+dormir\b",
    r"\bhasta\s+ma[ñn]ana\b",
    r"\bseguimos\s+ma[ñn]ana\b",
]

WIND_DOWN_ZH: list[str] = [
    r"晚\s*安",
    r"我\s*(要|去)\s*睡\s*(觉|了)",
    r"累\s*了",
    r"(该|到)\s*睡\s*(觉)?\s*了",
    r"明\s*天\s*见",
    r"明\s*天\s*继\s*续",
]

WIND_DOWN_BY_LANG: dict[str, list[str]] = {
    "en": WIND_DOWN_EN,
    "ru": WIND_DOWN_RU,
    "ja": WIND_DOWN_JA,
    "ar": WIND_DOWN_AR,
    "de": WIND_DOWN_DE,
    "fr": WIND_DOWN_FR,
    "es": WIND_DOWN_ES,
    "zh": WIND_DOWN_ZH,
}

_COMPILED: dict[str, list[re.Pattern]] = {
    lang: [re.compile(p, re.IGNORECASE) for p in pats]
    for lang, pats in WIND_DOWN_BY_LANG.items()
}

WIND_DOWN_LANGUAGES_SUPPORTED: frozenset[str] = frozenset(WIND_DOWN_BY_LANG.keys())


def detect_wind_down_phrase(cue: str, language: str) -> Tuple[bool, str]:
    if not cue:
        return False, ""

    for p in _COMPILED.get(language or "", []):
        if p.search(cue):
            return True, p.pattern

    if language != "en":
        for p in _COMPILED["en"]:
            if p.search(cue):
                return True, p.pattern

    return False, ""


def is_late_in_quiet_window(
    window: Optional[Tuple[int, int]],
    now: datetime,
    tz: ZoneInfo,
) -> bool:
    if not window:
        return False

    start_bucket, duration = window
    try:
        now_local = now.astimezone(tz)
    except (TypeError, ValueError, OverflowError):
        return False

    cur_minutes = now_local.hour * 60 + now_local.minute
    start_minutes = start_bucket * BUCKET_MINUTES
    end_minutes = (start_bucket + duration) * BUCKET_MINUTES

    if end_minutes > 24 * 60:
        wrapped_end = end_minutes - 24 * 60
        inside = cur_minutes >= start_minutes or cur_minutes < wrapped_end
    else:
        inside = start_minutes <= cur_minutes < end_minutes

    if inside:
        return True

    minutes_until_start = (start_minutes - cur_minutes) % (24 * 60)
    return 0 <= minutes_until_start <= WIND_DOWN_GATE_MINUTES_BEFORE


def detect_wind_down(
    cue: str,
    language: str,
    state: dict,
    now: datetime,
    tz: ZoneInfo,
) -> Optional[dict]:
    matched, pattern = detect_wind_down_phrase(cue, language)
    if not matched:
        return None

    from iai_mcp.quiet_window import effective_consolidation_window

    # Same resolver as the consolidation gate: a reset or unlearned window
    # must fall back to the night default here too, never go dark.
    window = effective_consolidation_window(
        state.get("quiet_window") if isinstance(state, dict) else None,
        manual=(
            state.get("quiet_window_manual_override")
            if isinstance(state, dict)
            else None
        ),
    )
    if not is_late_in_quiet_window(window, now, tz):
        return None

    start_bucket, duration = window
    return {
        "message_hint": "user_wind_down_detected",
        "matched_pattern": pattern,
        "quiet_window_start_bucket": int(start_bucket),
        "quiet_window_duration": int(duration),
    }


__all__ = [
    "WIND_DOWN_AR",
    "WIND_DOWN_BY_LANG",
    "WIND_DOWN_DE",
    "WIND_DOWN_EN",
    "WIND_DOWN_ES",
    "WIND_DOWN_FR",
    "WIND_DOWN_GATE_MINUTES_BEFORE",
    "WIND_DOWN_JA",
    "WIND_DOWN_LANGUAGES_SUPPORTED",
    "WIND_DOWN_RU",
    "WIND_DOWN_ZH",
    "detect_wind_down",
    "detect_wind_down_phrase",
    "is_late_in_quiet_window",
]
