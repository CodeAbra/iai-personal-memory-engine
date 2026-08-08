from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore

BUCKET_COUNT = 48
BUCKET_MINUTES = 30

MIN_WINDOW_HOURS = 3
MAX_WINDOW_HOURS = 8

MIN_DAYS_FOR_LEARN = 7
BOOTSTRAP_IDLE_HOURS = 2

#: Rolling depth of the per-day presence masks the daemon stamps each tick.
PRESENCE_KEEP_DAYS = 14
#: A bucket counts as "busy" for the presence learner when it was busy on at
#: least this fraction of the observed days.
PRESENCE_BUSY_DAY_FRACTION = 0.3

WIND_DOWN_GATE_MINUTES_BEFORE = 30
DIGEST_SHOW_THRESHOLD_HOURS = 18


def learn_quiet_window(
    store: MemoryStore,
    now: datetime,
    tz: ZoneInfo,
) -> Optional[tuple[int, int]]:
    since = now - timedelta(days=MIN_DAYS_FOR_LEARN)
    events = query_events(store, kind="session_started", since=since, limit=10000)
    if not events:
        return None

    counts = [0] * BUCKET_COUNT
    days_seen: set[tuple[int, int, int]] = set()
    for e in events:
        ts = e["ts"]
        if not isinstance(ts, datetime):
            try:
                ts = ts.to_pydatetime()
            except (AttributeError, TypeError, ValueError):
                continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        try:
            ts_local = ts.astimezone(tz)
        except (TypeError, ValueError, OverflowError):
            continue
        bucket = (ts_local.hour * 60 + ts_local.minute) // BUCKET_MINUTES
        if 0 <= bucket < BUCKET_COUNT:
            counts[bucket] += 1
        days_seen.add((ts_local.year, ts_local.month, ts_local.day))

    if len(days_seen) < MIN_DAYS_FOR_LEARN:
        return None

    peak = max(counts)
    if peak == 0:
        return None
    threshold = max(1, int(peak * 0.2))
    return _window_from_counts(counts, threshold)


def _window_from_counts(
    counts: list[int], threshold: int
) -> Optional[tuple[int, int]]:
    doubled = counts + counts
    best_start, best_len = 0, 0
    cur_start, cur_len = None, 0
    for i, c in enumerate(doubled):
        if c < threshold:
            if cur_start is None:
                cur_start = i
                cur_len = 1
            else:
                cur_len += 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_start, cur_len = None, 0

    min_buckets = MIN_WINDOW_HOURS * (60 // BUCKET_MINUTES)
    max_buckets = MAX_WINDOW_HOURS * (60 // BUCKET_MINUTES)
    if best_len < min_buckets:
        return None
    duration = min(best_len, max_buckets)
    if duration > BUCKET_COUNT:
        duration = BUCKET_COUNT
    return (best_start % BUCKET_COUNT, duration)


def local_bucket(ts_local: datetime) -> int:
    return ((ts_local.hour * 60 + ts_local.minute) // BUCKET_MINUTES) % BUCKET_COUNT


def stamp_presence_mask(
    day_masks: dict, ts_local: datetime, *, busy: bool
) -> bool:
    """Set the busy bit for ts_local's bucket. Returns True when the mask
    changed (the caller persists only then — at most 48 writes a day)."""
    if not busy:
        return False
    key = ts_local.strftime("%Y-%m-%d")
    bit = 1 << local_bucket(ts_local)
    try:
        current = int(day_masks.get(key, 0))
    except (TypeError, ValueError):
        current = 0
    if current & bit:
        return False
    day_masks[key] = current | bit
    return True


def prune_presence_masks(
    day_masks: dict, today_local: datetime, *, keep_days: int = PRESENCE_KEEP_DAYS
) -> bool:
    cutoff = (today_local - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    stale = [k for k in day_masks if not isinstance(k, str) or k < cutoff]
    for k in stale:
        day_masks.pop(k, None)
    return bool(stale)


def learn_quiet_window_from_presence(
    day_masks: "dict | None",
) -> Optional[tuple[int, int]]:
    """Learn the quiet window from per-day presence masks.

    A bucket counts as busy when it was active on at least
    PRESENCE_BUSY_DAY_FRACTION of the observed days; the window is the
    longest stretch of non-busy buckets, clamped to MIN/MAX_WINDOW_HOURS.
    """
    if not isinstance(day_masks, dict) or len(day_masks) < MIN_DAYS_FOR_LEARN:
        return None
    masks: list[int] = []
    for v in day_masks.values():
        try:
            masks.append(int(v))
        except (TypeError, ValueError):
            continue
    if len(masks) < MIN_DAYS_FOR_LEARN:
        return None
    counts = [0] * BUCKET_COUNT
    for mask in masks:
        for b in range(BUCKET_COUNT):
            if mask & (1 << b):
                counts[b] += 1
    threshold = max(1, int(round(PRESENCE_BUSY_DAY_FRACTION * len(masks))))
    return _window_from_counts(counts, threshold)


def should_relearn(last_learned_at: Optional[datetime], now: datetime) -> bool:
    if last_learned_at is None:
        return True
    if last_learned_at.tzinfo is None:
        last_learned_at = last_learned_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - last_learned_at) >= timedelta(hours=24)


def should_bootstrap_trigger(last_session_ts: Optional[datetime], now: datetime) -> bool:
    if last_session_ts is None:
        return True
    if last_session_ts.tzinfo is None:
        last_session_ts = last_session_ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - last_session_ts) >= timedelta(hours=BOOTSTRAP_IDLE_HOURS)


DEFAULT_WINDOW_START_BUCKET = 4   # 2h past local midnight
DEFAULT_WINDOW_BUCKETS = 8        # 4 hours


def parse_window_spec(raw: "str | None") -> Optional[tuple[int, int]]:
    """Parse an "HH:MM-HH:MM" local-time window into (start_bucket, buckets)."""
    if not raw:
        return None
    try:
        start_s, end_s = raw.strip().split("-")
        sh, sm = (int(x) for x in start_s.split(":"))
        eh, em = (int(x) for x in end_s.split(":"))
        if not (0 <= sh <= 23 and 0 <= eh <= 23 and 0 <= sm <= 59 and 0 <= em <= 59):
            return None
        start = (sh * 60 + sm) // BUCKET_MINUTES
        end = (eh * 60 + em) // BUCKET_MINUTES
        buckets = (end - start) % BUCKET_COUNT
        if buckets == 0:
            return None
        return (start % BUCKET_COUNT, buckets)
    except (ValueError, AttributeError):
        return None


def effective_consolidation_window(
    learned: "tuple[int, int] | list | None",
    manual: "list | tuple | None" = None,
) -> tuple[int, int]:
    """Resolve the active consolidation window.

    Precedence: env override -> operator manual override (the
    ``set-quiet-window`` configure verb, stored as ["HH:MM", "HH:MM"]) ->
    the learned quiet window -> the fixed night default. There is ALWAYS
    a window: consolidation never runs around the clock.
    """
    import os

    env = parse_window_spec(os.environ.get("IAI_MCP_CONSOLIDATION_WINDOW"))
    if env is not None:
        return env
    if isinstance(manual, (list, tuple)) and len(manual) == 2:
        parsed = parse_window_spec(f"{manual[0]}-{manual[1]}")
        if parsed is not None:
            return parsed
    if (
        isinstance(learned, (tuple, list))
        and len(learned) == 2
        and all(isinstance(x, int) for x in learned)
    ):
        return (learned[0] % BUCKET_COUNT, max(1, min(int(learned[1]), BUCKET_COUNT)))
    return (DEFAULT_WINDOW_START_BUCKET, DEFAULT_WINDOW_BUCKETS)


def within_window(
    window: "tuple[int, int]", now: datetime, tz: ZoneInfo,
) -> bool:
    start, buckets = window
    try:
        local = now.astimezone(tz)
    except (TypeError, ValueError, OverflowError):
        return False
    bucket = (local.hour * 60 + local.minute) // BUCKET_MINUTES
    return ((bucket - start) % BUCKET_COUNT) < buckets
