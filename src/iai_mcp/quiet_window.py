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
) -> tuple[int, int]:
    """Resolve the active consolidation window.

    Precedence: explicit env override -> the learned quiet window (the system
    adapts to the user's observed rhythm) -> the fixed night default. There is
    ALWAYS a window: consolidation never runs around the clock.
    """
    import os

    env = parse_window_spec(os.environ.get("IAI_MCP_CONSOLIDATION_WINDOW"))
    if env is not None:
        return env
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
