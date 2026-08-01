from __future__ import annotations

from datetime import datetime

from iai_mcp.events import flush_event_buffer, query_events, write_event
from iai_mcp.store import MemoryStore


METRIC_NAMES: list[str] = ["m1", "m2", "m3", "m4", "m5", "m6"]


def record_session_metrics(
    store: MemoryStore,
    session_id: str,
    metrics: dict[str, float],
) -> None:
    for m, v in metrics.items():
        if m not in METRIC_NAMES:
            continue
        try:
            value = float(v)
        except (TypeError, ValueError):
            continue
        write_event(
            store,
            kind="trajectory_metric",
            data={"metric": m, "value": value},
            severity="info",
            session_id=session_id,
        )


def aggregate_trajectory(
    store: MemoryStore,
    since: datetime | None = None,
) -> dict[str, list[tuple[datetime, float]]]:
    events = query_events(
        store, kind="trajectory_metric", since=since, limit=10000,
    )
    out: dict[str, list[tuple[datetime, float]]] = {m: [] for m in METRIC_NAMES}
    for e in events:
        m = e["data"].get("metric")
        v = e["data"].get("value")
        if m in METRIC_NAMES and v is not None:
            try:
                out[m].append((e["ts"], float(v)))
            except (TypeError, ValueError):
                continue
    return out


def compute_m1_clarifying_questions_per_session(
    store: MemoryStore,
    session_id: str,
) -> float:
    events = query_events(store, kind="curiosity_question", limit=1000)
    count = sum(1 for e in events if e.get("session_id") == session_id)
    return float(count)


def compute_m3_token_budget(
    store: MemoryStore,
    session_id: str,
) -> float:
    # assemble_session_start stamps every live serve with session_started
    # carrying total_cached_tokens — the instrument for the <=3000-token
    # session-start budget.
    events = query_events(store, kind="session_started", limit=100)
    session_events = [e for e in events if e.get("session_id") == session_id]
    if not session_events:
        return 0.0
    total = 0.0
    for e in session_events:
        try:
            total += float(e["data"].get("total_cached_tokens", 0))
        except (TypeError, ValueError):
            continue
    return total / len(session_events)


def compute_m5_curiosity_frequency(
    store: MemoryStore,
    session_id: str,
) -> float:
    silent = query_events(store, kind="curiosity_silent_log", limit=1000)
    questions = query_events(store, kind="curiosity_question", limit=1000)
    total = 0
    for ev_list in (silent, questions):
        total += sum(1 for e in ev_list if e.get("session_id") == session_id)
    return float(total)


def compute_session_metrics_snapshot(
    store: MemoryStore,
    session_id: str,
) -> dict[str, float]:
    return {
        "m1": compute_m1_clarifying_questions_per_session(store, session_id),
        "m2": m2_precision_at_5_live(store),
        "m3": compute_m3_token_budget(store, session_id),
        "m4": m4_profile_variance_live(store),
        "m5": compute_m5_curiosity_frequency(store, session_id),
        "m6": m6_context_repeat_rate_live(store),
    }


M2_SYNTHETIC_CONSTANT: float = 0.0
M4_SYNTHETIC_CONSTANT: float = 0.0
M6_SYNTHETIC_CONSTANT: float = 0.0


def m2_precision_at_5_live(
    store: MemoryStore,
    *,
    window: int = 100,
) -> float:
    flush_event_buffer(store)
    events = query_events(store, kind="retrieval_used", limit=window)
    if not events:
        return 0.0

    precisions: list[float] = []
    for ev in events:
        data = ev.get("data") or {}
        hits = data.get("hit_ids") or []
        ground_truth = set(data.get("ground_truth") or [])
        top5 = list(hits)[:5]
        if ground_truth:
            tp = sum(1 for h in top5 if h in ground_truth)
            precisions.append(tp / 5.0)
        else:
            precisions.append(1.0 if top5 else 0.0)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def m4_profile_variance_live(
    store: MemoryStore,
    *,
    n_updates: int = 20,
) -> float:
    events = query_events(store, kind="profile_updated", limit=n_updates * 5)
    if not events:
        return 0.0

    per_knob: dict[str, list[float]] = {}
    for ev in events[:n_updates]:
        data = ev.get("data") or {}
        knob = data.get("knob")
        new_val = data.get("new")
        if knob is None or new_val is None:
            continue
        if isinstance(new_val, bool) or not isinstance(new_val, (int, float)):
            continue
        per_knob.setdefault(str(knob), []).append(float(new_val))

    if not per_knob:
        return 0.0

    variances: list[float] = []
    for _knob, vals in per_knob.items():
        if len(vals) < 2:
            variances.append(0.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        variances.append(var)
    if not variances:
        return 0.0
    return sum(variances) / len(variances)


def m6_context_repeat_rate_live(
    store: MemoryStore,
    *,
    window_days: int = 30,
) -> float:
    from datetime import datetime, timedelta, timezone
    since = datetime.now(timezone.utc) - timedelta(days=window_days)
    events = query_events(
        store, kind="session_started", since=since, limit=10000,
    )
    if not events:
        return 0.0

    hashes: list[str] = []
    for ev in events:
        data = ev.get("data") or {}
        hsh = data.get("session_state_hash")
        if hsh:
            hashes.append(str(hsh))
    if not hashes:
        return 0.0
    total = len(hashes)
    unique = len(set(hashes))
    return (total - unique) / total


def m2(store: MemoryStore) -> float:
    return m2_precision_at_5_live(store)


def m4(store: MemoryStore) -> float:
    return m4_profile_variance_live(store)


def m6(store: MemoryStore) -> float:
    return m6_context_repeat_rate_live(store)
