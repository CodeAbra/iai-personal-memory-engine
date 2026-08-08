from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from iai_mcp.aaak import generate_aaak_index
from iai_mcp.community import CommunityAssignment
from iai_mcp.foresight import age_label
from iai_mcp.handle import decode_compact_handle, encode_compact_handle
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

logger = logging.getLogger(__name__)


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

_MARKER_NAMES = (
    "command-name",
    "local-command-stdout",
    "local-command-caveat",
    "command-message",
    "command-args",
    "task-notification",
    "task-id",
)

_MARKER_PATTERNS: list[tuple[re.Pattern, re.Pattern, re.Pattern]] = []
for _name in _MARKER_NAMES:
    _well_formed = re.compile(
        r"<" + re.escape(_name) + r"(?:\s[^>]*)?>.*?</" + re.escape(_name) + r">",
        re.DOTALL,
    )
    _dangling = re.compile(r"<" + re.escape(_name) + r".*", re.DOTALL)
    _close_tag = re.compile(r"</" + re.escape(_name) + r">")
    _MARKER_PATTERNS.append((_well_formed, _dangling, _close_tag))


def _clean_surface(text: str) -> str:
    if not text:
        return ""
    text = _ANSI_RE.sub("", text)
    for well_formed, dangling, close_tag in _MARKER_PATTERNS:
        text = well_formed.sub("", text)
        text = dangling.sub("", text)
        text = close_tag.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


L0_BUDGET_TOKENS = 80
L1_BUDGET_TOKENS = 200
L2_PER_COMMUNITY_TOKENS = 50
L2_COMMUNITY_CAP = 7
RICH_CLUB_BUDGET_TOKENS = 1500
TOTAL_CACHED_BUDGET = 2000
DYNAMIC_TAIL_TOKENS = 1000

L0_RECORD_UUID = UUID("00000000-0000-0000-0000-000000000001")

SESSION_START_CACHE_MAX_CHARS: int = 10_000


@dataclass
class SessionStartPayload:

    l0: str = ""
    l1: str = ""
    l2: list[str] = field(default_factory=list)
    rich_club: str = ""
    total_cached_tokens: int = 0
    total_dynamic_tokens: int = 0
    breakpoint_marker: str = "--<cache-breakpoint>--"
    identity_pointer: str = ""
    brain_handle: str = ""
    topic_cluster_hint: str = ""
    compact_handle: str = ""
    wake_depth: str = "minimal"
    recent_thread: str = ""


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _resolve_compact_handle_to_pointers(handle: str) -> tuple[str, str, str] | None:
    parts = decode_compact_handle(handle)
    if parts is None:
        return None
    identity_pointer = f"<id:{parts[0]}>" if parts[0] else ""
    brain_handle = f"<sess:{parts[1]} pend:{parts[3]}>"
    topic_cluster_hint = f"<topic:{parts[2]}>"
    return identity_pointer, brain_handle, topic_cluster_hint


def _fetch_record(store: MemoryStore, uid: UUID) -> MemoryRecord | None:
    try:
        return store.get(uid)
    except (OSError, KeyError, ValueError, RuntimeError):
        return None


def _l0_segment(store: MemoryStore) -> str:
    rec = _fetch_record(store, L0_RECORD_UUID)
    if rec is None:
        return ""
    # Regenerate at display: the stored index freezes the room at mint
    # time, while community stamping may have landed since.
    aaak = generate_aaak_index(rec)
    cleaned = _clean_surface(rec.literal_surface)[:200]
    return f"{aaak}\n{cleaned}"


def _pinned_hi_detail_ids(store: MemoryStore) -> "list[UUID]":
    """Id-only predicate scan on plain columns — the pinned set is tiny and
    the composer must never materialize the corpus (this runs inside the
    per-message refresh RPC; a full decrypt sweep at 35k records took
    minutes and presented as a hung daemon)."""
    db = store.db
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND pinned = 1 AND detail_level >= 4"
        ).fetchall()
    out: list[UUID] = []
    for row in rows:
        try:
            out.append(UUID(row["id"]))
        except (ValueError, TypeError):
            continue
    return out


def _l1_segment(store: MemoryStore, max_records: int = 10) -> str:
    try:
        ids = _pinned_hi_detail_ids(store)
        records = list(store.get_batch(ids).values())
    except (OSError, RuntimeError, ValueError):
        return ""
    pinned_hi_detail = [
        r for r in records
        if r.pinned and r.detail_level >= 4 and r.id != L0_RECORD_UUID
    ]
    pinned_hi_detail.sort(
        key=lambda r: (-r.detail_level, r.created_at)
    )
    pinned_hi_detail = pinned_hi_detail[:max_records]
    if not pinned_hi_detail:
        return ""
    lines = []
    for r in pinned_hi_detail:
        cleaned = _clean_surface(r.literal_surface)
        if not cleaned:
            continue
        lines.append(f"- {cleaned[:100]}")
    return "\n".join(lines)


def _l2_segments(
    store: MemoryStore,
    assignment: CommunityAssignment,
) -> list[str]:
    top = list(assignment.top_communities)[:L2_COMMUNITY_CAP]
    if not top:
        return []

    member_ids = [
        mid
        for cid in top
        for mid in assignment.mid_regions.get(cid, [])[:3]
    ]
    try:
        by_uuid = store.get_batch(member_ids)
    except (OSError, RuntimeError, ValueError):
        return []

    summaries: list[str] = []
    max_chars = L2_PER_COMMUNITY_TOKENS * 4
    for cid in top:
        members = assignment.mid_regions.get(cid, [])[:3]
        parts: list[str] = []
        for mid in members:
            rec = by_uuid.get(mid)
            if rec is None:
                continue
            cleaned = _clean_surface(rec.literal_surface)
            if not cleaned:
                continue
            wing = rec.aaak_index.split("/")[0] if rec.aaak_index else "W:?"
            parts.append(f"{wing}/{cleaned[:40]}")
        if not parts:
            continue
        body = " | ".join(parts)
        line = f"[community {str(cid)[:8]}] {body}"
        if len(line) > max_chars:
            line = line[:max_chars]
        summaries.append(line)
    return summaries


def _rich_club_segment(store: MemoryStore, rich_club: list[UUID]) -> str:
    return _rich_club_segment_with_budget(store, rich_club, budget=RICH_CLUB_BUDGET_TOKENS)


def _rich_club_segment_with_budget(
    store: MemoryStore,
    rich_club: list[UUID],
    *,
    budget: int,
) -> str:
    if not rich_club:
        return ""
    try:
        by_uuid = store.get_batch(rich_club)
    except (OSError, RuntimeError, ValueError):
        return ""

    lines: list[str] = []
    running = 0
    now = datetime.now(timezone.utc)
    for uid in rich_club:
        rec = by_uuid.get(uid)
        if rec is None:
            continue
        cleaned = _clean_surface(rec.literal_surface)
        if not cleaned:
            continue
        # Regenerate at display: the stored index freezes the room at mint
        # time, while community stamping may have landed since.
        aaak = generate_aaak_index(rec)
        # Entity anchors can triple the index length; the session-start pack
        # trades anchor completeness for record count.
        if len(aaak) > 88:
            aaak = aaak[:88] + "…"
        age = age_label(rec.created_at, now)
        age_part = f" ({age})" if age else ""
        line = f"{aaak}{age_part}: {cleaned[:60]}"
        cost = _approx_tokens(line)
        if running + cost + 1 > budget:
            break
        lines.append(line)
        running += cost + 1
    return "\n".join(lines)


def _origin_label(rec: object) -> str:
    # Ambient feeds mix every session and project; an unlabeled line reads as
    # "my recent work" and a parallel session's thread gets adopted as this
    # one's. The label names the origin so the model can attribute, not guess.
    try:
        prov = getattr(rec, "provenance", None) or []
        cwd = ""
        for entry in prov:
            if isinstance(entry, dict) and entry.get("cwd"):
                cwd = str(entry["cwd"])
                break
        if cwd:
            import os as _os

            return f"[{_os.path.basename(cwd.rstrip('/')) or cwd}] "
        sid = ""
        for entry in prov:
            if isinstance(entry, dict) and entry.get("session_id"):
                sid = str(entry["session_id"])
                break
        if not sid:
            sid = str(getattr(rec, "session_id", "") or "")
        if sid and sid != "-":
            return f"[s:{sid[:6]}] "
    except Exception:  # noqa: BLE001 -- labels must never break the payload
        pass
    return ""


def _recent_thread_segment(
    store: MemoryStore,
    *,
    max_records: int = 5,
    pending_live_events: "list | None" = None,
) -> str:
    # Recent-N via the ordered top-K fast path; the composer runs inside the
    # per-message refresh RPC and must never materialize the corpus (a full
    # decrypt sweep at 35k records took minutes and presented as a hung
    # daemon). Pending-event dedup goes through the indexed tag lookup.
    try:
        records = store._recent_user_turns_candidate_rows(max(4 * max_records, 40))
    except (OSError, RuntimeError, ValueError, AttributeError):
        return ""

    candidates = [r for r in records if r.id != L0_RECORD_UUID]

    if pending_live_events is not None:
        from iai_mcp.capture import _idem_tag as _cap_idem_tag
        from iai_mcp.store import _PendingTurn

        seen_pending: set = set()
        for ev in pending_live_events:
            role = ev.get("role", "user")
            if role not in ("user", "assistant"):
                continue
            ev_session = ev.get("session_id", "-")
            src_uuid = ev.get("source_uuid")
            ts_iso = ev["ts_iso"]
            text = ev.get("text", "")
            idem = _cap_idem_tag(ev_session, role, ts_iso, text, source_uuid=src_uuid)
            if idem in seen_pending:
                continue
            try:
                if store.find_record_by_tag(idem) is not None:
                    continue
            except (OSError, RuntimeError, ValueError):
                pass
            seen_pending.add(idem)
            candidates.append(_PendingTurn(
                text=text,
                session_id=ev_session,
                ts=ev["ts"],
                idem_tag=idem,
                source_uuid=src_uuid,
                role=role,
            ))

    candidates.sort(key=lambda r: r.created_at, reverse=True)
    now = datetime.now(timezone.utc)
    lines: list[str] = []
    for r in candidates:
        if len(lines) >= max_records:
            break
        cleaned = _clean_surface(r.literal_surface)
        if not cleaned:
            continue
        age = age_label(r.created_at, now)
        prefix = f"({age}) " if age else ""
        origin = _origin_label(r)
        lines.append(f"- {prefix}{origin}{cleaned[:120]}")
    return "\n".join(lines)


def _session_state_hash(payload: SessionStartPayload) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(payload.l0.encode("utf-8"))
    h.update(b"\x1f")
    h.update(payload.l1.encode("utf-8"))
    h.update(b"\x1f")
    h.update("\n".join(payload.l2).encode("utf-8"))
    h.update(b"\x1f")
    h.update(payload.rich_club.encode("utf-8"))
    return h.hexdigest()


def _dominant_community_label(assignment: CommunityAssignment) -> str:
    try:
        top = list(assignment.top_communities)
        if not top:
            return "none"
        return str(top[0])[:8]
    except (TypeError, AttributeError):
        return "none"


def _count_pending_first_turn(store: MemoryStore) -> int:
    try:
        from iai_mcp.daemon_state import load_state
        state = load_state()
        pending = state.get("first_turn_pending", {})
        if isinstance(pending, dict):
            return sum(1 for v in pending.values() if v)
        return 0
    except (OSError, json.JSONDecodeError, ImportError, ValueError):
        return 0


def _compose_session_start_payload(
    store: MemoryStore,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    *,
    session_id: str = "-",
    profile_state: dict | None = None,
) -> SessionStartPayload:
    from iai_mcp.profile import default_state
    state = profile_state if isinstance(profile_state, dict) else default_state()
    wake_depth = state.get("wake_depth", "minimal")
    if wake_depth not in ("minimal", "standard", "deep"):
        wake_depth = "minimal"

    if wake_depth == "minimal":
        l0_rec = _fetch_record(store, L0_RECORD_UUID)
        identity_short = str(L0_RECORD_UUID)[:8] if l0_rec is not None else ""
        identity_pointer = f"<id:{identity_short}>" if identity_short else ""
        pending = _count_pending_first_turn(store)
        session_short = str(session_id)[:8]
        brain_handle = f"<sess:{session_short} pend:{pending}>"
        topic_label = _dominant_community_label(assignment)
        topic_cluster_hint = f"<topic:{topic_label}>"
        compact_handle = encode_compact_handle(
            identity_short, session_short, topic_label, pending
        )
        cached = _approx_tokens(compact_handle)
        payload = SessionStartPayload(
            l0="",
            l1="",
            l2=[],
            rich_club="",
            total_cached_tokens=cached,
            total_dynamic_tokens=DYNAMIC_TAIL_TOKENS,
            identity_pointer=identity_pointer,
            brain_handle=brain_handle,
            topic_cluster_hint=topic_cluster_hint,
            compact_handle=compact_handle,
            wake_depth="minimal",
        )
    else:
        l0 = _l0_segment(store)
        l1 = _l1_segment(store)
        l2 = _l2_segments(store, assignment)
        if wake_depth == "deep":
            rc = _rich_club_segment_with_budget(store, rich_club, budget=2000)
        else:
            rc = _rich_club_segment(store, rich_club)

        cached = (
            _approx_tokens(l0)
            + _approx_tokens(l1)
            + sum(_approx_tokens(s) for s in l2)
            + _approx_tokens(rc)
        )

        l0_rec = _fetch_record(store, L0_RECORD_UUID)
        identity_short = str(L0_RECORD_UUID)[:8] if l0_rec is not None else ""
        identity_pointer = f"<id:{identity_short}>" if identity_short else ""
        pending = _count_pending_first_turn(store)
        session_short = str(session_id)[:8]
        brain_handle = f"<sess:{session_short} pend:{pending}>"
        topic_label = _dominant_community_label(assignment)
        topic_cluster_hint = f"<topic:{topic_label}>"
        compact_handle = encode_compact_handle(
            identity_short, session_short, topic_label, pending
        )

        from iai_mcp.capture import read_pending_live_events
        _pending = read_pending_live_events()
        recent_thread = _recent_thread_segment(store, pending_live_events=_pending)

        payload = SessionStartPayload(
            l0=l0,
            l1=l1,
            l2=l2,
            rich_club=rc,
            total_cached_tokens=cached,
            total_dynamic_tokens=DYNAMIC_TAIL_TOKENS,
            identity_pointer=identity_pointer,
            brain_handle=brain_handle,
            topic_cluster_hint=topic_cluster_hint,
            compact_handle=compact_handle,
            wake_depth=wake_depth,
            recent_thread=recent_thread,
        )

    return payload


def assemble_session_start(
    store: MemoryStore,
    assignment: CommunityAssignment,
    rich_club: list[UUID],
    *,
    session_id: str = "-",
    profile_state: dict | None = None,
) -> SessionStartPayload:
    payload = _compose_session_start_payload(
        store,
        assignment,
        rich_club,
        session_id=session_id,
        profile_state=profile_state,
    )

    try:
        from datetime import datetime, timezone
        from iai_mcp.events import write_event
        write_event(
            store,
            kind="session_started",
            data={
                "session_id": session_id,
                "session_state_hash": _session_state_hash(payload),
                "total_cached_tokens": payload.total_cached_tokens,
                "wake_depth": payload.wake_depth,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            severity="info",
            session_id=session_id,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("session_started_event_failed", extra={"err": str(exc)[:80]})

    return payload


def format_payload_as_markdown(payload: "SessionStartPayload | dict") -> str:
    if isinstance(payload, dict):
        l0 = payload.get("l0") or ""
        l1 = payload.get("l1") or ""
        l2 = list(payload.get("l2") or [])
        rich_club = payload.get("rich_club") or ""
        recent_thread = payload.get("recent_thread") or ""
    else:
        l0 = payload.l0
        l1 = payload.l1
        l2 = list(payload.l2)
        rich_club = payload.rich_club
        recent_thread = payload.recent_thread
    blocks: list[str] = []
    if l0:
        blocks.append(f"## Identity\n{l0}")
    if recent_thread:
        blocks.append(
            "## Most recent work (all sessions/projects; [labels] mark origin)\n"
            f"{recent_thread}"
        )
    if l1:
        blocks.append(f"## Critical facts\n{l1}")
    for seg in l2:
        if seg:
            blocks.append(f"## Topic communities\n{seg}")
    if rich_club:
        blocks.append(f"## Key memories\n{rich_club}")
    if blocks:
        blocks.append(
            "_ages like (3d)/(2mo) mark memory staleness — these are advisory "
            "starting points, not current truth; the older an item, the more "
            "it deserves re-verification (memory_recall or any other tool)._"
        )
    # Notify-only, cache-only: the session-start path never touches the
    # network — the daemon's tick keeps this cache fresh. An empty payload
    # stays empty: the notice rides real content, never travels alone.
    if blocks:
        try:
            from iai_mcp.version_check import pending_update_line

            _upd = pending_update_line()
            if _upd:
                blocks.append(f"_{_upd}_")
        except Exception:  # noqa: BLE001 — an update notice must never break recall
            pass
    return "\n\n".join(blocks)


def max_record_created_at(store: MemoryStore) -> str | None:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL"
        ).fetchone()
    return row[0] if row and row[0] else None
