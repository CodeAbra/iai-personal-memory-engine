from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from iai_mcp.aaak import _TIER_TO_WING, generate_aaak_index
from iai_mcp.community import CommunityAssignment
from iai_mcp.directive_budget import (
    AGENT_REGISTRY_BUDGET_TOKENS,
    AGENT_REGISTRY_LINE_CHAR_CAP,
    AGENT_REGISTRY_MAX_RENDERED,
    CONT_BUDGET_TOKENS,
    DIRECTIVE_BUDGET_TOKENS,
    DIRECTIVE_LINE_CHAR_CAP,
)
from iai_mcp.foresight import age_label
from iai_mcp.handle import decode_compact_handle, encode_compact_handle
from iai_mcp.store import MemoryStore
from iai_mcp.types import CLS_SUMMARY_PREFIX_RE, MemoryRecord

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
    "iai-mcp-directives",
    "iai-mcp-live-state",
    "iai-mcp-agent-registry",
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
# Selection is always keyed to the legacy (verbose) line cost, never to the
# compact-label render cost -- raising this over-admits, lowering it drops
# records. IAI_MCP_RICH_CLUB_COMPACT_LABEL only swaps which label gets
# EMITTED for an already-selected record; it never changes the budget.
RICH_CLUB_BUDGET_TOKENS = 1500
RICH_CLUB_CLS_SUMMARY_CAP = 30
_RICH_CLUB_DEEP_BUDGET_TOKENS = 2000
TOTAL_CACHED_BUDGET = 2000
DYNAMIC_TAIL_TOKENS = 1000

L0_RECORD_UUID = UUID("00000000-0000-0000-0000-000000000001")

SESSION_START_CACHE_MAX_CHARS: int = 10_000

# Per-turn refresh delta contract: the renderer must fit inside this ceiling
# instead of re-injecting the full session-start brief on every advance.
DELTA_MAX_TOKENS: int = 500
K_DELTA: int = 12


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
    directives: str = ""
    live_state: str = ""


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


def _compact_label_enabled() -> bool:
    return os.environ.get("IAI_MCP_RICH_CLUB_COMPACT_LABEL", "1") != "0"


_COMPACT_LABEL_ENTITY_CAP = 4
_COMPACT_LABEL_ENTITY_CHAR_CAP = 40


def _compact_aaak_label(tier: str, tags: list[str]) -> str:
    # Render-site transform built directly from record fields -- never from
    # a generate-then-reparse round trip through generate_aaak_index, so a
    # tag/entity value containing '/' can't mis-split into a spoofed field.
    wing = _TIER_TO_WING.get(tier, "?")
    # Capped by BOTH count and character length: a handful of long entity
    # names can grow the joined string past the shared 88-char aaak
    # truncation guard just as easily as too many short ones, at which
    # point the compact form no longer saves anything over the legacy one.
    all_entities = [t[len("entity:"):] for t in tags if t.startswith("entity:")]
    entities = all_entities[:_COMPACT_LABEL_ENTITY_CAP]
    if not entities:
        return wing
    joined = ",".join(entities)
    count_truncated = len(all_entities) > _COMPACT_LABEL_ENTITY_CAP
    char_truncated = len(joined) > _COMPACT_LABEL_ENTITY_CHAR_CAP
    if char_truncated:
        joined = joined[:_COMPACT_LABEL_ENTITY_CHAR_CAP]
    marker = "…" if (count_truncated or char_truncated) else ""
    return f"{wing} ·{joined}{marker}"


def _display_aaak(rec: MemoryRecord) -> str:
    if _compact_label_enabled():
        return _compact_aaak_label(rec.tier, rec.tags)
    return generate_aaak_index(rec)


def _l0_segment(store: MemoryStore) -> str:
    rec = _fetch_record(store, L0_RECORD_UUID)
    if rec is None:
        return ""
    # Regenerate at display: the stored index freezes the room at mint
    # time, while community stamping may have landed since.
    aaak = _display_aaak(rec)
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


def _live_directive_ids(store: MemoryStore) -> "list[UUID]":
    """Plain-column scan, no embedding/cue/rank/similarity gate -- directives
    must survive every wake_depth including minimal."""
    db = store.db
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND directive = 1 ORDER BY created_at"
        ).fetchall()
    out: list[UUID] = []
    for row in rows:
        try:
            out.append(UUID(row["id"]))
        except (ValueError, TypeError):
            continue
    return out


def render_directive_segment(store: MemoryStore) -> str:
    """Renders the full accepted set with no truncation -- the capture-time
    budget gate already bounds it; a cut here would silently drop an order."""
    if os.environ.get("IAI_MCP_DIRECTIVES_OFF") == "1":
        return ""
    try:
        ids = _live_directive_ids(store)
        records = list(store.get_batch(ids).values())
    except (OSError, RuntimeError, ValueError):
        return ""
    records.sort(key=lambda r: r.created_at)
    lines: list[str] = []
    for r in records:
        cleaned = _clean_surface(r.literal_surface)
        if not cleaned:
            continue
        lines.append(f"- {cleaned[:DIRECTIVE_LINE_CHAR_CAP]}")
    rendered = "\n".join(lines)
    if _approx_tokens(rendered) > DIRECTIVE_BUDGET_TOKENS:
        logger.warning(
            "directive_segment_over_budget",
            extra={"tokens": _approx_tokens(rendered), "budget": DIRECTIVE_BUDGET_TOKENS},
        )
    return rendered


def render_live_state_segment(*, fold_sensory: bool = True) -> str:
    """Narrow session-continuity render from the process-global focal task:
    goal + focus + next_action only, never subgoals/hypotheses/results/raw
    sensory. Reads the GLOBAL focal task (no session_id) -- the composer's
    session_id is a placeholder on the daemon's cached fast path and must
    never scope this read. No store/embedding/cue/rank/similarity gate.

    fold_sensory is threaded straight to working_tier.read_task -- this
    render itself reads only goal/focus/next_action either way, so
    fold_sensory=False changes no output, only whether the read holds
    read_task's lock across the (unused here) sensory-tail disk fold.
    """
    from iai_mcp import working_tier

    entry = working_tier.read_task(fold_sensory=fold_sensory)
    if entry is None:
        return ""
    lines: list[str] = []
    goal = _clean_surface(entry.goal)
    if goal:
        lines.append(f"goal: {goal}")
    focus = _clean_surface(entry.focus)
    if focus:
        lines.append(f"focus: {focus}")
    next_action = _clean_surface(entry.next_action)
    if next_action:
        lines.append(f"next action: {next_action}")
    rendered = "\n".join(lines)
    if _approx_tokens(rendered) > CONT_BUDGET_TOKENS:
        logger.warning(
            "live_state_segment_over_budget",
            extra={"tokens": _approx_tokens(rendered), "budget": CONT_BUDGET_TOKENS},
        )
    return rendered


def _agent_model_prefix(model: str | None) -> str:
    return f"[model:{model}] " if model else ""


def render_agent_registry_segment(now: datetime | None = None) -> str:
    """Direct, lock-free, fail-soft read of the running-agent registry in
    daemon_state.json -- no store, no embedding/cue/rank/similarity gate,
    off the awake-recall critical path.

    Filters to status=='pending' AND spawned_at within
    RUNNING_AGENT_TTL_HOURS at READ TIME: a /clear scenario fires no
    subsequent write to trigger prune_stale_agents, so an abandoned pending
    agent must never render as active. A completed entry never renders.
    """
    from iai_mcp.daemon_state import RUNNING_AGENT_TTL_HOURS, load_state

    try:
        state = load_state()
    except Exception:  # noqa: BLE001 -- registry render must never break the payload
        return ""
    agents = state.get("running_agents")
    if not isinstance(agents, dict) or not agents:
        return ""

    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current - timedelta(hours=RUNNING_AGENT_TTL_HOURS)

    live: list[tuple[str, dict, datetime]] = []
    for agent_id, entry in agents.items():
        if not isinstance(entry, dict) or entry.get("status") != "pending":
            continue
        spawned_at = entry.get("spawned_at")
        try:
            spawned_dt = datetime.fromisoformat(str(spawned_at))
            if spawned_dt.tzinfo is None:
                spawned_dt = spawned_dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if spawned_dt < cutoff:
            continue
        live.append((str(agent_id), entry, spawned_dt))

    live.sort(key=lambda item: item[2])
    lines: list[str] = []
    for agent_id, entry, spawned_dt in live[:AGENT_REGISTRY_MAX_RENDERED]:
        model_prefix = _agent_model_prefix(entry.get("model"))
        role = entry.get("role") or ""
        artifact = entry.get("expected_artifact") or ""
        line = (
            f"- {model_prefix}agent {agent_id[:8]} ({role}): "
            f"expecting {artifact} (spawned {spawned_dt.isoformat()})"
        )
        lines.append(line[:AGENT_REGISTRY_LINE_CHAR_CAP])

    rendered = "\n".join(lines)
    if _approx_tokens(rendered) > AGENT_REGISTRY_BUDGET_TOKENS:
        logger.warning(
            "agent_registry_segment_over_budget",
            extra={"tokens": _approx_tokens(rendered), "budget": AGENT_REGISTRY_BUDGET_TOKENS},
        )
    return rendered


_CONTINUITY_CACHE_NAME = ".session-continuity.cached.md"


def _continuity_cache_path(store: Any) -> Path:
    root = getattr(store, "root", None)
    base = Path(root) if root is not None else Path.home() / ".iai-mcp"
    return base / _CONTINUITY_CACHE_NAME


def _live_state_block_is_substantive(block: str) -> bool:
    return any(
        line.startswith("focus: ") or line.startswith("next action: ")
        for line in block.splitlines()
    )


def _extract_sentinel_block(text: str, name: str) -> str:
    """Inner content between <name> and </name> (first well-formed pair),
    or "" if the pair is absent/malformed."""
    open_tag = f"<{name}>"
    close_tag = f"</{name}>"
    start = text.find(open_tag)
    if start == -1:
        return ""
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return ""
    return text[start:end].strip("\n")


def _resanitize_preserved_live_state(block: str) -> str:
    """A block re-read from disk may predate a marker-stripping fix; only
    re-use lines matching the exact fold-free shape this renderer emits,
    and reject any surviving marker-tag punctuation -- never launder disk
    content back into the file unexamined."""
    lines: list[str] = []
    for line in block.splitlines():
        if not (
            line.startswith("goal: ")
            or line.startswith("focus: ")
            or line.startswith("next action: ")
        ):
            continue
        if "<" in line or ">" in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def write_continuity_cache(store: Any, *, allow_downgrade: bool = False) -> None:
    """Write the ONE session-agnostic eager continuity file: a live-state
    block (phase/step, rendered fold-free) then an agent-registry block
    (pending agents), each independently sentinel-delimited. Mirrors the
    working-tier snapshot's atomic tmp+chmod(0600)+os.replace idiom, resolved
    under the same store root. Nothing precedes the first open sentinel; an
    empty segment writes its sentinel pair with no inner lines.

    A thin park-then-reopen (a task switch that opens a fresh, still-empty
    entry) must never clobber a substantive live-state block already on
    disk -- the eager file is the sole reconstruction source across a
    /clear, and the new session's own entry has no next_action yet.
    allow_downgrade=True bypasses this guard for an explicit close or an
    explicit caller-cleared focus/next_action, where an empty block IS the
    new ground truth.
    """
    path = _continuity_cache_path(store)
    live_state = render_live_state_segment(fold_sensory=False)
    agent_registry = render_agent_registry_segment()

    existing_text = ""
    if path.exists():
        try:
            existing_text = path.read_text(encoding="utf-8")
        except OSError:
            existing_text = ""

    if (
        not allow_downgrade
        and not _live_state_block_is_substantive(live_state)
        and existing_text
    ):
        preserved = _resanitize_preserved_live_state(
            _extract_sentinel_block(existing_text, "iai-mcp-live-state")
        )
        if _live_state_block_is_substantive(preserved):
            live_state = preserved

    lines = ["<iai-mcp-live-state>"]
    if live_state:
        lines.append(live_state)
    lines.append("</iai-mcp-live-state>")
    lines.append("<iai-mcp-agent-registry>")
    if agent_registry:
        lines.append(agent_registry)
    lines.append("</iai-mcp-agent-registry>")
    text = "\n".join(lines) + "\n"

    if existing_text == text and path.exists():
        # mtime means "last confirmed current," not "last content change" --
        # a stable, unchanging monotropic session must not read as stale
        # once RUNNING_AGENT_TTL_HOURS of identical content has elapsed.
        try:
            os.utime(path, None)
        except OSError:
            pass
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


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


def _rich_club_admission(
    store: MemoryStore,
    rich_club: list[UUID],
    *,
    budget: int,
    now: datetime | None = None,
) -> list[tuple[UUID, MemoryRecord, str, str, str]]:
    # `now` is injectable so a caller threading its own value observes the
    # same age snapshot the render uses -- required for callers that must
    # compare two passes without a clock-read race between them.
    if not rich_club:
        return []
    try:
        by_uuid = store.get_batch(rich_club)
    except (OSError, RuntimeError, ValueError):
        return []

    now = now or datetime.now(timezone.utc)
    admitted: list[tuple[UUID, MemoryRecord, str, str, str]] = []
    running = 0
    for uid in rich_club:
        rec = by_uuid.get(uid)
        if rec is None:
            continue
        cleaned = _clean_surface(rec.literal_surface)
        if not cleaned:
            continue
        age = age_label(rec.created_at, now)
        age_part = f" ({age})" if age else ""
        # Selection is always keyed to the legacy (verbose) line cost, after
        # the same 88-char truncation the render applies -- so the admitted
        # set is identical to the pre-compaction render at any corpus size.
        # The toggle only changes what gets EMITTED, never what gets picked.
        legacy_aaak = generate_aaak_index(rec)
        if len(legacy_aaak) > 88:
            legacy_aaak = legacy_aaak[:88] + "…"
        cost = _approx_tokens(f"{legacy_aaak}{age_part}: {cleaned[:60]}")
        if running + cost + 1 > budget:
            break
        admitted.append((uid, rec, cleaned, age_part, legacy_aaak))
        running += cost + 1
    return admitted


def _rich_club_segment_with_budget(
    store: MemoryStore,
    rich_club: list[UUID],
    *,
    budget: int,
    now: datetime | None = None,
) -> str:
    compact = _compact_label_enabled()
    lines: list[str] = []
    for uid, rec, cleaned, age_part, legacy_aaak in _rich_club_admission(
        store, rich_club, budget=budget, now=now
    ):
        if compact:
            aaak = _compact_aaak_label(rec.tier, rec.tags)
            if len(aaak) > 88:
                aaak = aaak[:88] + "…"
        else:
            aaak = legacy_aaak
        if "cls_summary" in rec.tags:
            prefix_match = CLS_SUMMARY_PREFIX_RE.match(rec.literal_surface)
            content = (
                _clean_surface(rec.literal_surface[prefix_match.end():])[:RICH_CLUB_CLS_SUMMARY_CAP]
                if prefix_match
                else cleaned[:60]
            )
        else:
            content = cleaned[:60]
        line = f"{aaak}{age_part}: {content}"
        lines.append(line)
    return "\n".join(lines)


def _candidate_session_id(rec: object) -> str:
    try:
        for entry in getattr(rec, "provenance", None) or []:
            if isinstance(entry, dict) and entry.get("session_id"):
                return str(entry["session_id"])
        return str(getattr(rec, "session_id", "") or "")
    except Exception:  # noqa: BLE001 -- attribution must never break the payload
        return ""


def _model_label(rec: object) -> str:
    from iai_mcp.model_attribution import normalize_model

    try:
        for entry in getattr(rec, "provenance", None) or []:
            value = entry.get("model") if isinstance(entry, dict) else None
            model = normalize_model(value)
            if model:
                return f"[model:{model}] "
    except Exception:  # noqa: BLE001 -- attribution must never break the payload
        pass
    return ""


def _origin_label(rec: object) -> str:
    # Ambient feeds mix every session and project; an unlabeled line reads as
    # "my recent work" and a parallel session's thread gets adopted as this
    # one's. The label names the origin so the model can attribute, not guess.
    model_label = _model_label(rec)
    try:
        prov = getattr(rec, "provenance", None) or []
        cwd = ""
        for entry in prov:
            if isinstance(entry, dict) and entry.get("cwd"):
                cwd = str(entry["cwd"])
                break
        if cwd:
            import os as _os

            return f"[{_os.path.basename(cwd.rstrip('/')) or cwd}] {model_label}"
        sid = ""
        for entry in prov:
            if isinstance(entry, dict) and entry.get("session_id"):
                sid = str(entry["session_id"])
                break
        if not sid:
            sid = str(getattr(rec, "session_id", "") or "")
        if sid and sid != "-":
            return f"[s:{sid[:6]}] {model_label}"
    except Exception:  # noqa: BLE001 -- labels must never break the payload
        pass
    return model_label


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
                model=ev.get("model"),
            ))

    candidates.sort(key=lambda r: r.created_at, reverse=True)
    now = datetime.now(timezone.utc)
    lines: list[str] = []
    for r in candidates:
        if len(lines) >= max_records:
            break
        # Procedural chunks are indexed/reachable but never served as
        # visible text -- mirrors core._passes_mode_filter.
        if r.tier == "procedural":
            continue
        cleaned = _clean_surface(r.literal_surface)
        if not cleaned:
            continue
        age = age_label(r.created_at, now)
        prefix = f"({age}) " if age else ""
        origin = _origin_label(r)
        lines.append(f"- {prefix}{origin}{cleaned[:120]}")
    return "\n".join(lines)


def _norm_delta_ts(ts: "datetime | str") -> str:
    # Same fail-open UTC-normalize compare idiom the RPC uses on the caller
    # watermark, extended to accept the datetime rows the delta read returns.
    try:
        dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(
            str(ts).replace("Z", "+00:00").replace(" ", "T")
        )
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return str(ts)


def render_session_delta(
    store: MemoryStore,
    watermark: str,
    *,
    session_id: str = "-",
) -> str:
    """Per-turn delta: only records newer than watermark, in the
    recent-thread line format, under one header — never the static
    session-start blocks (those already shipped once at session start).
    """
    fetch_k = max(4 * K_DELTA, 40)
    try:
        rows = store._recent_user_turns_candidate_rows(fetch_k)
    except (OSError, RuntimeError, ValueError, AttributeError):
        rows = []

    fetched = [r for r in rows if r.id != L0_RECORD_UUID]

    watermark_norm = _norm_delta_ts(watermark) if watermark else ""

    window_incomplete = False
    if fetched:
        oldest_norm = _norm_delta_ts(fetched[-1].created_at)
        if not watermark_norm or oldest_norm > watermark_norm:
            window_incomplete = True

    candidates: list = list(fetched)
    try:
        from iai_mcp.capture import _idem_tag as _cap_idem_tag
        from iai_mcp.capture import read_pending_live_events
        from iai_mcp.store import _PendingTurn

        pending_events = read_pending_live_events()
    except (OSError, RuntimeError, ValueError, ImportError):
        pending_events = []

    if pending_events:
        seen_pending: set = set()
        for ev in pending_events:
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
                model=ev.get("model"),
            ))

    candidates.sort(key=lambda r: r.created_at, reverse=True)

    now = datetime.now(timezone.utc)
    survivors: list[str] = []
    for r in candidates:
        r_norm = _norm_delta_ts(r.created_at)
        if watermark_norm and r_norm <= watermark_norm:
            continue
        # The delta is cross-session memory: the caller's own turns are
        # already in its context and would only echo back.
        if session_id != "-" and _candidate_session_id(r) == session_id:
            continue
        # Procedural chunks are indexed/reachable but never served as
        # visible text -- mirrors core._passes_mode_filter.
        if r.tier == "procedural":
            continue
        cleaned = _clean_surface(r.literal_surface)
        if not cleaned:
            continue
        age = age_label(r.created_at, now)
        prefix = f"({age}) " if age else ""
        origin = _origin_label(r)
        survivors.append(f"- {prefix}{origin}{cleaned[:120]}")

    if not survivors:
        return ""

    overflow = window_incomplete or len(survivors) > K_DELTA

    header = "## New since last turn (all sessions/projects; [labels] mark origin)"
    marker = "- …earlier new records elided"
    body = survivors[:K_DELTA]

    def _assemble(lines: list[str], with_marker: bool) -> str:
        parts = [header, *lines]
        if with_marker:
            parts.append(marker)
        return "\n".join(parts)

    rendered = _assemble(body, overflow)
    while _approx_tokens(rendered) > DELTA_MAX_TOKENS and body:
        body = body[:-1]
        overflow = True
        rendered = _assemble(body, overflow)

    return rendered


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

    directives_segment = render_directive_segment(store)
    live_state_segment = render_live_state_segment()

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
            directives=directives_segment,
            live_state=live_state_segment,
        )
    else:
        l0 = _l0_segment(store)
        l1 = _l1_segment(store)
        l2 = _l2_segments(store, assignment)
        if wake_depth == "deep":
            rc = _rich_club_segment_with_budget(
                store, rich_club, budget=_RICH_CLUB_DEEP_BUDGET_TOKENS
            )
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
            directives=directives_segment,
            live_state=live_state_segment,
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
        directives = payload.get("directives") or ""
        live_state = payload.get("live_state") or ""
    else:
        l0 = payload.l0
        l1 = payload.l1
        l2 = list(payload.l2)
        rich_club = payload.rich_club
        recent_thread = payload.recent_thread
        directives = payload.directives
        live_state = payload.live_state
    blocks: list[str] = []
    if directives:
        blocks.append(f"## Standing orders (always active)\n{directives}")
    if live_state:
        blocks.append(f"## Session continuity (always active)\n{live_state}")
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
