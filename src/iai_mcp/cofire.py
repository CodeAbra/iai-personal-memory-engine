"""Ambient recall co-firing: transcript -> used_ids -> retrieval_cofired.

Isolated from `capture.py` on purpose -- this parses untrusted, model-influenced
`tool_result` content, so the attack surface stays out of the hot ambient-capture
parser. The hook subprocess has no store; `write_cofire_spool` stages the record,
`drain_cofire_spool` is the only writer of the `retrieval_cofired` event.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iai_mcp.capture import CORRECTION_CANDIDATE_TEXT_SCAN_CAP, _spool_root
from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)

RECALL_TOOL_NAME_SUBSTR = "memory_recall"

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")

# Every tool_result.content shape classify_tool_result_shape can emit, save
# the open-ended other_* catch-all. A shape observed in a real transcript
# corpus that is NOT in this set is an unclassified gap -- see
# tests/test_cofire_shape_coverage.py.
KNOWN_SHAPES: frozenset[str] = frozenset({
    "str_error",
    "str_other_json",
    "str_hits_json",
    "list_text_hits_json",
    "list_text_other",
    "list_other",
    "dict_hits",
    "dict_other",
})

# Subset of KNOWN_SHAPES the extractor deliberately dispositions: either
# genuinely extracts hits from (str_hits_json, list_text_hits_json) or
# intentionally skips with no hits (str_error). A shape observed in
# tests/fixtures/cofire/observed_shapes.json that is NOT in this set is an
# undispositioned gap -- see tests/test_cofire_wiring.py's coverage gate.
HANDLED_SHAPES: frozenset[str] = frozenset({
    "list_text_hits_json", "str_hits_json", "str_error",
})


def classify_tool_result_shape(content: Any) -> str:
    """Stable shape key for a memory_recall tool_result.content value.

    Read-only classification, never raises. The keys this returns are exactly
    KNOWN_SHAPES plus the open-ended other_* catch-all for anything not
    otherwise matched.

    Deliberately independent of bench/cofire_shape_probe.py's copy: cofire.py
    needs this at runtime without bench/ installed, and bench/ must stay
    store-free at import time. tests/test_cofire_shape_coverage.py pins the
    two in agreement.
    """
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
            return "str_error"
        if isinstance(parsed, dict) and isinstance(parsed.get("hits"), list):
            return "str_hits_json"
        return "str_other_json"
    if isinstance(content, list):
        if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
            text = content[0].get("text")
            if isinstance(text, str):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
                    return "list_text_other"
                if isinstance(parsed, dict) and isinstance(parsed.get("hits"), list):
                    return "list_text_hits_json"
            return "list_text_other"
        return "list_other"
    if isinstance(content, dict):
        if isinstance(content.get("hits"), list):
            return "dict_hits"
        return "dict_other"
    return f"other_{type(content).__name__}"


def _cofire_spool_dir() -> Path:
    return _spool_root() / ".cofire-spool"


# Bounds every hit array before compute_used_ids' O(hits^2 x spans) scan.
HIT_ARRAY_CAP = 64


def _extract_hit_arrays(parsed: dict) -> "tuple[list[str], list[str]]":
    """Given a dict already known to carry a `hits` list, pull (hit_ids,
    hit_surfaces). A hit with a missing/empty/non-string record_id is dropped
    entirely (both lists stay index-aligned) -- an empty id must never flow
    into a persisted event. Dedupe by record_id (first occurrence kept) must
    run before HIT_ARRAY_CAP truncation, not after.
    """
    hit_ids: list[str] = []
    hit_surfaces: list[str] = []
    seen_ids: set = set()
    for hit in parsed.get("hits") or []:
        if not isinstance(hit, dict):
            continue
        record_id = hit.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            continue
        if record_id in seen_ids:
            continue
        seen_ids.add(record_id)
        hit_ids.append(record_id)
        hit_surfaces.append(str(hit.get("literal_surface", "")))
        if len(hit_ids) >= HIT_ARRAY_CAP:
            break
    return hit_ids, hit_surfaces


def _parse_list_hits(content: Any) -> "tuple[list[str], list[str]] | tuple[None, None]":
    """LIST-shape tool_result.content -> (hit_ids, hit_surfaces), ranked order.

    Non-list or unparsable content yields (None, None); never raises. Kept
    for a shape hosts can still emit, not the dominant real-traffic shape
    (see _parse_str_hits_json).
    """
    if not isinstance(content, list):
        return None, None
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
            continue
        if not isinstance(parsed, dict) or not isinstance(parsed.get("hits"), list):
            continue
        return _extract_hit_arrays(parsed)
    return None, None


def _parse_str_hits_json(content: Any) -> "tuple[list[str], list[str]] | tuple[None, None]":
    """STR-shape tool_result.content -> (hit_ids, hit_surfaces): json.loads
    the string; malformed/partial JSON degrades to (None, None), never
    raises. This is the 149x real-traffic shape and the primary extraction
    path.
    """
    if not isinstance(content, str):
        return None, None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        return None, None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("hits"), list):
        return None, None
    return _extract_hit_arrays(parsed)


def _parse_hits(content: Any) -> "tuple[list[str], list[str]] | tuple[None, None]":
    """Shape-dispatched hit extraction for a memory_recall tool_result.content.

    str_hits_json is the primary real-traffic path; list_text_hits_json stays
    supported for hosts that still emit it. Any other classified (e.g.
    str_error) or unclassified shape yields (None, None) and never raises on
    the capture path; an unclassified shape is logged.
    """
    shape = classify_tool_result_shape(content)
    if shape == "str_hits_json":
        return _parse_str_hits_json(content)
    if shape == "list_text_hits_json":
        return _parse_list_hits(content)
    if shape not in KNOWN_SHAPES:
        log.debug("cofire_unknown_tool_result_shape: %s", shape)
    return None, None


def _scan_assistant_text(transcript_objs: "list[dict]", start_idx: int) -> "tuple[str, bool]":
    """Assistant text following a tool_result -> (text, closed).

    A user turn carrying conversational text (not just tool_result plumbing)
    closes the window -- mirrors capture.py's _has_conversational_text
    boundary. The next memory_recall tool_use also closes the window (even
    mid-turn, before that block) -- otherwise two back-to-back recalls with
    no intervening text turn let the first absorb the second's response.
    `closed` is False when transcript_objs runs out before either boundary is
    hit -- the caller (extract_recall_pairs_carrying) carries the recall
    forward to the next hook window instead of resolving it here.
    """
    texts: list[str] = []
    for obj in transcript_objs[start_idx:]:
        if not isinstance(obj, dict):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant" and isinstance(content, list):
            hit_next_recall = False
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and RECALL_TOOL_NAME_SUBSTR in str(block.get("name") or ""):
                    hit_next_recall = True
                    break
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
            if hit_next_recall:
                return "\n".join(texts), True
        elif role == "user" and isinstance(content, list):
            if any(
                isinstance(b, dict) and b.get("type") == "text" and str(b.get("text") or "").strip()
                for b in content
            ):
                return "\n".join(texts), True
    return "\n".join(texts), False


def _following_assistant_text(transcript_objs: "list[dict]", start_idx: int) -> str:
    """Assistant text following a tool_result, same window only. See
    _scan_assistant_text for the window-boundary-aware form."""
    text, _closed = _scan_assistant_text(transcript_objs, start_idx)
    return text


def extract_recall_pairs(transcript_objs: "list[dict]") -> "list[dict]":
    """memory_recall tool_use + its tool_result -> ranked pairs.

    Yields {hit_ids, hit_surfaces, assistant_text} per recall call.
    str_hits_json (the 149x real-traffic shape) is the primary path;
    list_text_hits_json stays supported for hosts that still emit it. A call
    whose tool_result content is neither shape yields nothing for that call
    (never raises).
    """
    pending_ids: set = set()
    pairs: list[dict] = []
    for idx, obj in enumerate(transcript_objs):
        if not isinstance(obj, dict):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and RECALL_TOOL_NAME_SUBSTR in str(block.get("name") or ""):
                tool_use_id = block.get("id")
                if tool_use_id:
                    pending_ids.add(tool_use_id)
            elif block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if tool_use_id not in pending_ids:
                    continue
                pending_ids.discard(tool_use_id)
                hit_ids, hit_surfaces = _parse_hits(block.get("content"))
                if hit_ids is None:
                    continue
                pairs.append({
                    "hit_ids": hit_ids,
                    "hit_surfaces": hit_surfaces,
                    "assistant_text": _following_assistant_text(transcript_objs, idx + 1),
                })
    return pairs


# Same bound as HIT_ARRAY_CAP -- the carried pending_recall payload written
# into turnstate.json can never exceed what extraction already capped it to;
# mirrors the existing `pending` (tool-trailer) cap.
PENDING_RECALL_HITS_CAP = HIT_ARRAY_CAP


def extract_recall_pairs_carrying(
    transcript_objs: "list[dict]",
    carried_pending: "dict | None" = None,
) -> "tuple[list[dict], dict | None]":
    """Window-boundary-aware co-fire extraction for the live per-turn hook.

    Same shape dispatch as extract_recall_pairs, but a trailing recall whose
    window has not closed within transcript_objs is returned as `pending`
    (JSON-safe, bounded) instead of a pair, so the caller can carry it into
    the next hook call's turnstate snapshot instead of dropping it.
    `carried_pending` (from a prior call) is resolved first against this
    window's leading assistant text. Never raises.
    """
    pending_ids: set = set()
    pairs: list[dict] = []
    trailing_pending: "dict | None" = None

    if carried_pending:
        carried_hit_ids = carried_pending.get("hit_ids") or []
        carried_hit_surfaces = carried_pending.get("hit_surfaces") or []
        carried_text = carried_pending.get("assistant_text") or ""
        if carried_hit_ids:
            text, closed = _scan_assistant_text(transcript_objs, 0)
            combined = f"{carried_text}\n{text}" if (carried_text and text) else (carried_text or text)
            combined = combined[:CORRECTION_CANDIDATE_TEXT_SCAN_CAP]
            if closed:
                pairs.append({
                    "hit_ids": carried_hit_ids,
                    "hit_surfaces": carried_hit_surfaces,
                    "assistant_text": combined,
                })
            else:
                trailing_pending = {
                    "hit_ids": carried_hit_ids,
                    "hit_surfaces": carried_hit_surfaces,
                    "assistant_text": combined,
                }

    for idx, obj in enumerate(transcript_objs):
        if not isinstance(obj, dict):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and RECALL_TOOL_NAME_SUBSTR in str(block.get("name") or ""):
                tool_use_id = block.get("id")
                if tool_use_id:
                    pending_ids.add(tool_use_id)
            elif block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if tool_use_id not in pending_ids:
                    continue
                pending_ids.discard(tool_use_id)
                hit_ids, hit_surfaces = _parse_hits(block.get("content"))
                if hit_ids is None:
                    continue
                text, closed = _scan_assistant_text(transcript_objs, idx + 1)
                if closed:
                    pairs.append({
                        "hit_ids": hit_ids,
                        "hit_surfaces": hit_surfaces,
                        "assistant_text": text,
                    })
                    trailing_pending = None
                else:
                    trailing_pending = {
                        "hit_ids": hit_ids[:PENDING_RECALL_HITS_CAP],
                        "hit_surfaces": hit_surfaces[:PENDING_RECALL_HITS_CAP],
                        "assistant_text": text[:CORRECTION_CANDIDATE_TEXT_SCAN_CAP],
                    }
    return pairs, trailing_pending


def _is_full_rank_echo(hit_ids: "list[str]", used_ids: "list[str]") -> bool:
    """True when used_ids covers every hit_id in exact rank order.

    A verbatim, in-order echo of the whole returned list ("here's what I
    recalled: A, B, C") is indistinguishable from genuine full-list use by
    bare substring presence -- this is the self-confirmation trap re-entering
    on the read side. A genuine co-firing signal is SELECTIVE (a proper
    subset) or reordered relative to rank; full-coverage-in-rank-order is
    suspect and gets suppressed by the caller, never a partial or reordered
    match. Precision of this rule is unmeasured (v1 heuristic).
    """
    return bool(hit_ids) and used_ids == list(hit_ids)


def compute_used_ids(
    hit_ids: "list[str]", hit_surfaces: "list[str]", assistant_text: str,
) -> "list[str]":
    """record_ids reflected in assistant_text, ordered by first-mention position.

    Membership and order both depend only on the assistant's own generated
    text -- never on hit_ids' rank order and never on set iteration, or the
    self-confirmation trap (the ranker agreeing with itself) reappears under
    a new name. The overlap scan is bounded to
    CORRECTION_CANDIDATE_TEXT_SCAN_CAP so a long transcript window cannot
    turn this into an unbounded regex/scan over adversarial-capable text. A
    full-coverage, rank-order echo is suppressed -- see _is_full_rank_echo.

    A hit's own occurrence must not be strictly nested inside a longer
    sibling hit's occurrence -- a shorter surface riding in on a longer
    quoted one is the sibling's evidence, not its own.
    """
    scanned = (assistant_text or "")[:CORRECTION_CANDIDATE_TEXT_SCAN_CAP]
    spans_by_hit: "list[tuple[str, list[tuple[int, int]]]]" = []
    for hit_id, surface in zip(hit_ids, hit_surfaces):
        surface = (surface or "").strip()
        if not hit_id or not surface:
            spans_by_hit.append((hit_id, []))
            continue
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(surface), scanned)]
        spans_by_hit.append((hit_id, spans))

    positions: list[tuple[int, str]] = []
    for i, (hit_id, own_spans) in enumerate(spans_by_hit):
        chosen = None
        for start, end in own_spans:
            nested = any(
                o_start <= start and end <= o_end and (o_end - o_start) > (end - start)
                for j, (_oid, other_spans) in enumerate(spans_by_hit)
                if j != i
                for o_start, o_end in other_spans
            )
            if not nested:
                chosen = start
                break
        if chosen is not None:
            positions.append((chosen, hit_id))
    positions.sort(key=lambda pair: pair[0])
    used_ids = [hit_id for _pos, hit_id in positions]
    if _is_full_rank_echo(hit_ids, used_ids):
        return []
    return used_ids


def write_cofire_spool(
    session_id: str, hit_ids: "list[str]", used_ids: "list[str]",
) -> Path:
    """Append one JSON line to the co-fire spool -- record_ids only, never
    literal_surface (verbatim fence). Sibling of, and never the same
    directory as, .deferred-captures: that dir is drained straight into
    episodic memory, and a co-firing record must never land there.
    """
    if not _SESSION_ID_RE.fullmatch(session_id or ""):
        raise ValueError(f"invalid session_id for spool path: {session_id!r}")
    spool_dir = _cofire_spool_dir()
    spool_dir.mkdir(parents=True, exist_ok=True)
    path = spool_dir / f"{session_id}.cofire.jsonl"
    line = json.dumps({
        "session_id": session_id,
        "hit_ids": list(hit_ids),
        "used_ids": list(used_ids),
    })
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
    except OSError:
        pass
    try:
        fh = os.fdopen(fd, "a", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
    with fh:
        fh.write(line + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError as exc:  # noqa: BLE001 -- fsync is best-effort
            log.debug("cofire_spool_fsync_failed: %s", exc)
    return path


def drain_cofire_spool(store: MemoryStore) -> dict:
    """Drain .cofire-spool into retrieval_cofired events, never
    retrieval_reinforced. Each consumed file is removed; a malformed line is
    logged and skipped, never raised (the drain must not wedge on one bad
    record).
    """
    spool_dir = _cofire_spool_dir()
    counts = {"files": 0, "events": 0}
    if not spool_dir.exists():
        return counts
    try:
        entries = sorted(spool_dir.glob("*.cofire.jsonl"))
    except OSError:
        return counts
    for path in entries:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.warning("cofire_spool_read_failed: %s", exc)
            continue
        counts["files"] += 1
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                log.warning("cofire_spool_line_skipped: %s", exc)
                continue
            session_id = record.get("session_id", "-")
            write_event(
                store,
                kind="retrieval_cofired",
                data={
                    "session_id": session_id,
                    "hit_ids": record.get("hit_ids", []),
                    "used_ids": record.get("used_ids", []),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                severity="info",
                session_id=session_id,
                buffered=True,
            )
            counts["events"] += 1
        try:
            path.unlink()
        except OSError as exc:
            log.warning("cofire_spool_unlink_failed: %s", exc)
    return counts
