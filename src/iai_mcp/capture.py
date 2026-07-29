
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from iai_mcp.exceptions import NativeError

MAX_DRAIN_EVENTS_PER_RUN = 5000

# Soft resident-memory ceiling for an in-daemon drain run. Sampled before each
# file is claimed; if the resident set already exceeds this, the run stops
# cleanly (leaving the remaining files on disk for the next cycle, exactly like
# the per-run event cap) and emits a telemetry event. Default 2.5 GiB sits well
# under the watchdog hard cap (4 GiB) so the drain yields BEFORE the watchdog
# would kill the process — self-limit instead of getting killed. Operator-
# overridable; ``0`` or a non-positive / malformed value disables the soft cap.
DRAIN_RSS_SOFT_CAP_DEFAULT_BYTES = 2_684_354_560
#: The drain may GROW the resident set by this much past its own starting
#: point before yielding. The cap exists to yield before the watchdog's
#: hard kill — it must measure what the DRAIN adds, not the daemon's
#: legitimate standing footprint (indexes + embedder grow with the corpus,
#: and an absolute cap below that baseline silences the drain forever).
DRAIN_RSS_GROWTH_BUDGET_DEFAULT_BYTES = 805_306_368


def _drain_rss_soft_cap_bytes() -> int:
    raw = os.environ.get("IAI_MCP_DRAIN_RSS_SOFT_CAP_BYTES")
    if raw is None:
        return DRAIN_RSS_SOFT_CAP_DEFAULT_BYTES
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return DRAIN_RSS_SOFT_CAP_DEFAULT_BYTES
    return val if val > 0 else 0


def _drain_rss_stop_threshold(baseline_rss: int) -> int:
    """Absolute-or-relative, whichever is HIGHER: the configured floor keeps
    small daemons yielding early, the growth budget keeps a daemon whose
    standing footprint already exceeds the floor able to drain at all."""
    growth_raw = os.environ.get("IAI_MCP_DRAIN_RSS_GROWTH_BUDGET_BYTES")
    try:
        growth = int(growth_raw) if growth_raw else DRAIN_RSS_GROWTH_BUDGET_DEFAULT_BYTES
    except (TypeError, ValueError):
        growth = DRAIN_RSS_GROWTH_BUDGET_DEFAULT_BYTES
    soft = _drain_rss_soft_cap_bytes()
    if soft <= 0:
        return 0
    return max(soft, baseline_rss + max(growth, 0))


def _indaemon_drain_disabled() -> bool:
    raw = os.environ.get("IAI_MCP_DISABLE_INDAEMON_DRAIN", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _drain_rss_bytes() -> int:
    """Sample the current process resident set, fail-soft to 0.

    A 0 reading is treated as "unknown" by the soft-cap check (never trips the
    cap on a flaky psutil), so the drain still runs when RSS cannot be read.
    """
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 -- psutil flakiness must not stop the drain
        return 0


# Serializes the in-daemon backlog drain so the startup drain and the drowsy-edge
# drain (both call ``drain_deferred_captures`` on their own thread via
# ``asyncio.to_thread``) can never run concurrently on the same store. Two
# overlapping drains would each hold up to ``MAX_DRAIN_EVENTS_PER_RUN`` events in
# memory at once (double resident cost). A second caller that cannot take the
# lock returns immediately with a ``skipped_single_flight`` marker rather than
# stacking. The per-file ``.processing-{pid}`` claim still guards double-
# processing; this guards double resident cost.
_DRAIN_SINGLE_FLIGHT_LOCK = threading.Lock()

_LIVE_ACTIVE_RE = re.compile(r"\.live\.jsonl$")

from iai_mcp.store import MemoryStore
from iai_mcp.types import (
    SCHEMA_VERSION_CURRENT,
    TIER_ENUM,
    MemoryRecord,
)

log = logging.getLogger(__name__)

DEDUP_COS_THRESHOLD = 0.95
MIN_CAPTURE_LEN = 12
MAX_CAPTURE_LEN = 8000


_EMBED_DATE_ALPHA_DEFAULT = 0.15


def _embed_date_alpha() -> float:
    try:
        raw = float(
            os.environ.get("IAI_MCP_EMBED_DATE_ALPHA", _EMBED_DATE_ALPHA_DEFAULT)
        )
    except (TypeError, ValueError):
        return _EMBED_DATE_ALPHA_DEFAULT
    return min(1.0, max(0.0, raw))


def _blend_date_component(embedder, text: str, now, *, alpha: float) -> list:
    """normalize(v_text + alpha * perp(v_date, v_text)): the capture date
    contributes a bounded direction ORTHOGONAL to the text, so temporal
    cues bind while the text's own alignment is untouched. The projection
    matters: the encoder's space is anisotropic (arbitrary sentence pairs
    already share a high baseline cosine), so a raw date vector acts as a
    hub that inflates same-day unrelated pairs past similarity floors."""
    v_text = list(embedder.embed(text))
    if alpha <= 0.0:
        return v_text
    v_date = list(embedder.embed(f"On {now.strftime('%-d %B %Y')}."))
    t_norm = sum(x * x for x in v_text) ** 0.5
    if t_norm <= 0.0:
        return v_text
    t_hat = [x / t_norm for x in v_text]
    dot = sum(t * d for t, d in zip(t_hat, v_date))
    perp = [d - dot * t for t, d in zip(t_hat, v_date)]
    blended = [t + alpha * p for t, p in zip(v_text, perp)]
    norm = sum(x * x for x in blended) ** 0.5
    if norm <= 0.0:
        return v_text
    return [x / norm for x in blended]


def _dedup_cos_threshold() -> float:
    """Operator-tunable near-duplicate floor for the capture-time cosine gate."""
    raw = os.environ.get("IAI_MCP_DEDUP_COS_THRESHOLD")
    if raw is None:
        return DEDUP_COS_THRESHOLD
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return DEDUP_COS_THRESHOLD
    return val if 0.0 < val <= 1.0 else DEDUP_COS_THRESHOLD

# Daemon RPC dispatch runs each request on its own thread (asyncio.to_thread),
# so concurrent capture_turn() calls (e.g. several sessions/forks draining
# deferred captures at once) can race the dedup check-then-insert sequence.
# This serializes that sequence so a tag can never be checked-as-absent by
# two threads before either has inserted it.
_CAPTURE_DEDUP_LOCK = threading.Lock()

FAILED_MAX_ATTEMPTS: int = 3
FAILED_BACKOFF_BASE_SEC: float = 60.0

_FAILED_ATTEMPT_RE = re.compile(r"-attempt-(\d+)\.jsonl$")
_FAILED_SHAPE_RE = re.compile(r"^(.+?)\.failed-(\d+)(?:-attempt-\d+)?\.jsonl$")

_PROCESSING_MARKER_RE = re.compile(r"\.processing-(\d+)\.jsonl$")
_CRASH_ATTEMPT_RE = re.compile(r"\.crash-(\d+)\.jsonl$")
QUARANTINE_MAX_ATTEMPTS: int = 2


def _pid_is_alive(pid: int) -> bool:
    from iai_mcp.lifecycle_lock import pid_exists

    return pid_exists(pid)


def _strip_processing_marker(
    path: Path, *, log_path: Path | None = None
) -> tuple[Path, bool]:
    new_name = _PROCESSING_MARKER_RE.sub(".jsonl", path.name)
    if new_name == path.name:
        return path, True
    new_path = path.with_name(new_name)
    try:
        path.rename(new_path)
    except OSError as e:
        if log_path is not None:
            try:
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"strip-marker-failed {path.name}: {type(e).__name__}\n"
                    )
            except (OSError, ValueError) as exc:
                log.debug("strip_marker_log_write_failed: %s", exc)
        return path, False
    return new_path, True


def _quarantine_file(
    fpath: Path,
    store: "MemoryStore",
    *,
    log_path: Path,
    attempts: int,
) -> Path:
    quarantine_dir = fpath.parent / ".quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    recovered = _PROCESSING_MARKER_RE.sub(".jsonl", fpath.name)
    recovered = _CRASH_ATTEMPT_RE.sub(".jsonl", recovered)

    ts_prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = quarantine_dir / f"{ts_prefix}-{recovered}"

    shutil.move(str(fpath), str(target))

    try:
        from iai_mcp.events import write_event

        write_event(
            store,
            "deferred_captures_quarantined",
            {
                "file": target.name,
                "reason": "crash_loop",
                "attempts": attempts,
            },
            severity="warning",
            domain="ops",
        )
    except Exception as exc:  # noqa: BLE001 -- fail-safe boundary
        log.debug("quarantine_event_write_failed: %s", exc)
        try:
            with log_path.open("a") as logf:
                logf.write(
                    f"{datetime.now(timezone.utc).isoformat()} "
                    f"quarantined-event-skipped {target.name}\n"
                )
        except (OSError, ValueError) as exc2:
            log.debug("quarantine_event_log_fallback_failed: %s", exc2)

    try:
        with log_path.open("a") as logf:
            logf.write(
                f"{datetime.now(timezone.utc).isoformat()} "
                f"quarantined {target.name}: crash_loop attempts={attempts}\n"
            )
    except (OSError, ValueError) as exc:
        log.debug("quarantine_log_write_failed: %s", exc)

    return target


def _parse_failed_attempt(name: str) -> int:
    m = _FAILED_ATTEMPT_RE.search(name)
    if m:
        return int(m.group(1))
    if ".failed-" in name:
        return 1
    return 0


def _advance_failed_path(
    fpath: Path,
    store: "MemoryStore",
    *,
    first_error: str,
    log_path: Path,
) -> Path:
    prior_attempt = _parse_failed_attempt(fpath.name)
    next_attempt = prior_attempt + 1
    m = _FAILED_SHAPE_RE.match(fpath.name)
    if m:
        base = m.group(1)
        ts_str = m.group(2)
    else:
        base = fpath.stem
        ts_str = str(int(time.time()))
    if next_attempt > FAILED_MAX_ATTEMPTS:
        new_name = f"{base}.permanent-failed-{ts_str}.jsonl"
        failed_path = fpath.with_name(new_name)
        fpath.rename(failed_path)
        try:
            from iai_mcp.events import write_event

            write_event(
                store,
                "permanent_capture_failure",
                {
                    "file": new_name,
                    "first_error": first_error,
                    "attempts": FAILED_MAX_ATTEMPTS,
                },
                severity="critical",
                domain="ops",
            )
        except Exception as exc:  # noqa: BLE001 -- fail-safe boundary
            log.debug("permanent_capture_failure_event_failed: %s", exc)
            try:
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"permanent_capture_failure-event-skipped {new_name}\n"
                    )
            except (OSError, ValueError) as exc2:
                log.debug("permanent_capture_failure_log_failed: %s", exc2)
        return failed_path
    new_name = f"{base}.failed-{ts_str}-attempt-{next_attempt}.jsonl"
    failed_path = fpath.with_name(new_name)
    fpath.rename(failed_path)
    return failed_path


def _run_shield(text: str) -> tuple[str, list[str]]:
    try:
        from iai_mcp.shield import evaluate

        result = evaluate(text)
        verdict = getattr(result, "verdict", "OK")
        tags = list(getattr(result, "tags", []) or [])
        return verdict, tags
    except Exception as exc:  # noqa: BLE001 -- capture fail-safe
        log.debug("shield_evaluate_failed: %s", exc)
        return "OK", []


def _resolve_ts(ts: str | None) -> datetime:
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)


def _idem_tag(
    session_id: str,
    role: str,
    ts_iso: str,
    text: str,
    *,
    source_uuid: str | None = None,
) -> str:
    if source_uuid:
        key = f"{session_id}|{role}|{source_uuid}"
    else:
        key = f"{session_id}|{role}|{ts_iso}|{text}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"idem:{digest}"


def _is_episodic_conversational(tier: str, role: str) -> bool:
    return tier == "episodic" and role in {"user", "assistant"}


def capture_turn(
    store: MemoryStore,
    *,
    cue: str,
    text: str,
    tier: str = "episodic",
    session_id: str = "-",
    role: str = "user",
    ts: str | None = None,
    source_uuid: str | None = None,
    provenance_extra: dict | None = None,
    extra_tags: "list[str] | None" = None,
    near_dup_gate: bool = False,
    live_turn: bool = False,
) -> dict[str, Any]:
    # live_turn=True marks a genuinely-live conversational turn and is the
    # only thing that refreshes the next-turn foresight pack. Replay and
    # bulk-ingest callers must leave it False: a replayed historical turn
    # overwriting the live pack is anticipation of a past that already
    # happened, and it costs an extra embed per event.
    # near_dup_gate=True runs the cosine near-duplicate gate even for
    # conversational-shaped records (episodic + user/assistant), AFTER the
    # exact-key idem check misses. Bulk ingest (study/teach) needs it: idem
    # keys only dedup byte-identical re-teaches, so paraphrases and reflows
    # would otherwise insert near-duplicates at cos 0.95+.
    if tier not in TIER_ENUM:
        return {"status": "skipped", "record_id": None, "reason": f"invalid tier {tier!r}"}

    text = (text or "").strip()
    # A durable working-tier result is stored byte-identical even below the
    # generic noise floor — the marker is carried out-of-band in
    # provenance_extra, never folded into text, so literal_surface stays
    # verbatim. Only the length floor is relaxed for the marked path.
    durable = bool((provenance_extra or {}).get("working_tier_durable"))
    if len(text) < MIN_CAPTURE_LEN and not durable:
        return {"status": "skipped", "record_id": None, "reason": "too short"}
    if not text:
        return {"status": "skipped", "record_id": None, "reason": "too short"}
    if len(text) > MAX_CAPTURE_LEN:
        text = text[:MAX_CAPTURE_LEN]

    verdict, shield_tags = _run_shield(text)
    if verdict == "HARD_BLOCK":
        return {"status": "skipped", "record_id": None, "reason": "shield HARD_BLOCK"}

    now = _resolve_ts(ts)

    from iai_mcp.embed import embedder_for_store
    from iai_mcp.events import TELEMETRY_EMBED_NATIVE_FAILURE, write_event

    try:
        # Embed the message content, never the cue. The cue is a provenance
        # label only (transcript drains and deferred-drain pass a positional
        # cue such as "session <id> turn <n>"); embedding it collapsed the
        # stored vector space and broke semantic recall. text is already
        # validated non-empty above (the MIN_CAPTURE_LEN guard), so embedding
        # text is safe for every caller.
        #
        # The capture date goes into the VECTOR only — the stored surface
        # stays verbatim. Both modes MUST stay opt-in: same-day binding at
        # the embedding layer NECESSARILY raises same-day unrelated-pair
        # cosine, and the similarity floors leave ~0.03 headroom on natural
        # short turns — any strength that binds also breaches a floor.
        # Temporal binding belongs in the rank layer (date-mention cues
        # matched against created_at), not in the vector.
        #   "true"  — literal shared prefix (+~0.18 same-day inflation).
        #   "blend" — normalize(v_text + alpha*perp(v_date, v_text));
        #             +~0.07 inflation at alpha=0.15, still a floor breach;
        #             a floor-safe alpha no longer binds.
        _embedder = embedder_for_store(store)
        _date_mode = os.environ.get("IAI_MCP_EMBED_DATE", "")
        if _date_mode == "true":
            emb = _embedder.embed(f"On {now.strftime('%-d %B %Y')}: {text}")
        elif _date_mode == "blend":
            emb = _blend_date_component(
                _embedder, text, now, alpha=_embed_date_alpha(),
            )
        else:
            emb = _embedder.embed(text)
    except Exception as exc:
        write_event(
            store,
            TELEMETRY_EMBED_NATIVE_FAILURE,
            {
                "op_type": "capture",
                "backend": "rust",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise NativeError(f"capture encode failed: {exc}") from exc
    embedding = list(emb)

    with _CAPTURE_DEDUP_LOCK:
        conversational = _is_episodic_conversational(tier, role)
        if conversational:
            ts_iso = now.isoformat()
            idem_t = _idem_tag(session_id, role, ts_iso, text, source_uuid=source_uuid)
            existing_id = store.find_record_by_tag(idem_t)
            if existing_id is not None:
                try:
                    store.reinforce_record(existing_id)
                except (ValueError, IOError) as exc:
                    log.warning(
                        "capture_dedup_reinforce_failed",
                        extra={
                            "err_type": type(exc).__name__,
                            "record_id": str(existing_id),
                        },
                    )
                if extra_tags:
                    # Shared-ownership union: a chunk two documents share must
                    # carry BOTH doc tags, or superseding the first document
                    # fades a record the second still owns.
                    try:
                        store.add_tags(existing_id, list(extra_tags)[:8])
                    except Exception as exc:  # noqa: BLE001 -- tag union is additive
                        log.debug("reinforce tag union failed: %s", exc)
                return {
                    "status": "reinforced",
                    "record_id": str(existing_id),
                    "reason": "exact-key re-drain",
                }
        if not conversational or near_dup_gate:
            try:
                neighbours = store.query_similar(embedding, k=3, tier=tier)
            except (ValueError, IOError) as exc:
                log.warning(
                    "capture_dedup_query_failed",
                    extra={"err_type": type(exc).__name__, "err": str(exc)[:120]},
                )
                neighbours = []

            dedup_floor = _dedup_cos_threshold()
            for record, score in neighbours:
                if score >= dedup_floor:
                    # A pinned neighbour is a hard lock; never reinforce it.
                    if getattr(record, "never_merge", False):
                        continue
                    try:
                        store.reinforce_record(record.id)
                    except (ValueError, IOError) as exc:
                        log.warning(
                            "capture_dedup_reinforce_failed",
                            extra={
                                "err_type": type(exc).__name__,
                                "record_id": str(record.id),
                            },
                        )
                    if extra_tags:
                        try:
                            store.add_tags(record.id, list(extra_tags)[:8])
                        except Exception as exc:  # noqa: BLE001 -- tag union is additive
                            log.debug("reinforce tag union failed: %s", exc)
                    return {
                        "status": "reinforced",
                        "record_id": str(record.id),
                        "reason": f"cos={score:.3f} >= {dedup_floor}",
                    }

        tags = ["capture", f"role:{role}"]
        if extra_tags:
            # Caller-scoped grouping tags (e.g. a per-document tag on a study
            # ingest). Bounded and deduped; never fold into the idem tag.
            for _t in list(extra_tags)[:8]:
                _t = str(_t).strip()
                if _t and _t not in tags:
                    tags.append(_t)
        if verdict == "FLAG_FOR_REVIEW":
            tags.append("shield:flagged")
            tags.extend(f"shield:{t}" for t in shield_tags[:3])

        if _is_episodic_conversational(tier, role):
            ts_iso = now.isoformat()
            tags.append(_idem_tag(session_id, role, ts_iso, text, source_uuid=source_uuid))

        provenance_list: list[dict] = [
            {"ts": now.isoformat(), "cue": cue or "(auto-capture)",
             "session_id": session_id, "role": role}
        ]
        if provenance_extra:
            provenance_list.append(dict(provenance_extra))

        rec = MemoryRecord(
            id=uuid4(),
            tier=tier,
            literal_surface=text,
            aaak_index="",
            embedding=embedding,
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=provenance_list,
            created_at=now,
            updated_at=now,
            tags=tags,
            language="en",
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=SCHEMA_VERSION_CURRENT,
        )

        try:
            store.insert(rec)
        except Exception as e:
            log.exception("capture_turn insert failed")
            return {"status": "skipped", "record_id": None, "reason": f"insert-failed: {type(e).__name__}"}

    try:
        from iai_mcp.peri_event_buffer import get_buffer
        buf = get_buffer()
        if buf is not None:
            buf.add(rec.id, rec.created_at, rec.tier)
    except Exception as exc:  # noqa: BLE001 -- capture fail-safe
        log.warning(
            "capture_peri_event_buffer_add_failed",
            extra={
                "record_id": str(rec.id),
                "err_type": type(exc).__name__,
            },
        )

    # Mirror every capture into the recent bank, not just transcript drains —
    # the daemon-down degraded-recall path (bank-recall) must surface direct
    # CLI/MCP captures too, or the transit layer goes blind to them.
    try:
        from iai_mcp.memory_bank import append_recent_record
        append_recent_record(store, rec)
    except Exception:  # noqa: BLE001 -- best-effort fail-safe boundary
        log.warning(
            "bank-recent append failed for record %s", rec.id, exc_info=True,
        )

    # Anticipate the next turn while the cue embedding is already in hand:
    # only a genuinely-live conversational user turn refreshes the next-turn
    # memory pack. Failures never fail the capture.
    if live_turn and role == "user" and tier == "episodic":
        try:
            from iai_mcp.foresight import refresh_pack
            refresh_pack(
                store,
                cue_text=text,
                cue_embedding=embedding,
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001 -- anticipation is additive
            log.debug("foresight refresh skipped: %s", exc)

    return {"status": "inserted", "record_id": str(rec.id), "reason": f"tier={tier}"}


def _drain_write_pending(
    store: MemoryStore,
    *,
    cue: str,
    text: str,
    tier: str = "episodic",
    session_id: str = "-",
    role: str = "user",
    ts: str | None = None,
    source_uuid: str | None = None,
) -> dict[str, Any]:
    """Write a captured turn as a pending (un-embedded) row.

    The backlog drain routes every genuinely-new event through this path instead
    of the synchronous embed. The row lands with ``embedding_pending=1`` and a
    zero-vector placeholder; a later deferred-embed pass fills the real vector.
    This keeps the drain a sequence of cheap SQLite writes whose resident-memory
    cost does not grow with the backlog size — a long backlog does not hold the
    embedder, JIT, and columnar pages resident through one synchronous run.

    The same validation, idempotency tag, tags, and provenance shape as the
    synchronous capture path are applied, so the row is dedup-findable by tag and
    verbatim-recallable the instant it is written (recall surfaces pending rows
    through its recency union, independent of the embedding). Pinned-record and
    cosine-dedup semantics that need an embedding are deferred with the vector —
    the drain handles only conversational episodic turns, whose dedup is the
    exact-key idem tag, never a cosine neighbour.
    """
    if tier not in TIER_ENUM:
        return {"status": "skipped", "record_id": None, "reason": f"invalid tier {tier!r}"}

    text = (text or "").strip()
    if len(text) < MIN_CAPTURE_LEN:
        return {"status": "skipped", "record_id": None, "reason": "too short"}
    if len(text) > MAX_CAPTURE_LEN:
        text = text[:MAX_CAPTURE_LEN]

    verdict, shield_tags = _run_shield(text)
    if verdict == "HARD_BLOCK":
        return {"status": "skipped", "record_id": None, "reason": "shield HARD_BLOCK"}

    now = _resolve_ts(ts)

    with _CAPTURE_DEDUP_LOCK:
        if _is_episodic_conversational(tier, role):
            ts_iso = now.isoformat()
            idem_t = _idem_tag(session_id, role, ts_iso, text, source_uuid=source_uuid)
            existing_id = store.find_record_by_tag(idem_t)
            if existing_id is not None:
                try:
                    store.reinforce_record(existing_id)
                except (ValueError, IOError) as exc:
                    log.warning(
                        "capture_dedup_reinforce_failed",
                        extra={
                            "err_type": type(exc).__name__,
                            "record_id": str(existing_id),
                        },
                    )
                return {
                    "status": "reinforced",
                    "record_id": str(existing_id),
                    "reason": "exact-key re-drain",
                }

        tags = ["capture", f"role:{role}"]
        if verdict == "FLAG_FOR_REVIEW":
            tags.append("shield:flagged")
            tags.extend(f"shield:{t}" for t in shield_tags[:3])
        if _is_episodic_conversational(tier, role):
            ts_iso = now.isoformat()
            tags.append(
                _idem_tag(session_id, role, ts_iso, text, source_uuid=source_uuid)
            )

        provenance_list: list[dict] = [
            {"ts": now.isoformat(), "cue": cue or "(auto-capture)",
             "session_id": session_id, "role": role}
        ]

        record_id = str(uuid4())
        try:
            store.insert_pending(
                record_id=record_id,
                tier=tier,
                literal_surface=text,
                tags_json=json.dumps(tags),
                provenance_json=json.dumps(provenance_list),
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
            )
        except Exception as e:  # noqa: BLE001 -- per-event isolation, report to caller
            log.exception("drain pending-row insert failed")
            return {
                "status": "skipped",
                "record_id": None,
                "reason": f"insert-failed: {type(e).__name__}",
            }

    try:
        from iai_mcp.peri_event_buffer import get_buffer
        buf = get_buffer()
        if buf is not None:
            buf.add(UUID(record_id), now, tier)
    except Exception as exc:  # noqa: BLE001 -- capture fail-safe
        log.warning(
            "drain_pending_peri_event_buffer_add_failed",
            extra={"record_id": record_id, "err_type": type(exc).__name__},
        )

    # A pending row is arrived memory: stamp the store-advance sidecar the
    # per-turn hooks watch, and feed the working tier — ambient turns must
    # update the active task exactly like fully-embedded inserts do.
    try:
        from iai_mcp.store_watermark import emit as _emit_watermark

        _emit_watermark(
            getattr(store.db, "_hippo_dir", store.root / "hippo"),
            now.isoformat(),
        )
    except Exception as exc:  # noqa: BLE001 -- sidecar is advisory
        log.debug("drain_pending_watermark_emit_failed: %s", exc)
    try:
        from types import SimpleNamespace

        from iai_mcp import working_tier

        working_tier.update_from_record(
            SimpleNamespace(
                created_at=now,
                literal_surface=text,
                provenance=provenance_list,
                tags=tags,
            ),
            store=store,
        )
    except Exception as exc:  # noqa: BLE001 -- hook isolation
        log.debug("drain_pending_working_feed_failed: %s", exc)

    return {"status": "inserted", "record_id": record_id, "reason": f"tier={tier}"}


def capture_transcript(
    store: MemoryStore,
    transcript_path: Path | str,
    *,
    session_id: str = "-",
    max_turns: int = 100_000,
) -> dict[str, Any]:
    path = Path(transcript_path).expanduser()
    if not path.exists():
        return {"inserted": 0, "reinforced": 0, "skipped": 0, "errors": 1,
                "reason": f"transcript not found: {path}"}

    counts = {"inserted": 0, "reinforced": 0, "skipped": 0, "errors": 0}
    seen = 0
    with path.open() as fh:
        for line in fh:
            if seen >= max_turns:
                break
            seen += 1
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                log.debug("capture_transcript_json_parse_failed: %s", exc)
                counts["errors"] += 1
                continue
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
            role = obj.get("type") or msg.get("role", "")
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = "\n".join(text_parts).strip()
            else:
                text = str(content).strip()
            if not text:
                continue
            result = capture_turn(
                store,
                cue=f"session {session_id} turn {seen}",
                text=text,
                tier="episodic",
                session_id=session_id,
                role=role,
                ts=obj.get("timestamp"),
                source_uuid=obj.get("uuid"),
            )
            status = result.get("status", "skipped")
            if status in counts:
                counts[status] += 1
            else:
                counts["skipped"] += 1

    return counts


_NOISE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("startswith", "<command-message>"),
    ("startswith", "<command-name>"),
    ("startswith", "Base directory for this skill:"),
    ("startswith", "<task-notification>"),
    ("equals",     "[Request interrupted by user]"),
)


def _is_noise(text: str) -> bool:
    for match_type, pattern in _NOISE_PATTERNS:
        if match_type == "startswith":
            if text.startswith(pattern):
                return True
        else:
            if text == pattern:
                return True
    return False


def _parse_transcript_line(
    line: str,
) -> tuple[str, str, str | None, str | None] | None:
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if obj.get("type") == "response_item":
        # Codex rollout transcript: user messages also appear as event_msg
        # records, so ONLY response_item is consumed — anything else would
        # double-capture every user turn.
        payload = obj.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message":
            return None
        role = payload.get("role", "")
        if role not in {"user", "assistant"}:
            return None
        content = payload.get("content", [])
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and isinstance(b.get("text"), str)
            ]
            text = "\n".join(p for p in parts if p).strip()
        else:
            text = str(content or "").strip()
        if not text or text.lstrip().startswith("<environment_context>"):
            return None
        if _is_noise(text):
            return None
        return role, text, payload.get("id") or obj.get("uuid"), obj.get("timestamp")
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
    role = obj.get("type") or msg.get("role", "")
    if role not in {"user", "assistant"}:
        return None
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(parts).strip()
    else:
        text = str(content).strip()
    if not text:
        return None
    if _is_noise(text):
        return None
    return role, text, obj.get("uuid"), obj.get("timestamp")


def write_deferred_event(
    session_id: str,
    role: str,
    text: str,
    *,
    cwd: str | None = None,
    ts: str | None = None,
    source_uuid: str | None = None,
) -> Path:
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}.live.jsonl"

    _compact_live_file_if_oversized(path, session_id)

    need_header = (not path.exists()) or path.stat().st_size == 0
    # Pre-encode plaintext boundary; mode 0600 mirrors capture_queue.py's
    # os.open precedent so no other local user/process can read it.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    # O_CREAT mode applies only on create; enforce 0600 on every open so a
    # pre-existing 0644 file is tightened too (idempotent, best-effort).
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        pass
    try:
        fh = os.fdopen(fd, "a")  # fdopen now owns fd; the with-block closes it
    except BaseException:
        os.close(fd)  # only reached if fdopen itself failed (fd not yet owned)
        raise
    with fh:
        if need_header:
            header = {
                "version": 1,
                "deferred_at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "cwd": cwd or os.getcwd(),
            }
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        event = {
            "text": text,
            "cue": f"session {session_id} turn",
            "tier": "episodic",
            "role": role,
            "ts": ts if ts else datetime.now(timezone.utc).isoformat(),
        }
        if source_uuid:
            event["source_uuid"] = source_uuid
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


_TAIL_MAX_EVENT_LINES: int = 500

_LIVE_SELECT_MAX_FILES: int = 20

# Write-side cap for a single session's .live.jsonl: past this many event
# lines, the next write triggers a drain (promote via drain_active_live_captures)
# then compacts away only the already-promoted lines. Generous relative to the
# 3 existing promotion triggers so this rarely fires in normal operation.
_WRITE_SIDE_MAX_EVENT_LINES: int = 5000

# Conservative min bytes/line used only to gate the expensive exact
# line-count behind a cheap os.stat() in the common case -- never used as
# the actual cap (the cap is always the exact _WRITE_SIDE_MAX_EVENT_LINES
# line count, counted only once the byte-size heuristic trips).
_WRITE_SIDE_MIN_BYTES_PER_LINE: int = 40

# Sentinel exclude_session_id for a self-compaction drain: never equal to a
# real session id, so drain_active_live_captures processes this session's own
# file too (mirrors the nightly SLEEP-tick's exclude_session_id="-").
_SELF_COMPACT_DRAIN_SENTINEL: str = "-"


def _compact_live_file_if_oversized(path: Path, session_id: str) -> None:
    """Bound a session's .live.jsonl before the next append.

    Cheap in the common case: a single os.stat() byte-size check. Only once
    the file is plausibly oversized (byte-size heuristic) does an exact
    line-count read run. Past the cap, drains THIS session's file via the
    existing promotion machinery, then keeps only the header plus lines
    still un-promoted (index >= the persisted drain-offset). Un-promoted
    content is never dropped -- if nothing was promoted, the file is left
    intact (lossless wins over bounded).
    """
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= _WRITE_SIDE_MAX_EVENT_LINES * _WRITE_SIDE_MIN_BYTES_PER_LINE:
        return

    try:
        with path.open(encoding="utf-8") as fh:
            line_count = sum(1 for _ in fh)
    except OSError:
        return
    if line_count <= _WRITE_SIDE_MAX_EVENT_LINES:
        return

    # Best-effort self-drain: succeeds only when NO daemon holds the writer
    # (daemon down). When the daemon IS alive it owns the single writer and
    # advances the drain-offset itself at its drowsy edges — so a failed open
    # here is normal, NOT a reason to skip the trim below. The trim keys off
    # the persisted offset regardless of who advanced it, which is what keeps
    # a marathon session's file bounded while the daemon is the one promoting.
    try:
        from iai_mcp.store import MemoryStore

        store = MemoryStore()
    except Exception as exc:  # noqa: BLE001 -- daemon-alive is the common case
        log.debug("live_compact_self_drain_skipped (daemon owns writer): %s", exc)
    else:
        try:
            drain_active_live_captures(
                store, exclude_session_id=_SELF_COMPACT_DRAIN_SENTINEL
            )
        except Exception as exc:  # noqa: BLE001 -- compaction is best-effort
            log.debug("live_compact_drain_failed: %s", exc)
        finally:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass

    state_dir = Path.home() / ".iai-mcp" / ".capture-state"
    offset_path = state_dir / f"{session_id}.drain-offset"
    try:
        drain_offset = int(offset_path.read_text().strip() or "0") if offset_path.exists() else 0
    except (ValueError, OSError):
        drain_offset = 0

    if drain_offset <= 0:
        return

    try:
        with path.open(encoding="utf-8") as fh:
            pre_stat = os.fstat(fh.fileno())
            raw_lines = fh.readlines()
    except OSError:
        return
    complete_lines = [ln for ln in raw_lines if ln.endswith("\n")]
    if not complete_lines:
        return

    header_line = complete_lines[0]
    event_lines = complete_lines[1:]
    if drain_offset >= len(event_lines):
        remaining = []
    else:
        remaining = event_lines[drain_offset:]

    tmp_path = path.with_suffix(".jsonl.compact-tmp")
    try:
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            tmp_fh = os.fdopen(fd, "w")  # fdopen now owns fd
        except BaseException:
            os.close(fd)  # only reached if fdopen itself failed
            raise
        with tmp_fh:
            tmp_fh.write(header_line)
            tmp_fh.writelines(remaining)
        # Lossless fence: if a concurrent same-session appender grew the file
        # between the read above and this swap, os.replace would discard that
        # un-promoted append. Re-stat and abort (leave the file intact) on any
        # change -- lossless wins over bounded, matching the drain_offset<=0
        # bail-out above.
        try:
            post_stat = path.stat()
        except OSError:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return
        if (
            post_stat.st_size != pre_stat.st_size
            or post_stat.st_mtime_ns != pre_stat.st_mtime_ns
        ):
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return
        os.replace(tmp_path, path)
    except OSError as exc:
        log.warning("live_compact_rewrite_failed: %s", exc)
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return

    try:
        tmp_offset = offset_path.with_suffix(".drain-offset.tmp")
        tmp_offset.write_text("0")
        os.replace(tmp_offset, offset_path)
    except OSError as exc:
        log.warning("live_compact_offset_reset_failed: %s", exc)


def read_pending_live_events(session_id: str | None = None) -> list[dict]:
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    if not deferred_dir.exists():
        return []

    allowlisted: list[tuple[Path, float]] = []
    try:
        with os.scandir(deferred_dir) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name = entry.name
                if _LIVE_ACTIVE_RE.search(name) or _PROCESSING_MARKER_RE.search(name):
                    try:
                        st = entry.stat()
                        allowlisted.append((Path(entry.path), st.st_mtime))
                    except OSError:
                        pass
    except OSError:
        return []

    if not allowlisted:
        return []

    if session_id is None:
        allowlisted.sort(key=lambda t: t[1], reverse=True)
        candidates = allowlisted[:_LIVE_SELECT_MAX_FILES]
    else:
        prefix = f"{session_id}.live"
        own = [(p, m) for p, m in allowlisted if p.name.startswith(prefix)]
        other = [(p, m) for p, m in allowlisted if not p.name.startswith(prefix)]

        own.sort(key=lambda t: t[1], reverse=True)
        other.sort(key=lambda t: t[1], reverse=True)

        cap = _LIVE_SELECT_MAX_FILES
        own_capped = own[:cap]
        remaining = cap - len(own_capped)
        candidates = own_capped + other[:remaining]

    events: list[dict] = []
    for path, _mtime in candidates:
        try:
            with path.open(encoding="utf-8") as fh:
                first_line = fh.readline()
                if not first_line.endswith("\n"):
                    continue
                try:
                    header = json.loads(first_line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if header.get("version", 0) > 1:
                    continue
                file_session_id = header.get("session_id", "-")
                if session_id is not None and file_session_id != session_id:
                    continue

                tail = deque(fh, maxlen=_TAIL_MAX_EVENT_LINES)

                complete_lines = [ln for ln in tail if ln.endswith("\n")]

                for line in complete_lines:
                    try:
                        ev = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    ts_raw = ev.get("ts")
                    ts_dt = _resolve_ts(ts_raw)
                    ts_iso = ts_dt.isoformat()
                    events.append({
                        "text": ev.get("text", ""),
                        "role": ev.get("role", "user"),
                        "tier": ev.get("tier", "episodic"),
                        "session_id": file_session_id,
                        "ts": ts_dt,
                        "ts_iso": ts_iso,
                        "source_uuid": ev.get("source_uuid"),
                    })
        except OSError:
            continue

    events.sort(key=lambda e: e["ts"], reverse=True)
    return events


def write_deferred_captures(
    session_id: str,
    transcript_path: Path | str,
    *,
    cwd: str | None = None,
    max_turns: int = 100_000,
) -> Path:
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    # Include pid for collision safety: two parallel bulk-import workers in
    # the same wall-clock second would otherwise race for the same final
    # path. Stream to a sibling .jsonl.tmp file and atomic-rename only after
    # flush+fsync — drain filters by ``suffix != ".jsonl"`` so the in-progress
    # .tmp is never claimed mid-write.
    final_name = f"{session_id}-{int(time.time())}-{os.getpid()}.jsonl"
    out_path = deferred_dir / final_name
    tmp_path = deferred_dir / f"{final_name}.tmp"
    with tmp_path.open("w") as fh:
        header = {
            "version": 1,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "cwd": cwd or os.getcwd(),
        }
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        path = Path(transcript_path).expanduser()
        if path.exists():
            seen = 0
            with path.open() as src:
                for line in src:
                    if seen >= max_turns:
                        break
                    seen += 1
                    try:
                        obj = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
                    role = obj.get("type") or msg.get("role", "")
                    if role not in {"user", "assistant"}:
                        continue
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        text_parts = [
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        text = "\n".join(text_parts).strip()
                    else:
                        text = str(content).strip()
                    if not text:
                        continue
                    event = {
                        "text": text,
                        "cue": f"session {session_id} turn {seen}",
                        "tier": "episodic",
                        "role": role,
                        "ts": obj.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    }
                    src_uuid = obj.get("uuid")
                    if src_uuid:
                        event["source_uuid"] = src_uuid
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError as exc:  # noqa: BLE001 -- fsync is best-effort
            log.debug("write_deferred_captures fsync failed: %s", exc)
    os.replace(tmp_path, out_path)
    return out_path


def drain_capture_backlog(store: MemoryStore) -> dict[str, int]:
    """Drain both the rotated/crashed deferred files and the still-open live
    spools (incremental, offset-tracked) into the store."""
    counts = drain_deferred_captures(store)
    try:
        live = drain_active_live_captures(store, exclude_session_id="-")
        for k, v in live.items():
            counts[f"live_{k}"] = v
    except Exception as exc:  # noqa: BLE001 -- live sweep must not sink the drain
        log.debug("live spool sweep failed: %s", exc)
    return counts


def drain_deferred_captures(store: MemoryStore) -> dict[str, int]:
    counts = {
        "files_drained": 0,
        "files_failed": 0,
        "events_inserted": 0,
        "events_reinforced": 0,
        "events_skipped_intentional": 0,
        "events_skipped_insert_failed": 0,
        "events_skipped_existing": 0,
    }

    # Rail: operator kill-switch. When set, the in-daemon drain is a no-op — no
    # files are claimed, no embed runs — deferring the whole backlog to the
    # offline ``iai-mcp deferred-drain`` tool. An escape hatch if the in-daemon
    # drain ever misbehaves; capture liveness is preserved because the deferred
    # files stay untouched on disk.
    if _indaemon_drain_disabled():
        counts["disabled"] = 1
        return counts

    # Rail: single-flight. A second concurrent drain on the same store would
    # double the in-memory event footprint; skip rather than stack.
    if not _DRAIN_SINGLE_FLIGHT_LOCK.acquire(blocking=False):
        counts["skipped_single_flight"] = 1
        return counts
    try:
        return _drain_deferred_captures_locked(store, counts)
    finally:
        _DRAIN_SINGLE_FLIGHT_LOCK.release()


def _drain_deferred_captures_locked(
    store: MemoryStore, counts: dict[str, int]
) -> dict[str, int]:
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    log_dir = Path.home() / ".iai-mcp" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir / f"deferred-drain-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    )
    if not deferred_dir.exists():
        return counts
    total_events_processed = 0
    cap_hit = False
    # Rail: soft resident-memory ceiling. Sampled before each claimed file; if the
    # resident set already exceeds the soft cap, the drain stops cleanly (the
    # remaining files stay on disk for the next cycle, like the event cap) so the
    # process yields BEFORE the watchdog hard cap would kill it.
    rss_soft_cap = _drain_rss_stop_threshold(_drain_rss_bytes())
    rss_soft_cap_hit = False
    # Cheap pre-embed idem skip, scoped to this whole drain call. A crash-rotated
    # backlog is dominated by duplicates of already-stored records; embedding each
    # one through capture_turn only to discard it burns the Rust BERT step. The
    # pre-check skips that embed for a known duplicate but still reinforces the
    # pre-existing record once (a re-seen turn strengthens the memory) — and the
    # seen_this_run set collapses all repeats of the same tag in one backlog to a
    # single reinforcement, so a giant repeat-backlog cannot inflate the signal.
    # The tag is reproduced exactly as capture_turn computes it, so the pre-check
    # matches capture_turn's own idem-check, which remains the correctness backstop
    # — the pre-check can only save the embed, never drop a genuinely-new record.
    seen_this_run: set[str] = set()
    # Newest user turn seen this pass (ts_iso, text, session_id) — the drain's
    # anchor for one next-turn pack refresh at the end of the pass. Tracked
    # regardless of dedup outcome: a re-seen turn is still the current context.
    newest_live_turn: "tuple[str, str, str] | None" = None

    for fpath in sorted(deferred_dir.iterdir()):
        if not fpath.is_file():
            continue
        m = _PROCESSING_MARKER_RE.search(fpath.name)
        if not m:
            continue
        pid = int(m.group(1))
        if _pid_is_alive(pid):
            continue
        base_no_marker = _PROCESSING_MARKER_RE.sub(".jsonl", fpath.name)
        crash_m = _CRASH_ATTEMPT_RE.search(base_no_marker)
        if crash_m:
            prior_n = int(crash_m.group(1))
            base_no_crash = _CRASH_ATTEMPT_RE.sub(".jsonl", base_no_marker)
        else:
            prior_n = 0
            base_no_crash = base_no_marker
        next_n = prior_n + 1
        if next_n > QUARANTINE_MAX_ATTEMPTS:
            try:
                _quarantine_file(
                    fpath, store, log_path=log_path, attempts=next_n
                )
            except Exception as exc:  # noqa: BLE001 -- fail-safe boundary
                log.debug("quarantine_file_failed: %s", exc)
        else:
            new_name = base_no_crash.replace(
                ".jsonl", f".crash-{next_n}.jsonl"
            )
            try:
                fpath.rename(fpath.with_name(new_name))
            except Exception as exc:  # noqa: BLE001
                log.debug("crash_rename_failed %s: %s", fpath.name, exc)

    candidates = []
    for fpath in sorted(deferred_dir.iterdir()):
        if not fpath.is_file():
            continue
        if fpath.suffix != ".jsonl":
            continue
        if _LIVE_ACTIVE_RE.search(fpath.name):
            continue
        if _PROCESSING_MARKER_RE.search(fpath.name):
            continue
        if ".permanent-failed-" in fpath.name:
            continue
        if ".failed-" in fpath.name:
            attempt_n = _parse_failed_attempt(fpath.name)
            backoff_sec = FAILED_BACKOFF_BASE_SEC * (2 ** (attempt_n - 1))
            try:
                file_mtime = fpath.stat().st_mtime
            except OSError:
                continue
            if (time.time() - file_mtime) < backoff_sec:
                continue
        candidates.append(fpath)

    # Oldest backlog first: a capped pass must make net progress on the tail,
    # not re-chew whichever session id happens to sort alphabetically first.
    def _mtime_or_inf(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return float("inf")

    candidates.sort(key=_mtime_or_inf)

    for fpath in candidates:
        if cap_hit:
            break
        # Rail: soft resident-memory ceiling — check before claiming the next file
        # so the run yields with the remaining backlog still on disk (deferred, not
        # lost). A 0 reading (psutil unavailable) is treated as unknown and never
        # trips the cap.
        if rss_soft_cap > 0:
            rss_now = _drain_rss_bytes()
            if rss_now > rss_soft_cap:
                rss_soft_cap_hit = True
                cap_hit = True
                try:
                    with log_path.open("a") as logf:
                        logf.write(
                            f"{datetime.now(timezone.utc).isoformat()} "
                            f"rss-soft-cap stop: rss={rss_now} > cap={rss_soft_cap}\n"
                        )
                except (OSError, ValueError) as exc:
                    log.debug("rss_soft_cap_log_write_failed: %s", exc)
                break
        claim_path = fpath.with_name(
            fpath.stem + f".processing-{os.getpid()}.jsonl"
        )
        try:
            fpath.rename(claim_path)
        except FileNotFoundError:
            continue
        except OSError as e:
            try:
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"claim-failed {fpath.name}: {type(e).__name__}\n"
                    )
            except (OSError, ValueError) as exc:
                log.debug("claim_failed_log_write_failed: %s", exc)
            continue
        work_path = claim_path

        file_had_insert_failure = False
        file_first_error: str | None = None
        try:
            # Stream the claimed file line-by-line instead of materializing it.
            # A single deferred file can hold up to the capture turn ceiling of
            # events; reading it all into a list would size the read step's
            # resident cost to the file, not to MAX_DRAIN_EVENTS_PER_RUN. The
            # handle stays open across the whole per-file loop so the per-run cap
            # bounds the resident batch to ~MAX_DRAIN_EVENTS_PER_RUN parsed
            # events, and the un-processed tail is streamed straight to
            # .partial.jsonl from the same handle (never buffered).
            with work_path.open() as fh:
                header_line: str | None = None
                for raw in fh:
                    if raw.strip():
                        header_line = raw.rstrip("\n")
                        break
                if header_line is None:
                    work_path.unlink()
                    continue
                header = json.loads(header_line)
                if header.get("version", 0) > 1:
                    with log_path.open("a") as logf:
                        logf.write(
                            f"{datetime.now(timezone.utc).isoformat()} skip "
                            f"{work_path.name}: version={header.get('version')}\n"
                        )
                    _strip_processing_marker(work_path, log_path=log_path)
                    continue
                session_id = header.get("session_id", "-")
                processed_in_file = 0
                for raw in fh:
                    if not raw.strip():
                        continue
                    ln = raw.rstrip("\n")
                    if total_events_processed >= MAX_DRAIN_EVENTS_PER_RUN:
                        work_path, _strip_ok = _strip_processing_marker(
                            work_path, log_path=log_path
                        )
                        if not _strip_ok:
                            cap_hit = True
                            break
                        partial_path = work_path.with_suffix(".partial.jsonl")
                        tmp_path = work_path.with_suffix(".partial.tmp")
                        # Write header + the cap-triggering line + the rest of the
                        # still-open handle directly. The marker rename above does
                        # not invalidate the open fd (same inode), so the tail is
                        # streamed through one line at a time — the remainder is
                        # preserved byte-for-byte without ever being buffered.
                        with tmp_path.open("w") as ph:
                            ph.write(header_line + "\n")
                            ph.write(ln + "\n")
                            for tail in fh:
                                if not tail.strip():
                                    continue
                                ph.write(tail.rstrip("\n") + "\n")
                            ph.flush()
                            os.fsync(ph.fileno())
                        os.replace(tmp_path, partial_path)
                        work_path.unlink()
                        counts["files_drained"] += 1
                        cap_hit = True
                        break
                    ev = json.loads(ln)
                    tier = ev.get("tier", "episodic")
                    role = ev.get("role", "user")

                    # Politeness: a live foreground recall preempts background
                    # ingestion. Each drained event costs an embed plus several
                    # shared-connection round trips; grinding through them while
                    # a recall is in flight serializes the recall behind this
                    # loop. The backoff is bounded per event and fail-open, so
                    # the drain always completes even under a continuous stream
                    # of foreground reads.
                    try:
                        from iai_mcp.concurrency import foreground_backoff

                        foreground_backoff(max_wait_s=2.0)
                    except Exception:  # noqa: BLE001 -- politeness is advisory
                        pass

                    # Pre-embed idem skip for conversational episodic events. The tag
                    # is reproduced exactly as capture_turn computes it (stripped,
                    # length-bounded text + resolved timestamp). A too-short event
                    # would never become a record (capture_turn skips it before
                    # embedding), so it can never match a stored tag — leave those to
                    # fall through to capture_turn's own "too short" skip.
                    if _is_episodic_conversational(tier, role):
                        norm_text = (ev.get("text", "") or "").strip()
                        if MIN_CAPTURE_LEN <= len(norm_text):
                            if len(norm_text) > MAX_CAPTURE_LEN:
                                norm_text = norm_text[:MAX_CAPTURE_LEN]
                            ts_iso = _resolve_ts(ev.get("ts")).isoformat()
                            # Anchor only turns near wall-clock now: a backlog
                            # replay must not aim the next-turn pack at a past
                            # conversation. Same horizon as the working tier's
                            # live-attention guard.
                            from iai_mcp.working_tier import (
                                WORKING_TIER_IDLE_CLOSE_SEC as _live_horizon,
                            )
                            _turn_age = time.time() - _resolve_ts(ev.get("ts")).timestamp()
                            if (
                                role == "user"
                                and _turn_age <= _live_horizon
                                and (
                                    newest_live_turn is None
                                    or ts_iso > newest_live_turn[0]
                                )
                            ):
                                newest_live_turn = (ts_iso, norm_text, session_id)
                            tag = _idem_tag(
                                session_id,
                                role,
                                ts_iso,
                                norm_text,
                                source_uuid=ev.get("source_uuid"),
                            )
                            if tag in seen_this_run:
                                # Already inserted-or-reinforced this exact tag earlier
                                # in this drain. A crash-rotated backlog can repeat the
                                # same turn tens of thousands of times; collapse those
                                # to a single reinforcement so the Hebbian signal is not
                                # inflated by the size of the backlog.
                                counts["events_skipped_existing"] += 1
                                total_events_processed += 1
                                processed_in_file += 1
                                continue
                            existing_id = store.find_record_by_tag(tag)
                            if existing_id is not None:
                                # The record already exists in the store. Skip the
                                # expensive embed, but still reinforce it once — re-seeing
                                # a turn is a memory-strengthening signal, exactly what
                                # capture_turn's own duplicate branch does. reinforce_record
                                # is a cheap edge boost (no embed), so the drain stays fast.
                                try:
                                    store.reinforce_record(existing_id)
                                    counts["events_reinforced"] += 1
                                except (ValueError, IOError) as exc:
                                    log.warning(
                                        "drain_dedup_reinforce_failed",
                                        extra={
                                            "err_type": type(exc).__name__,
                                            "record_id": str(existing_id),
                                        },
                                    )
                                seen_this_run.add(tag)
                                total_events_processed += 1
                                processed_in_file += 1
                                continue
                            seen_this_run.add(tag)

                    # First phase of the two-phase drain: write the genuinely-new event as
                    # a pending (un-embedded) row. No embedder, JIT, or columnar pages
                    # are held resident during the drain, so a large backlog does not
                    # climb the resident set through one long synchronous embed run.
                    # The deferred-embed pass (driven by the wake sequence after the
                    # drain) fills the real vector in bounded batches. The pending row
                    # is dedup-findable by tag and verbatim-recallable immediately.
                    result = _drain_write_pending(
                        store,
                        cue=ev.get("cue", ""),
                        text=ev.get("text", ""),
                        tier=tier,
                        session_id=session_id,
                        role=role,
                        ts=ev.get("ts"),
                        source_uuid=ev.get("source_uuid"),
                    )
                    status = result.get("status", "skipped")
                    reason = result.get("reason", "")
                    if status == "inserted":
                        counts["events_inserted"] += 1
                        # Mirror the new turn into the recent bank so the daemon-down
                        # degraded-recall path (bank-recall) still surfaces it. The
                        # recent-bank recall is verbatim substring matching — it never
                        # reads the stored vector — so the pending row's placeholder
                        # vector is irrelevant here; the verbatim text is what matters.
                        try:
                            from iai_mcp.memory_bank import append_recent_record

                            rid_str = result.get("record_id")
                            if rid_str:
                                rec = store.get(UUID(rid_str))
                                if rec is not None:
                                    append_recent_record(store, rec)
                        except Exception:  # noqa: BLE001 -- best-effort fail-safe boundary
                            log.warning(
                                "bank-recent append failed for record %s",
                                result.get("record_id"),
                                exc_info=True,
                            )
                    elif status == "reinforced":
                        counts["events_reinforced"] += 1
                    elif status == "skipped" and reason.startswith("insert-failed:"):
                        counts["events_skipped_insert_failed"] += 1
                        file_had_insert_failure = True
                        if file_first_error is None:
                            file_first_error = reason
                    else:
                        counts["events_skipped_intentional"] += 1
                    total_events_processed += 1
                    processed_in_file += 1
            if cap_hit:
                break
            if file_had_insert_failure:
                work_path, _strip_ok = _strip_processing_marker(
                    work_path, log_path=log_path
                )
                if not _strip_ok:
                    try:
                        with log_path.open("a") as logf:
                            logf.write(
                                f"{datetime.now(timezone.utc).isoformat()} "
                                f"insert-failed-skip {work_path.name}: "
                                f"strip-failed, leaving for next pass\n"
                            )
                    except (OSError, ValueError) as exc:
                        log.debug("insert_failed_skip_log_write_failed: %s", exc)
                    counts["files_failed"] += 1
                    continue
                failed_path = _advance_failed_path(
                    work_path,
                    store,
                    first_error=file_first_error or "unknown",
                    log_path=log_path,
                )
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} insert-failed "
                        f"{work_path.name}: first_error={file_first_error}\n"
                    )
                counts["files_failed"] += 1
            else:
                work_path.unlink()
                counts["files_drained"] += 1
        except Exception as e:  # noqa: BLE001 -- per-file isolation, never raise
            try:
                work_path, _strip_ok = _strip_processing_marker(
                    work_path, log_path=log_path
                )
                if not _strip_ok:
                    try:
                        with log_path.open("a") as logf:
                            logf.write(
                                f"{datetime.now(timezone.utc).isoformat()} "
                                f"exception-skip {work_path.name}: "
                                f"strip-failed, leaving for next pass: {e!r}\n"
                            )
                    except (OSError, ValueError) as exc:
                        log.debug("exception_skip_log_write_failed: %s", exc)
                    counts["files_failed"] += 1
                    continue
                failed_path = _advance_failed_path(
                    work_path,
                    store,
                    first_error=file_first_error or repr(e),
                    log_path=log_path,
                )
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} failed "
                        f"{work_path.name}: {type(e).__name__}: {e}\n"
                    )
            except Exception as exc:  # noqa: BLE001 -- capture fail-safe
                log.debug("drain_exception_handler_failed: %s", exc)
            counts["files_failed"] += 1
    try:
        from iai_mcp.memory_bank import prune_recent_windows

        prune_recent_windows()
    except Exception:  # noqa: BLE001 -- best-effort fail-safe boundary
        log.warning("bank-recent prune failed", exc_info=True)

    if rss_soft_cap_hit:
        counts["rss_soft_cap_hit"] = 1
        try:
            from iai_mcp.events import TELEMETRY_DRAIN_RSS_SOFT_CAP, write_event

            write_event(
                store,
                TELEMETRY_DRAIN_RSS_SOFT_CAP,
                {
                    "rss_soft_cap_bytes": rss_soft_cap,
                    "events_processed": total_events_processed,
                    "files_drained": counts["files_drained"],
                },
                severity="warning",
                domain="ops",
            )
        except Exception as exc:  # noqa: BLE001 -- telemetry must not break the drain
            log.debug("drain_rss_soft_cap_event_failed: %s", exc)

    # Flush the drained records out of the in-memory insert buffer HERE, in
    # the drain's own background context. Left unflushed, the buffer's
    # read-your-writes discipline makes the FIRST post-drain read that needs
    # a buffered row (a recall's by-id batch fetch) pay the whole flush —
    # encrypt + batch insert + index feeds — synchronously on the awake
    # recall path, and every concurrent recall serializes behind it.
    if counts["events_inserted"]:
        try:
            from iai_mcp.store import flush_record_buffer

            flush_record_buffer(store)
        except Exception as exc:  # noqa: BLE001 -- flush failure defers to the
            # read-side flush; never breaks the drain.
            log.debug("drain_post_flush_failed: %s", exc)

    # Rail: post-drain memory relief. After a run that did real work, hand idle
    # allocator pages back to the OS (arrow pool release + gc + macOS pressure
    # relief) so the per-run transient does not accumulate into the warm plateau.
    # Reuses the existing relief helper; adds no consolidation-pipeline step.
    if counts["files_drained"] or counts["events_inserted"] or counts["events_reinforced"]:
        try:
            from iai_mcp.lilli.cycle.sleep_pipeline._memory_relief import (
                _step_memory_relief,
            )

            _step_memory_relief(label="deferred_drain")
        except Exception as exc:  # noqa: BLE001 -- relief is advisory, never fatal
            log.debug("drain_post_relief_failed: %s", exc)

    # Stash the newest user turn as the next-turn pack anchor. The drain
    # itself never embeds (its resident-set discipline depends on that); the
    # wake-sequence pass — where the embedder is warm anyway — consumes the
    # anchor and refreshes the pack once, so anticipation tracks the
    # conversation's current point instead of thrashing per replayed event.
    if newest_live_turn is not None:
        store._foresight_anchor = newest_live_turn

    return counts


_PERMANENT_FAILED_RE = re.compile(r"^\.permanent-failed-([^.]+)\.jsonl$")
_PERMANENT_FAILED_NAMED_RE = re.compile(r"^(.+)\.permanent-failed-([^.]+)\.jsonl$")


def _count_lines(fpath: Path) -> int:
    try:
        with fpath.open() as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return 0


def drain_permanent_failed_files(
    store: MemoryStore,
    *,
    deferred_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    if deferred_dir is None:
        store_env = os.environ.get("IAI_MCP_STORE")
        if store_env:
            deferred_dir = Path(store_env).parent / ".deferred-captures"
        else:
            deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"

    if not deferred_dir.exists():
        if dry_run:
            return {"dry_run": True, "files": [], "count": 0}
        return {
            "dry_run": False,
            "files": [],
            "inserted": 0,
            "dropped": 0,
            "files_recovered": [],
            "quarantine_dir": str(deferred_dir / ".quarantine"),
        }

    terminal_files: list[Path] = []
    for entry in sorted(deferred_dir.iterdir()):
        if not entry.is_file():
            continue
        if ".permanent-failed-" in entry.name and entry.suffix == ".jsonl":
            terminal_files.append(entry)

    if dry_run:
        file_list = [
            {"name": f.name, "line_count": _count_lines(f)}
            for f in terminal_files
        ]
        return {"dry_run": True, "files": file_list, "count": len(file_list)}

    quarantine_dir = deferred_dir / ".quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    inserted_total = 0
    dropped_total = 0
    files_recovered: list[str] = []
    file_list = []

    for fpath in terminal_files:
        try:
            shutil.copy2(fpath, quarantine_dir / fpath.name)
        except Exception as exc:  # noqa: BLE001 -- fail-safe; log and continue
            log.warning("drain_permanent_failed_quarantine_failed %s: %s", fpath.name, exc)
            continue

        line_count = 0
        file_inserted = 0
        file_dropped = 0

        try:
            with fpath.open() as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

            if not lines:
                fpath.unlink(missing_ok=True)
                files_recovered.append(fpath.name)
                file_list.append({"name": fpath.name, "line_count": 0})
                continue

            line_count = len(lines)

            first_obj: dict | None = None
            try:
                first_obj = json.loads(lines[0])
            except (json.JSONDecodeError, ValueError):
                pass

            has_header = isinstance(first_obj, dict) and "version" in first_obj
            if has_header:
                session_id = (first_obj or {}).get("session_id", "-")
                event_lines = lines[1:]
                for ln in event_lines:
                    try:
                        ev = json.loads(ln)
                    except (json.JSONDecodeError, ValueError):
                        file_dropped += 1
                        continue
                    text = (ev.get("text") or "").strip()
                    role = ev.get("role", "user")
                    if not text or _is_noise(text):
                        file_dropped += 1
                        continue
                    result = capture_turn(
                        store,
                        cue=ev.get("cue") or "recovered turn",
                        text=text,
                        tier=ev.get("tier", "episodic"),
                        session_id=session_id,
                        role=role,
                        ts=ev.get("ts"),
                        source_uuid=ev.get("source_uuid"),
                    )
                    if result.get("status") in ("inserted", "reinforced"):
                        file_inserted += 1
                    else:
                        file_dropped += 1
            else:
                raw_session_id = "-"
                for ln in lines:
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict) and "session_id" in obj:
                            raw_session_id = obj.get("session_id") or "-"
                    except (json.JSONDecodeError, ValueError):
                        pass
                    parsed = _parse_transcript_line(ln)
                    if parsed is None:
                        file_dropped += 1
                        continue
                    role, text, src_uuid, src_ts = parsed
                    result = capture_turn(
                        store,
                        cue="recovered turn",
                        text=text,
                        tier="episodic",
                        session_id=raw_session_id,
                        role=role,
                        ts=src_ts,
                        source_uuid=src_uuid,
                    )
                    if result.get("status") in ("inserted", "reinforced"):
                        file_inserted += 1
                    else:
                        file_dropped += 1

            try:
                fpath.unlink()
            except OSError as exc:
                log.warning("drain_permanent_failed_unlink_failed %s: %s", fpath.name, exc)

            inserted_total += file_inserted
            dropped_total += file_dropped
            files_recovered.append(fpath.name)
            file_list.append({"name": fpath.name, "line_count": line_count})

        except Exception as exc:  # noqa: BLE001 -- per-file isolation
            log.warning("drain_permanent_failed_file_error %s: %s", fpath.name, exc)
            dropped_total += 1
            file_list.append({"name": fpath.name, "line_count": line_count})

    return {
        "dry_run": False,
        "files": file_list,
        "inserted": inserted_total,
        "dropped": dropped_total,
        "files_recovered": files_recovered,
        "quarantine_dir": str(quarantine_dir),
    }


def drain_active_live_captures(
    store: MemoryStore,
    *,
    exclude_session_id: str,
) -> dict[str, int]:
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    state_dir = Path.home() / ".iai-mcp" / ".capture-state"
    counts: dict[str, int] = {
        "files_drained": 0,
        "events_inserted": 0,
        "events_reinforced": 0,
        "events_skipped": 0,
    }
    if not deferred_dir.exists():
        return counts

    for fpath in sorted(deferred_dir.iterdir()):
        if not fpath.is_file():
            continue
        if not _LIVE_ACTIVE_RE.search(fpath.name):
            continue
        try:
            with fpath.open() as fh:
                raw_lines = fh.readlines()
        except OSError:
            continue
        if not raw_lines:
            continue

        complete_lines = [ln for ln in raw_lines if ln.endswith("\n")]
        if not complete_lines:
            continue

        try:
            header = json.loads(complete_lines[0])
        except (json.JSONDecodeError, ValueError):
            continue
        if header.get("version", 0) > 1:
            continue

        file_session_id: str = header.get("session_id", "-")
        if file_session_id == exclude_session_id:
            continue

        offset_path = state_dir / f"{file_session_id}.drain-offset"
        prev_offset: int = 0
        try:
            if offset_path.exists():
                prev_offset = int(offset_path.read_text().strip() or "0")
        except (ValueError, OSError):
            prev_offset = 0

        event_lines = complete_lines[1:]
        new_lines = event_lines[prev_offset:]
        if not new_lines:
            continue

        new_offset = prev_offset
        file_had_insert = False
        for ln in new_lines:
            try:
                ev = json.loads(ln)
            except (json.JSONDecodeError, ValueError):
                new_offset += 1
                counts["events_skipped"] += 1
                continue
            result = capture_turn(
                store,
                cue=ev.get("cue", ""),
                text=ev.get("text", ""),
                tier=ev.get("tier", "episodic"),
                session_id=file_session_id,
                role=ev.get("role", "user"),
                ts=ev.get("ts"),
                source_uuid=ev.get("source_uuid"),
            )
            status = result.get("status", "skipped")
            if status == "inserted":
                counts["events_inserted"] += 1
                file_had_insert = True
            elif status == "reinforced":
                counts["events_reinforced"] += 1
            else:
                counts["events_skipped"] += 1
            new_offset += 1

        if file_had_insert:
            try:
                from iai_mcp.store import flush_record_buffer
                flush_record_buffer(store)
            except Exception as _flush_exc:  # noqa: BLE001 -- flush is best-effort
                log.warning("drain_active_flush_failed: %s", _flush_exc)

        state_dir.mkdir(parents=True, exist_ok=True)
        tmp_offset = offset_path.with_suffix(".drain-offset.tmp")
        try:
            tmp_offset.write_text(str(new_offset))
            os.replace(tmp_offset, offset_path)
        except OSError as exc:
            log.warning("drain_active_offset_write_failed: %s", exc)

        if file_had_insert:
            counts["files_drained"] += 1

    # Rail: post-drain memory relief on the wake-edge live-drain path too — hand
    # idle allocator pages back to the OS after real work. Reuses the existing
    # relief helper; adds no consolidation-pipeline step.
    if counts["events_inserted"] or counts["events_reinforced"]:
        try:
            from iai_mcp.lilli.cycle.sleep_pipeline._memory_relief import (
                _step_memory_relief,
            )

            _step_memory_relief(label="active_live_drain")
        except Exception as exc:  # noqa: BLE001 -- relief is advisory, never fatal
            log.debug("active_live_drain_post_relief_failed: %s", exc)

    return counts
