
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
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


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
) -> dict[str, Any]:
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

    from iai_mcp.embed import embedder_for_store
    from iai_mcp.events import TELEMETRY_EMBED_NATIVE_FAILURE, write_event

    try:
        # Embed the CONTENT, never the cue. capture_transcript / deferred-drain pass
        # a positional cue ("session <id> turn <n>"); embedding that collapsed every
        # stored vector onto near-identical label embeddings, destroying graph
        # structure and semantic recall. The cue stays a provenance label only.
        emb = embedder_for_store(store).embed(text)
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
        else:
            try:
                neighbours = store.query_similar(embedding, k=3, tier=tier)
            except (ValueError, IOError) as exc:
                log.warning(
                    "capture_dedup_query_failed",
                    extra={"err_type": type(exc).__name__, "err": str(exc)[:120]},
                )
                neighbours = []

            for record, score in neighbours:
                if score >= DEDUP_COS_THRESHOLD:
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
                    return {
                        "status": "reinforced",
                        "record_id": str(record.id),
                        "reason": f"cos={score:.3f} >= {DEDUP_COS_THRESHOLD}",
                    }

        tags = ["capture", f"role:{role}"]
        if verdict == "FLAG_FOR_REVIEW":
            tags.append("shield:flagged")
            tags.extend(f"shield:{t}" for t in shield_tags[:3])

        if _is_episodic_conversational(tier, role):
            ts_iso = now.isoformat()
            tags.append(_idem_tag(session_id, role, ts_iso, text, source_uuid=source_uuid))

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
            provenance=[{"ts": now.isoformat(), "cue": cue or "(auto-capture)",
                         "session_id": session_id, "role": role}],
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

    return {"status": "inserted", "record_id": str(rec.id), "reason": f"tier={tier}"}


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
    need_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a") as fh:
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
    out_path = deferred_dir / f"{session_id}-{int(time.time())}.jsonl"
    with out_path.open("w") as fh:
        header = {
            "version": 1,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "cwd": cwd or os.getcwd(),
        }
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        path = Path(transcript_path).expanduser()
        if not path.exists():
            return out_path
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
    return out_path


def drain_deferred_captures(store: MemoryStore) -> dict[str, int]:
    deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
    log_dir = Path.home() / ".iai-mcp" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir / f"deferred-drain-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
    )
    counts = {
        "files_drained": 0,
        "files_failed": 0,
        "events_inserted": 0,
        "events_reinforced": 0,
        "events_skipped_intentional": 0,
        "events_skipped_insert_failed": 0,
    }
    if not deferred_dir.exists():
        return counts
    total_events_processed = 0
    cap_hit = False

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

    for fpath in candidates:
        if cap_hit:
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
            with work_path.open() as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
            if not lines:
                work_path.unlink()
                continue
            header = json.loads(lines[0])
            if header.get("version", 0) > 1:
                with log_path.open("a") as logf:
                    logf.write(
                        f"{datetime.now(timezone.utc).isoformat()} skip "
                        f"{work_path.name}: version={header.get('version')}\n"
                    )
                _strip_processing_marker(work_path, log_path=log_path)
                continue
            session_id = header.get("session_id", "-")
            event_lines = lines[1:]
            processed_in_file = 0
            for idx, ln in enumerate(event_lines):
                if total_events_processed >= MAX_DRAIN_EVENTS_PER_RUN:
                    remainder = event_lines[idx:]
                    work_path, _strip_ok = _strip_processing_marker(
                        work_path, log_path=log_path
                    )
                    if not _strip_ok:
                        cap_hit = True
                        break
                    partial_path = work_path.with_suffix(".partial.jsonl")
                    tmp_path = work_path.with_suffix(".partial.tmp")
                    with tmp_path.open("w") as ph:
                        ph.write(lines[0] + "\n")
                        for r in remainder:
                            ph.write(r + "\n")
                        ph.flush()
                        os.fsync(ph.fileno())
                    os.replace(tmp_path, partial_path)
                    work_path.unlink()
                    counts["files_drained"] += 1
                    cap_hit = True
                    break
                ev = json.loads(ln)
                result = capture_turn(
                    store,
                    cue=ev.get("cue", ""),
                    text=ev.get("text", ""),
                    tier=ev.get("tier", "episodic"),
                    session_id=session_id,
                    role=ev.get("role", "user"),
                    ts=ev.get("ts"),
                    source_uuid=ev.get("source_uuid"),
                )
                status = result.get("status", "skipped")
                reason = result.get("reason", "")
                if status == "inserted":
                    counts["events_inserted"] += 1
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

    return counts
