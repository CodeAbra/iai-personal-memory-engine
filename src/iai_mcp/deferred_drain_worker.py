"""Spawn-context worker that drains one byte-bounded batch of deferred-capture
files into the store.

The parent sends only file paths and a store-root path over the pipe. The child
constructs its OWN ``MemoryStore`` and derives the at-rest encryption key
in-process from the on-disk key material — no key bytes ever cross the process
boundary. Each child drains its batch via ``capture_turn`` (forwarding every
event's ``source_uuid`` and ``ts`` unchanged so a re-drain reinforces instead of
double-inserting), reports its own counts and PID, and exits so the OS reclaims
its peak resident set between batches.

Crash-rotation backlogs are dominated by duplicates: the same conversations are
re-queued every time the writer dies, so the overwhelming majority of enumerated
events already exist in the store. ``capture_turn`` embeds (the expensive Rust
BERT step) BEFORE its idem-tag dedup check, so draining a duplicate-heavy
backlog through it alone would embed every duplicate only to discard it — and
would re-reinforce each already-present record once per duplicate, corrupting the
reinforcement signal. The per-event loop therefore runs a cheap pre-embed
idem-tag check for conversational episodic events: if the tag already exists
(in this run or in the store), the event is skipped before any embedding. The
tag is computed with the same helpers ``capture_turn`` uses, so it matches
exactly; ``capture_turn``'s own idem-check remains the correctness backstop, so
the pre-check can only save work, never drop a genuinely-new record.

Top-level import surface is only ``from __future__ import annotations``. All
compute-side imports live inside the worker body so the spawn-context child's
import does not drag store/capture modules at module-import time.
"""

from __future__ import annotations


def _drain_files(store, paths) -> dict:  # noqa: ANN001
    """Drain a list of backlog files into an open store; return the count tally.

    Runs the cheap pre-embed idem-tag skip for conversational episodic events,
    then forwards the genuinely-new (and all non-episodic) events to
    ``capture_turn``. Counts are mutually exclusive and partition every enumerated
    event: ``inserted + reinforced + skipped_existing + skipped == events_seen``.
    Each file is unlinked only after it is fully drained.
    """
    import json
    from pathlib import Path

    from iai_mcp.capture import (
        MAX_CAPTURE_LEN,
        MIN_CAPTURE_LEN,
        SpoolKeyUnavailable,
        _decode_spool_line,
        _idem_tag,
        _is_episodic_conversational,
        _resolve_ts,
        capture_turn,
    )

    inserted = 0
    reinforced = 0
    skipped = 0
    skipped_existing = 0
    events_seen = 0
    seen_this_run: set[str] = set()

    for p in paths:
        fpath = Path(p)
        try:
            with fpath.open("r", encoding="utf-8") as fh:
                lines = [ln for ln in fh if ln.strip()]
        except OSError:
            continue
        if not lines:
            fpath.unlink(missing_ok=True)
            continue

        try:
            header = json.loads(_decode_spool_line(lines[0]))
        except SpoolKeyUnavailable:
            # No readable key in this process — leave the file untouched for
            # a pass that has one; unlinking here would destroy the turns.
            continue
        except (ValueError, TypeError):
            # Malformed header — skip the whole file rather than abort the batch.
            continue
        session_id = header.get("session_id", "-")

        key_deferred = False
        for ln in lines[1:]:
            events_seen += 1
            try:
                ev = json.loads(_decode_spool_line(ln))
            except SpoolKeyUnavailable:
                key_deferred = True
                break
            except (ValueError, TypeError):
                skipped += 1
                continue

            tier = ev.get("tier", "episodic")
            role = ev.get("role", "user")

            # Cheap pre-embed idem skip for conversational episodic events. The
            # tag is reproduced exactly as capture_turn computes it: from the
            # stripped, length-bounded text and the resolved timestamp. A
            # too-short event would never become a record (capture_turn skips it
            # before embedding), so it can never match a stored tag and falls
            # through to capture_turn, which returns its own "too short" skip.
            if _is_episodic_conversational(tier, role):
                norm_text = (ev.get("text", "") or "").strip()
                if MIN_CAPTURE_LEN <= len(norm_text):
                    if len(norm_text) > MAX_CAPTURE_LEN:
                        norm_text = norm_text[:MAX_CAPTURE_LEN]
                    ts_iso = _resolve_ts(ev.get("ts")).isoformat()
                    tag = _idem_tag(
                        session_id,
                        role,
                        ts_iso,
                        norm_text,
                        source_uuid=ev.get("source_uuid"),
                    )
                    if tag in seen_this_run or store.find_record_by_tag(tag) is not None:
                        skipped_existing += 1
                        seen_this_run.add(tag)
                        continue
                    seen_this_run.add(tag)

            result = capture_turn(
                store,
                cue=ev.get("cue", ev.get("text", "")),
                text=ev.get("text", ""),
                tier=tier,
                session_id=session_id,
                role=role,
                ts=ev.get("ts"),
                source_uuid=ev.get("source_uuid"),
                # This is the ambient transcript-batch drain: the drained
                # event's role/text are a verbatim record of the human's
                # actual turn, not an assistant-composed string. This is the
                # ONE call site allowed to mint a directive from the typed
                # marker.
                directive_marker_allowed=True,
            )
            status = result.get("status")
            if status == "inserted":
                inserted += 1
            elif status == "reinforced":
                reinforced += 1
            else:
                skipped += 1

        if key_deferred:
            continue
        # Unlink only after the file is fully drained — the source_uuid
        # idem-tag makes a re-run loss-free if a crash interrupts here.
        fpath.unlink(missing_ok=True)

    return {
        "inserted": inserted,
        "reinforced": reinforced,
        "skipped": skipped,
        "skipped_existing": skipped_existing,
        "events_seen": events_seen,
    }


def _worker_entry(conn) -> None:  # noqa: ANN001
    """Drain a batch of backlog files into a freshly-constructed store.

    Protocol:
      Parent sends:  ("files", {"store_root": str, "paths": [str, ...]})
      Worker sends:  ("done", {"inserted": int, "reinforced": int,
                               "skipped": int, "skipped_existing": int,
                               "events_seen": int, "pid": int})
                     ("error", str) on failure
    """
    import os
    import sys
    from pathlib import Path

    try:
        kind, payload = conn.recv()
        if kind != "files":
            conn.send(("error", f"expected 'files', got {kind!r}"))
            return
    except (BrokenPipeError, EOFError):
        return  # parent died before sending the batch — exit silently

    from iai_mcp.store import MemoryStore

    store_root = payload["store_root"]
    paths = payload["paths"]

    store = None
    try:
        store = MemoryStore(path=Path(store_root))
        tally = _drain_files(store, paths)
        tally["pid"] = os.getpid()
        conn.send(("done", tally))
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc!s}"[:500]))
        except (BrokenPipeError, OSError):
            pass
        sys.exit(1)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
