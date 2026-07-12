"""Transcript timestamp re-derivation migration.

Re-derives per-turn ``created_at`` values from on-disk session transcripts
for records whose timestamps collapsed to a single shared value.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from iai_mcp.events import write_event
from iai_mcp.store import (
    MemoryStore,
)

from iai_mcp.migrate import _progress_read, _progress_write, _progress_clear


log = logging.getLogger(__name__)


def _find_transcript_ts(
    session_id: str,
    source_uuid: str | None,
    literal_surface: str,
    transcript_root: Path,
) -> "datetime | None":
    """Return the parsed transcript timestamp for a record, or None if unresolvable.

    Scans all JSONL files under transcript_root matching */<session_id>.jsonl.
    Fast path: match by source_uuid. Fallback: match by content hash of literal_surface
    against the transcript line text field.
    """
    from iai_mcp.capture import MAX_CAPTURE_LEN, _resolve_ts

    # Validate session_id to prevent path traversal.
    if not session_id or "/" in session_id or ".." in session_id:
        return None

    pattern = f"*/{session_id}.jsonl"
    matches = list(transcript_root.glob(pattern))
    if not matches:
        return None

    import hashlib

    surface_hash = hashlib.sha256(literal_surface.encode("utf-8")).hexdigest()

    for transcript_path in matches:
        try:
            with transcript_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    ts_str = obj.get("timestamp")
                    if not ts_str:
                        continue
                    # Fast path: uuid match.
                    if source_uuid and obj.get("uuid") == source_uuid:
                        return _resolve_ts(ts_str)
                    # Content-hash fallback: real transcript lines nest the
                    # text under message.content, not at the top level — mirror
                    # the extraction capture_transcript_into_episodic() uses so
                    # the hash matches what was actually stored as literal_surface.
                    msg = obj.get("message")
                    msg = msg if isinstance(msg, dict) else obj
                    # Internal events (e.g. queue-operation) can carry the same
                    # text as the real user/assistant turn but appear earlier in
                    # the file; skip them so the real turn's timestamp wins.
                    role = obj.get("type") or msg.get("role", "")
                    if role not in {"user", "assistant"}:
                        continue
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                parts.append(block.get("text") or "")
                        text_candidate = "\n".join(parts).strip()
                    else:
                        text_candidate = str(content or "").strip()
                    if len(text_candidate) > MAX_CAPTURE_LEN:
                        text_candidate = text_candidate[:MAX_CAPTURE_LEN]
                    if text_candidate:
                        candidate_hash = hashlib.sha256(
                            text_candidate.encode("utf-8")
                        ).hexdigest()
                        if candidate_hash == surface_hash:
                            return _resolve_ts(ts_str)
        except (OSError, UnicodeDecodeError):
            continue

    return None


def migrate_rederive_collapsed_timestamps(
    store: "MemoryStore",
    *,
    dry_run: bool = False,
    transcript_root: "Path | None" = None,
) -> dict:
    """Re-derive per-turn created_at from on-disk transcripts for records
    whose timestamps collapsed to a single shared value.

    Returns a dict with keys: records_updated, skipped_no_transcript,
    skipped_no_match, dry_run.

    Safe to call multiple times (idempotent). Records whose transcripts are
    absent or unmatched are never modified.  Only created_at is updated —
    literal_surface and provenance_json are never touched.
    """
    from iai_mcp.hippo import HippoDB

    if transcript_root is None:
        transcript_root = Path.home() / ".claude" / "projects"
    else:
        transcript_root = Path(transcript_root)

    db = store.db
    if not isinstance(db, HippoDB):
        return {
            "records_updated": 0,
            "skipped_no_transcript": 0,
            "skipped_no_match": 0,
            "dry_run": dry_run,
        }

    # Load collapsed-group candidates: episodic records sharing created_at
    # with at least 2 other records (group size >= 3).
    with db._conn_lock:
        rows = db._conn.execute(
            "SELECT id, created_at FROM records"
            " WHERE tier = 'episodic'"
            "   AND tombstoned_at IS NULL"
            " GROUP BY created_at"
            " HAVING COUNT(*) >= 3"
        ).fetchall()

    if not rows:
        return {
            "records_updated": 0,
            "skipped_no_transcript": 0,
            "skipped_no_match": 0,
            "dry_run": dry_run,
        }

    # Collect all record IDs in collapsed groups.
    candidate_created_ats = {row[1] for row in rows}
    with db._conn_lock:
        all_candidates = db._conn.execute(
            "SELECT id FROM records"
            " WHERE tier = 'episodic'"
            "   AND tombstoned_at IS NULL"
            "   AND created_at IN ({})".format(
                ",".join("?" * len(candidate_created_ats))
            ),
            list(candidate_created_ats),
        ).fetchall()

    record_ids = [row[0] for row in (all_candidates or [])]

    progress = _progress_read(store)
    done_ids: set[str] = set(progress.get("done_ids", []))

    records_updated = 0
    skipped_no_transcript = 0
    skipped_no_match = 0

    for rec_id_str in record_ids:
        if rec_id_str in done_ids:
            continue

        try:
            from uuid import UUID
            rec = store.get(UUID(rec_id_str))
        except (ValueError, Exception):
            skipped_no_match += 1
            continue

        if rec is None:
            skipped_no_match += 1
            continue

        prov = (rec.provenance or [{}])[0]
        session_id = prov.get("session_id") or ""
        source_uuid = prov.get("source_uuid") or None

        if not session_id:
            skipped_no_transcript += 1
            done_ids.add(rec_id_str)
            continue

        # Check whether any transcript file exists for this session.
        if not session_id or "/" in session_id or ".." in session_id:
            skipped_no_transcript += 1
            done_ids.add(rec_id_str)
            continue

        transcript_matches = list(transcript_root.glob(f"*/{session_id}.jsonl"))
        if not transcript_matches:
            skipped_no_transcript += 1
            done_ids.add(rec_id_str)
            continue

        ts = _find_transcript_ts(
            session_id=session_id,
            source_uuid=source_uuid,
            literal_surface=rec.literal_surface,
            transcript_root=transcript_root,
        )

        if ts is None:
            skipped_no_match += 1
            done_ids.add(rec_id_str)
            continue

        if not dry_run:
            with db._conn_lock:
                db._conn.execute(
                    "UPDATE records SET created_at = ? WHERE id = ?",
                    (ts, rec_id_str),
                )
            records_updated += 1
        else:
            records_updated += 1

        done_ids.add(rec_id_str)

        if not dry_run:
            _progress_write(
                store,
                {"done_ids": list(done_ids)},
            )

    if not dry_run:
        try:
            write_event(
                store,
                "migration_rederive_timestamps",
                {
                    "records_updated": records_updated,
                    "skipped_no_transcript": skipped_no_transcript,
                    "skipped_no_match": skipped_no_match,
                },
            )
        except (OSError, ValueError, RuntimeError) as exc:
            log.error("migration_rederive_timestamps event write failed: %s", exc)

        _progress_clear(store)

    return {
        "records_updated": records_updated,
        "skipped_no_transcript": skipped_no_transcript,
        "skipped_no_match": skipped_no_match,
        "dry_run": dry_run,
    }
