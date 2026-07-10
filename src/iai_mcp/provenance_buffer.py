from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from iai_mcp.store import MemoryStore

log = logging.getLogger(__name__)

_BUFFER_FILENAME = ".deferred-provenance.jsonl"


def _buffer_path(store: MemoryStore) -> Path:
    return Path(store.root) / _BUFFER_FILENAME


def defer_provenance(
    store: MemoryStore,
    entries: list[tuple[UUID, str, str]],
) -> None:
    if not entries:
        return
    path = _buffer_path(store)
    now_iso = datetime.now(timezone.utc).isoformat()
    lines = []
    for record_id, cue, session_id in entries:
        lines.append(json.dumps({
            "record_id": str(record_id),
            "ts": now_iso,
            "cue": cue,
            "session_id": session_id,
        }))
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def flush_deferred_provenance(store: MemoryStore) -> int:
    path = _buffer_path(store)
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            raw_lines = f.read().strip().splitlines()
    except OSError:
        return 0
    if not raw_lines:
        return 0

    pairs: list[tuple[UUID, dict]] = []
    for line in raw_lines:
        try:
            entry = json.loads(line)
            pairs.append((
                UUID(entry["record_id"]),
                {
                    "ts": entry["ts"],
                    "cue": entry["cue"],
                    "session_id": entry["session_id"],
                },
            ))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue

    if pairs:
        try:
            store.append_provenance_batch(pairs, records_cache=None)
        except Exception as exc:  # noqa: BLE001 -- flush must never crash daemon sleep step
            log.warning("flush_deferred_provenance_failed", extra={"err": str(exc)[:120]})
            return 0

    try:
        path.write_text("")
    except OSError:
        pass
    return len(pairs)
