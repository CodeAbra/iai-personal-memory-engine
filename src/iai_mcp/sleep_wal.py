from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

logger = logging.getLogger(__name__)

WAL_FILENAME = ".sleep-wal.jsonl"
TOMBSTONE_TTL_DAYS = 7

OperationType = Literal[
    "tombstone", "edge_prune", "consolidate_merge", "optimize_drop"
]


def _wal_path() -> Path:
    store_dir = os.environ.get("IAI_MCP_STORE", os.path.expanduser("~/.iai-mcp"))
    return Path(store_dir) / WAL_FILENAME


def is_dry_run() -> bool:
    return os.environ.get("IAI_MCP_ERASURE_DRY_RUN", "").lower() in ("1", "true", "yes")


class WALEntry:
    __slots__ = ("id", "operation", "target_ids", "ts", "status", "metadata")

    def __init__(
        self,
        operation: OperationType,
        target_ids: list[str],
        metadata: dict | None = None,
    ):
        self.id = str(uuid4())
        self.operation = operation
        self.target_ids = target_ids
        self.ts = datetime.now(timezone.utc).isoformat()
        self.status: Literal["pending", "committed", "rolled_back"] = "pending"
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "operation": self.operation,
            "target_ids": self.target_ids,
            "ts": self.ts,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WALEntry":
        entry = cls.__new__(cls)
        entry.id = d["id"]
        entry.operation = d["operation"]
        entry.target_ids = d.get("target_ids", [])
        entry.ts = d["ts"]
        entry.status = d.get("status", "pending")
        entry.metadata = d.get("metadata", {})
        return entry


class SleepWAL:

    def __init__(self, path: Path | None = None):
        self.path = path or _wal_path()
        self._dry_run = is_dry_run()

    def begin(
        self,
        operation: OperationType,
        target_ids: list[str],
        metadata: dict | None = None,
    ) -> WALEntry:
        entry = WALEntry(operation, target_ids, metadata)
        self._append(entry)
        if self._dry_run:
            logger.info("DRY-RUN: would %s %d targets", operation, len(target_ids))
        return entry

    def commit(self, entry: WALEntry) -> None:
        entry.status = "committed"
        self._append(entry)

    def rollback(self, entry: WALEntry) -> None:
        entry.status = "rolled_back"
        self._append(entry)

    def pending_entries(self) -> list[WALEntry]:
        if not self.path.exists():
            return []
        entries: dict[str, WALEntry] = {}
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        e = WALEntry.from_dict(d)
                        entries[e.id] = e
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            return []
        return [e for e in entries.values() if e.status == "pending"]

    def cleanup(self, max_age_hours: int = 168) -> int:
        if not self.path.exists():
            return 0
        cutoff = time.time() - (max_age_hours * 3600)
        kept: list[str] = []
        removed = 0
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        ts = datetime.fromisoformat(d["ts"]).timestamp()
                        if d.get("status") != "pending" and ts < cutoff:
                            removed += 1
                            continue
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                    kept.append(line)
            if removed > 0:
                self.path.write_text("\n".join(kept) + "\n" if kept else "")
        except OSError:
            pass
        return removed

    def _append(self, entry: WALEntry) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("WAL write failed: %s", exc)
