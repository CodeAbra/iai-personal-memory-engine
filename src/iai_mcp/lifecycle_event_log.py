from __future__ import annotations

import errno
import fcntl
import gzip
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR: Path = Path.home() / ".iai-mcp" / "logs"

KNOWN_EVENT_KINDS: frozenset[str] = frozenset(
    {
        "state_transition",
        "wrapper_event",
        "shadow_run_warning",
        "sleep_step_started",
        "sleep_step_completed",
        "quarantine_entered",
        "quarantine_lifted",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_date_string(dt: datetime | None = None) -> str:
    moment = dt if dt is not None else _utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


class LifecycleEventLog:

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir if log_dir is not None else DEFAULT_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)


    def file_for_date(self, date_str: str) -> Path:
        return self._log_dir / f"lifecycle-events-{date_str}.jsonl"

    def current_file(self, now: datetime | None = None) -> Path:
        return self.file_for_date(_utc_date_string(now))


    def append(self, event: dict[str, Any], now: datetime | None = None) -> None:
        if not isinstance(event, dict):
            raise TypeError(
                f"event must be a dict, got {type(event).__name__}"
            )
        kind = event.get("event")
        if not isinstance(kind, str) or not kind:
            raise ValueError("event['event'] must be a non-empty string")

        moment = now if now is not None else _utc_now()
        if "ts" not in event:
            event = {"ts": moment.astimezone(timezone.utc).isoformat(), **event}

        line = json.dumps(event, separators=(",", ":")) + "\n"
        target = self.current_file(moment)
        target.parent.mkdir(parents=True, exist_ok=True)

        fd = os.open(
            str(target),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


    def rotate_old_files(
        self,
        retention_days: int = 30,
        now: datetime | None = None,
    ) -> int:
        moment = now if now is not None else _utc_now()
        cutoff_date = (moment - timedelta(days=retention_days)).date()

        compressed = 0
        for path in self._log_dir.glob("lifecycle-events-*.jsonl"):
            stem = path.stem
            try:
                date_part = stem.rsplit("-", 3)[-3:]
                file_date = datetime.strptime(
                    "-".join(date_part), "%Y-%m-%d"
                ).date()
            except (ValueError, IndexError):
                continue
            if file_date > cutoff_date:
                continue

            gz_path = path.with_suffix(".jsonl.gz")
            if gz_path.exists():
                continue
            try:
                with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.chmod(gz_path, 0o600)
                os.unlink(path)
                compressed += 1
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EPERM):
                    continue
                raise
        return compressed


    def read_all(self, date_str: str | None = None) -> list[dict[str, Any]]:
        target = self.file_for_date(
            date_str if date_str is not None else _utc_date_string()
        )
        if not target.exists():
            return []
        out: list[dict[str, Any]] = []
        with target.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
