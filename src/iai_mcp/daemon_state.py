from __future__ import annotations

from iai_mcp import _flock
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

DAEMON_STATE_FILENAME: str = ".daemon-state.json"
DAEMON_STATE_LOCK_FILENAME: str = ".daemon-state.lock"
STATE_PATH: Path = Path.home() / ".iai-mcp" / DAEMON_STATE_FILENAME

DIGEST_SHOW_THRESHOLD_HOURS: int = 18

FIRST_TURN_TTL_HOURS: int = 24
MAX_FIRST_TURN_ENTRIES: int = 100


def daemon_state_path(store_root: Path | str | None = None) -> Path:
    """Resolve the daemon runtime-state file path under the active store root.

    Mirrors ``lifecycle_state_path``: an explicit ``store_root`` wins; absent
    that, the ``IAI_MCP_STORE`` environment variable is honored (read at call
    time, so every caller of ``load_state``/``save_state`` picks up the
    currently active store without a signature change); absent both, the
    module-level ``STATE_PATH`` is returned as the home-rooted default,
    keeping default resolution byte-identical to the pre-existing constant.
    """
    if store_root is not None:
        return Path(store_root) / DAEMON_STATE_FILENAME
    env = os.environ.get("IAI_MCP_STORE")
    if env:
        return Path(env) / DAEMON_STATE_FILENAME
    return STATE_PATH


def load_state() -> dict:
    target = daemon_state_path()
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# Re-entrant per-process guard over the cross-process flock: update_state
# holds the lock across its load-mutate-save, and its inner save_state call
# must not re-acquire the flock through a second fd (same-process flock on a
# fresh fd blocks against itself).
import threading

_state_mutex = threading.RLock()
_state_lock_depth = threading.local()


@contextmanager
def _state_write_lock() -> Iterator[None]:
    depth = getattr(_state_lock_depth, "value", 0)
    if depth > 0:
        _state_lock_depth.value = depth + 1
        try:
            yield
        finally:
            _state_lock_depth.value = depth
        return

    target = daemon_state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _state_mutex:
        lock_fd = os.open(
            str(target.parent / DAEMON_STATE_LOCK_FILENAME),
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        try:
            _flock.flock(lock_fd, _flock.LOCK_EX)
            _state_lock_depth.value = 1
            try:
                yield
            finally:
                _state_lock_depth.value = 0
                try:
                    _flock.flock(lock_fd, _flock.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(lock_fd)


def save_state(state: dict) -> None:
    # Whole-dict replace: legal ONLY for a dict loaded moments ago (boot,
    # tests) or under update_state's lock. A long-held dict saved here
    # silently erases every key another writer persisted since that dict
    # was loaded вЂ” concurrent writers must go through update_state.
    with _state_write_lock():
        target = daemon_state_path()
        fd, tmp = tempfile.mkstemp(
            prefix=".daemon-state.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except (OSError, TypeError, ValueError):
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def update_state(mutator: Callable[[dict], None]) -> dict:
    # The one safe write path under concurrency: EX-flock (cross-process,
    # daemon + CLI), fresh load, targeted mutation, save. Mutators must touch
    # only their own keys. load_state/save_state stay the patchable seams вЂ”
    # tests that stub them see every update_state write.
    with _state_write_lock():
        state = load_state()
        mutator(state)
        save_state(state)
        return state


def prune_stale_first_turn(
    state: dict,
    now: datetime | None = None,
    ttl_hours: int = FIRST_TURN_TTL_HOURS,
    max_entries: int = MAX_FIRST_TURN_ENTRIES,
) -> int:
    pending = state.get("first_turn_pending")
    if not isinstance(pending, dict) or not pending:
        return 0

    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current - timedelta(hours=ttl_hours)

    def _as_dt(value: object) -> datetime:
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                return datetime.fromtimestamp(0, tz=timezone.utc)
        return datetime.fromtimestamp(0, tz=timezone.utc)

    removed = 0
    for sid, value in list(pending.items()):
        dt = _as_dt(value)
        if dt < cutoff:
            pending.pop(sid, None)
            removed += 1
        elif not isinstance(value, str):
            pending[sid] = dt.isoformat()

    if len(pending) > max_entries:
        ordered = sorted(
            pending.items(),
            key=lambda kv: _as_dt(kv[1]),
            reverse=True,
        )
        keep = dict(ordered[:max_entries])
        removed += len(pending) - len(keep)
        state["first_turn_pending"] = keep

    return removed


def mark_session_opened(state: dict, session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id:
        return
    pending = state.setdefault("first_turn_pending", {})
    pending[session_id] = datetime.now(timezone.utc).isoformat()
    prune_stale_first_turn(state)


def consume_first_turn(state: dict, session_id: str) -> bool:
    try:
        pending = state.get("first_turn_pending")
        if not isinstance(pending, dict):
            return False
        if pending.pop(session_id, False):
            try:
                def _drop(d: dict) -> None:
                    fresh = d.get("first_turn_pending")
                    if isinstance(fresh, dict):
                        fresh.pop(session_id, None)

                update_state(_drop)
            except (OSError, TypeError, ValueError):
                pass
            return True
        return False
    except (KeyError, TypeError, AttributeError):
        return False


FIRST_TURN_PENDING_TTL_SEC_DEFAULT: float = 3600.0


def prune_first_turn_pending(
    state: dict,
    now: datetime | None = None,
    ttl_sec: float = FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
) -> tuple[dict, list[str]]:
    pending = state.get("first_turn_pending")
    if not isinstance(pending, dict) or not pending:
        return state, []

    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current - timedelta(seconds=ttl_sec)

    dropped: list[str] = []
    fresh: dict = {}
    for sid, value in pending.items():
        if isinstance(value, str):
            try:
                ts = datetime.fromisoformat(value)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                dropped.append(sid)
                continue
            if ts < cutoff:
                dropped.append(sid)
                continue
            fresh[sid] = value
        else:
            dropped.append(sid)

    state["first_turn_pending"] = fresh
    return state, dropped


def get_pending_digest(state: dict, now: datetime) -> dict | None:
    last_shown = state.get("last_digest_shown_at")
    if last_shown:
        try:
            last_dt = datetime.fromisoformat(last_shown)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            now_cmp = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
            if now_cmp - last_dt < timedelta(hours=DIGEST_SHOW_THRESHOLD_HOURS):
                return None
        except (TypeError, ValueError):
            pass

    digest = state.get("pending_digest")
    if not digest:
        return None

    now_cmp = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    state["last_digest_shown_at"] = now_cmp.isoformat()
    state.pop("pending_digest", None)

    def _consume(d: dict) -> None:
        d["last_digest_shown_at"] = now_cmp.isoformat()
        d.pop("pending_digest", None)

    update_state(_consume)
    return digest

