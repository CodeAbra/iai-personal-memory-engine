from __future__ import annotations

import contextlib
import enum
import errno
from iai_mcp import _flock as fcntl
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from iai_mcp.hippo import _schema

from iai_mcp.crypto import (
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import EMBED_DIM

_log = logging.getLogger(__name__)


class AccessMode(enum.Enum):

    EXCLUSIVE = "exclusive"
    SHARED = "shared"



HNSW_M = int(os.environ.get("IAI_MCP_HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.environ.get("IAI_MCP_HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF = int(os.environ.get("IAI_MCP_HNSW_EF", "50"))
HNSW_SAVE_INTERVAL = int(os.environ.get("IAI_MCP_HNSW_SAVE_INTERVAL", "200"))
RECALL_INDEX_EF = 200
HNSW_RESIZE_HEADROOM: float = 0.85
HNSW_INITIAL_CAPACITY: int = 10_000


_DEFAULT_IAI_ROOT = Path.home() / ".iai-mcp"


def _operator_home() -> Path:
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (KeyError, ImportError, AttributeError):
        return Path.home()


_REAL_IAI_ROOT = _operator_home() / ".iai-mcp"


def _resolve_root(path: str | Path | None = None) -> Path:
    env_path = os.environ.get("IAI_MCP_STORE")
    if env_path:
        return Path(env_path)
    if path is not None:
        return Path(path)
    resolved = _DEFAULT_IAI_ROOT
    if os.environ.get("PYTEST_CURRENT_TEST") and resolved == _REAL_IAI_ROOT:
        raise RuntimeError(
            "hermeticity guard: store-root resolved to the real home store "
            "during a test run; tests must use a tmp path (autouse redirect "
            "fixture). This guard never fires in normal operation."
        )
    return resolved


class HippoLockHeldError(RuntimeError):

    def __init__(self, lock_path: Path | str, holding_pid: str = "unknown") -> None:
        self.lock_path = lock_path
        self.holding_pid = holding_pid
        msg = (
            f"Hippo storage at {Path(lock_path).parent} is already opened by another "
            f"process (pid={holding_pid}); cannot acquire exclusive lock on "
            f"{lock_path}. Stop the daemon or other hippo client first."
        )
        super().__init__(msg)


class ConsolidationPendingError(RuntimeError):

    def __init__(self, lock_path: Path | str) -> None:
        self.lock_path = lock_path
        msg = (
            f"Hippo consolidation in progress at {Path(lock_path).parent}; "
            "SHARED lock not acquired within the <1.5 s SLO. "
            "Retry after the consolidation window."
        )
        super().__init__(msg)


class HippoDecryptError(RuntimeError):
    pass


class HippoIntegrityError(RuntimeError):
    pass


_PROCESS_LOCKS: dict[str, tuple[int, int]] = {}
_PROCESS_LOCKS_SHARED: dict[str, tuple[int, int]] = {}
_PROCESS_LOCKS_GUARD: threading.Lock = threading.Lock()

_SHARED_RETRY_SLEEP_S: float = 0.040
_SHARED_MAX_RETRIES: int = 30
_SHARED_LOCK_TIMEOUT_S: float = 1.45

# Staleness ceiling for the .consolidation-pending intent file. The primary
# liveness discriminator is the holder PID written into the intent: if that PID
# is no longer alive the intent is orphaned (the daemon that set it crashed) and
# is removed on the spot, so a dead daemon can never gate a write. This time
# ceiling is the secondary safety net for the PID-can't-be-read case (legacy or
# truncated intent, or a recycled PID): an intent older than this is treated as
# stale even if some live process happens to hold the same PID. It is set far
# above the longest legitimate consolidation window so a real in-progress
# consolidation is never misjudged as stale.
_CONSOLIDATION_INTENT_TTL_S: float = 900.0


def _pid_alive(pid: int) -> bool:
    from iai_mcp.lifecycle_lock import pid_exists

    return pid_exists(pid)
    return True


def _consolidation_intent_is_stale(intent_path: Path) -> bool:
    """True when the consolidation intent is orphaned and safe to remove.

    A live in-progress consolidation writes its PID (and a timestamp) into the
    intent and holds the EXCLUSIVE flock for the whole window — its PID is alive,
    so this returns False and the direct write correctly defers. A SIGKILL'd
    daemon leaves the regular intent file behind with no live holder: its PID is
    dead (primary signal) or the intent is unreadable/legacy, so this returns
    True and the caller removes the orphan and proceeds.
    """
    try:
        raw = intent_path.read_text()
    except FileNotFoundError:
        return False
    except OSError:
        # Unreadable intent → treat as orphaned rather than wedge forever.
        return True

    lines = raw.split("\n")
    pid_s = lines[0].strip() if lines else ""
    ts_s = lines[1].strip() if len(lines) > 1 else ""

    if pid_s:
        try:
            pid = int(pid_s)
        except ValueError:
            return True
        if not _pid_alive(pid):
            return True
        # Holder PID is alive — only the time ceiling can override (recycled PID).
        if ts_s:
            try:
                age = time.time() - float(ts_s)
            except ValueError:
                return False
            return age > _CONSOLIDATION_INTENT_TTL_S
        return False

    # No PID recorded (legacy empty intent written by an older daemon). Fall back
    # to the mtime ceiling so a crashed legacy daemon still self-heals.
    try:
        age = time.time() - intent_path.stat().st_mtime
    except OSError:
        return True
    return age > _CONSOLIDATION_INTENT_TTL_S


_ENCRYPTED_RECORD_COLUMNS: tuple[str, ...] = (
    "literal_surface",
    "provenance_json",
    "profile_modulation_gain_json",
)

_ENCRYPTED_EVENTS_COLUMNS: tuple[str, ...] = (
    "data_json",
)


from iai_mcp.hippo._table import (
    HippoTableList,
    HippoTable,
    HippoQuery,
    HippoMergeInsert,
    _normalize_to_row_list,
    _encode_embedding,
    _decode_embedding,
    _encode_row_for_insert,
    _decode_df_embedding,
    _decrypt_df_columns,
    _PA_TO_SQLITE,
    _pa_type_to_sqlite,
    _BOOL_COLUMNS,
    _STRICT_BOOL_COLUMNS,
    _sqlite_type_to_pa,
    _DDL_RECORDS,
    _DDL_RECORDS_INDEXES,
    _DDL_EDGES,
    _DDL_EDGES_INDEXES,
    _DDL_EVENTS,
    _DDL_EVENTS_INDEXES,
    _DDL_BUDGET_LEDGER,
    _DDL_BUDGET_LEDGER_INDEXES,
    _DDL_RATELIMIT_LEDGER,
    _DDL_HIPPO_META,
    _DDL_RECORD_TAGS,
    _DDL_RECORD_TAGS_INDEXES,
    _TABLE_SQL,
)



from iai_mcp.hippo._db import (
    _txn,
    _txn_owners,
    _txn_owners_lock,
    _validate_table_name,
    HippoDB,
)

from iai_mcp.hippo._recall import (
    _DIRECT_RECENCY_SQL,
    _DIRECT_RECENCY_SQL_LIMITED,
    _no_flock_recency_rows_from_store,
    reconcile_index_mid_run,
    direct_recency_rows_from_store,
    load_hnsw_readonly,
    _ann_lookup_client,
    degraded_semantic_recall,
)

from iai_mcp.hippo._raw_open import open_store_conn

__all__ = [
    "AccessMode",
    "ConsolidationPendingError",
    "HippoDB",
    "HippoTable",
    "HippoQuery",
    "HippoMergeInsert",
    "HippoTableList",
    "HippoLockHeldError",
    "HippoDecryptError",
    "HippoIntegrityError",
    "_ENCRYPTED_RECORD_COLUMNS",
    "_ENCRYPTED_EVENTS_COLUMNS",
    "direct_recency_rows_from_store",
    "load_hnsw_readonly",
    "_ann_lookup_client",
    "degraded_semantic_recall",
    "open_store_conn",
    "EMBED_DIM",
]
