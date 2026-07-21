from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from iai_mcp.hippo import (
    AccessMode,
    ConsolidationPendingError,
    HippoDB,
    HippoLockHeldError,
    _SHARED_LOCK_TIMEOUT_S,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


_TEST_PASSPHRASE = "iai-mcp-test-passphrase-2026-04-30-phase-07.10"
_HOLDER_BARRIER_TIMEOUT_S = 60.0


_HOLDER_PROGRAM = r"""
import os, sys, time

src = os.environ["IAI_MCP_TEST_SRC"]
if src not in sys.path:
    sys.path.insert(0, src)

from iai_mcp.store import MemoryStore

store_root = os.environ["IAI_MCP_STORE"]
sentinel = os.environ["IAI_MCP_TEST_SENTINEL"]

store = MemoryStore(path=store_root)

with open(sentinel, "w") as fh:
    fh.write("held")
    fh.flush()
    os.fsync(fh.fileno())

time.sleep(120)
"""


def _seed_one_record(store_root: Path) -> None:
    store = MemoryStore(path=store_root)
    try:
        now = datetime.now(timezone.utc)
        store.insert(
            MemoryRecord(
                id=uuid4(),
                tier="episodic",
                literal_surface="alice baseline record for the bounded-client check",
                aaak_index="",
                embedding=[0.1] * EMBED_DIM,
                community_id=None,
                centrality=0.0,
                detail_level=2,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[],
                created_at=now,
                updated_at=now,
                tags=["baseline"],
                language="en",
            )
        )
    finally:
        store.close()


def _spawn_exclusive_holder(store_root: Path, sentinel: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_MCP_TEST_SENTINEL"] = str(sentinel)
    env["IAI_MCP_TEST_SRC"] = str(Path(__file__).resolve().parent.parent / "src")
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_PASSPHRASE
    return subprocess.Popen(
        [sys.executable, "-c", _HOLDER_PROGRAM],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_client_shared_read_is_bounded_under_held_lock():
    tmp_root = Path(tempfile.mkdtemp(prefix="iai-bounded-client-"))
    sentinel = tmp_root / ".holder-held"
    holder: subprocess.Popen | None = None
    try:
        _seed_one_record(tmp_root)

        holder = _spawn_exclusive_holder(tmp_root, sentinel)
        deadline = time.monotonic() + _HOLDER_BARRIER_TIMEOUT_S
        while not sentinel.exists():
            if holder.poll() is not None:
                _out, err = holder.communicate()
                raise AssertionError(
                    "exclusive holder exited before signalling "
                    f"(rc={holder.returncode}); stderr=\n{err.decode(errors='replace')}"
                )
            if time.monotonic() >= deadline:
                raise AssertionError("exclusive holder never signalled lock-held")
            time.sleep(0.01)

        bound_s = max(3.0, _SHARED_LOCK_TIMEOUT_S + 1.5)
        t0 = time.monotonic()
        client: HippoDB | None = None
        try:
            client = HippoDB(
                tmp_root,
                access_mode=AccessMode.SHARED,
                read_only=True,
            )
        except (ConsolidationPendingError, HippoLockHeldError):
            pass
        finally:
            if client is not None:
                client.close()
        elapsed = time.monotonic() - t0

        assert elapsed < bound_s, (
            f"client SHARED open took {elapsed:.3f}s (>= bound {bound_s:.3f}s) — "
            "the client blocked instead of degrading within the bound"
        )
    finally:
        if holder is not None:
            try:
                os.kill(holder.pid, signal.SIGKILL)
                holder.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


def test_shared_lock_timeout_constant_is_bounded():
    assert 0 < _SHARED_LOCK_TIMEOUT_S < 1.5
