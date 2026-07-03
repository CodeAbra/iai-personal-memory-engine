from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from iai_mcp.hippo import HippoLockHeldError
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


_TEST_PASSPHRASE = "iai-mcp-test-passphrase-2026-04-30"

K_SEEDS = 5
_CHILD_BARRIER_TIMEOUT_S = 60.0


def _seed_content(i: int) -> str:
    return (
        f"alice pinned fact {i}: the lossless cat sat on durable mat number {i} "
        f"and the verbatim invariant held exactly token-{i}-{i * 7}"
    )


def _make_pinned_seed(i: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=_seed_content(i),
        aaak_index="",
        embedding=[0.0] * i + [1.0] + [0.0] * (EMBED_DIM - i - 1),
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["seed", "lossless-gate"],
        language="en",
    )


_CHILD_PROGRAM = r"""
import os, sys, time
from datetime import datetime, timezone
from uuid import uuid4

src = os.environ["IAI_MCP_TEST_SRC"]
if src not in sys.path:
    sys.path.insert(0, src)

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

store_root = os.environ["IAI_MCP_STORE"]
sentinel = os.environ["IAI_MCP_TEST_SENTINEL"]
n_durable = int(os.environ["IAI_MCP_TEST_CHURN_DURABLE"])
seed_dims = int(os.environ["IAI_MCP_TEST_SEED_DIMS"])


def _make(i):
    now = datetime.now(timezone.utc)
    pos = seed_dims + (i % (EMBED_DIM - seed_dims))
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="alice churn write %d in flight" % i,
        aaak_index="",
        embedding=[0.0] * pos + [1.0] + [0.0] * (EMBED_DIM - pos - 1),
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
        tags=["churn"],
        language="en",
    )


store = MemoryStore(path=store_root)

for i in range(n_durable):
    store.insert(_make(i))
flush_record_buffer(store)

with open(sentinel, "w") as fh:
    fh.write("ready")
    fh.flush()
    os.fsync(fh.fileno())

i = n_durable
while True:
    store.insert(_make(i))
    i += 1
    if i % 20 == 0:
        flush_record_buffer(store)
"""


CHURN_DURABLE = 40


def _spawn_writer_child(store_root: Path, sentinel: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_MCP_TEST_SENTINEL"] = str(sentinel)
    env["IAI_MCP_TEST_SRC"] = str(Path(__file__).resolve().parent.parent / "src")
    env["IAI_MCP_TEST_CHURN_DURABLE"] = str(CHURN_DURABLE)
    env["IAI_MCP_TEST_SEED_DIMS"] = str(K_SEEDS)
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_PASSPHRASE
    env["IAI_MCP_RECONSOLIDATION_TIER1"] = "0"
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD_PROGRAM],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _active_count(store: MemoryStore) -> int:
    row = store.db._conn.execute(
        "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
    ).fetchone()
    return int(row[0]) if row else 0


def test_sigkill_mid_write_leaves_store_lossless():
    tmp_root = Path(tempfile.mkdtemp(prefix="iai-lossless-gate-"))
    sentinel = tmp_root / ".child-ready"
    child: subprocess.Popen | None = None
    seed_ids: list[UUID] = []
    try:
        seed_store = MemoryStore(path=tmp_root)
        for i in range(K_SEEDS):
            rec = _make_pinned_seed(i)
            seed_ids.append(rec.id)
            seed_store.insert(rec)
        seed_store.close()

        child = _spawn_writer_child(tmp_root, sentinel)

        deadline = time.monotonic() + _CHILD_BARRIER_TIMEOUT_S
        while not sentinel.exists():
            if child.poll() is not None:
                out, err = child.communicate()
                raise AssertionError(
                    "writer child exited before signalling readiness "
                    f"(rc={child.returncode}); stderr=\n{err.decode(errors='replace')}"
                )
            if time.monotonic() >= deadline:
                raise AssertionError("writer child never signalled readiness")
            time.sleep(0.01)

        # Windows has no SIGKILL; os.kill(pid, SIGTERM) maps to TerminateProcess,
        # which is the same abrupt no-cleanup kill this test needs to prove the
        # store stays lossless after a hard kill mid-write.
        os.kill(child.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        child.wait(timeout=30)
        child = None

        churn_vec = [0.0] * K_SEEDS + [1.0] + [0.0] * (EMBED_DIM - K_SEEDS - 1)

        # Windows TerminateProcess (os.kill SIGTERM) releases the killed child's
        # byte-range lock on the hippo .lock asynchronously during process
        # teardown, so the reopen can briefly race the lock release. Poll until
        # the exclusive lock is free. First attempt succeeds on POSIX.
        reopened = None
        for _ in range(50):
            try:
                reopened = MemoryStore(path=tmp_root)
                break
            except HippoLockHeldError:
                time.sleep(0.1)
        assert reopened is not None, (
            "hippo lock not released within 5s after killing the writer child"
        )
        try:

            for i, sid in enumerate(seed_ids):
                got = reopened.get(sid)
                assert got is not None, f"seed {i} lost after mid-write SIGKILL"
                assert got.literal_surface == _seed_content(i), (
                    f"seed {i} content corrupted after SIGKILL: "
                    f"{got.literal_surface!r}"
                )

            active = _active_count(reopened)
            assert active > K_SEEDS, (
                "SIGKILL hit dead air (no durable child write committed): "
                f"active={active} <= K={K_SEEDS}. Setup failure, NOT a pass."
            )

            from iai_mcp.daemon import _hippo_health_check_on_boot

            health = _hippo_health_check_on_boot(reopened)
            assert health["sqlite_count"] == active
            raw_at_boot = int(reopened.db._hnsw.get_current_count())

            _run_consolidation_clean(reopened, tmp_root)

            raw_after = int(reopened.db._hnsw.get_current_count())
            assert raw_after == active, (
                f"index not reconciled by consolidation: raw={raw_after} != "
                f"active={active} (raw at boot was {raw_at_boot})"
            )
            assert raw_after >= raw_at_boot, "consolidation shrank the index"
            churn_hits = reopened.query_similar(churn_vec, n=5)
            assert any(r.id not in set(seed_ids) for r in churn_hits), (
                "no surviving churn record resolvable via the index after the "
                "post-kill consolidation rebuild"
            )
            for i, sid in enumerate(seed_ids):
                vec = [0.0] * i + [1.0] + [0.0] * (EMBED_DIM - i - 1)
                hit_ids = {r.id for r in reopened.query_similar(vec, n=3)}
                assert sid in hit_ids, (
                    f"seed {i} not resolvable after the consolidation rebuild"
                )
        finally:
            reopened.close()

        hnsw_path = tmp_root / "hippo" / "records.hnsw"
        if hnsw_path.exists():
            hnsw_path.unlink()
        rebuilt = MemoryStore(path=tmp_root)
        try:
            from iai_mcp.daemon import _hippo_health_check_on_boot

            health2 = _hippo_health_check_on_boot(rebuilt)
            assert health2["sqlite_count"] == health2["hnsw_active_count"], (
                f"index not rebuilt from SQLite after deletion: {health2}"
            )
            assert int(health2["sqlite_count"]) >= K_SEEDS
            for i, sid in enumerate(seed_ids):
                got = rebuilt.get(sid)
                assert got is not None, f"seed {i} lost after index-delete rebuild"
                assert got.literal_surface == _seed_content(i)
                vec = [0.0] * i + [1.0] + [0.0] * (EMBED_DIM - i - 1)
                hit_ids = {r.id for r in rebuilt.query_similar(vec, n=3)}
                assert sid in hit_ids, (
                    f"seed {i} not resolvable via index rebuilt purely from SQLite"
                )
        finally:
            rebuilt.close()
    finally:
        if child is not None:
            try:
                child.kill()
                child.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


def _run_consolidation_clean(store: MemoryStore, tmp_root: Path) -> None:
    import unittest.mock as mock

    import iai_mcp.claude_cli as _cc
    import iai_mcp.reconsolidation_critic as _rc
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    def _raise_remote(*_a, **_k):  # pragma: no cover - must never be reached
        raise AssertionError("remote subprocess must not be invoked under test")

    log_dir = tmp_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_root / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=log_dir),
    )
    with (
        mock.patch.object(_cc, "invoke_claude_sync", _raise_remote),
        mock.patch.object(_cc, "invoke_claude_once", _raise_remote),
        mock.patch.object(_rc, "evaluate_batch_reconsolidation", lambda *_a, **_k: {}),
    ):
        result = pipeline.run()
    assert result.get("error") is None, (
        f"consolidation did not re-run cleanly after SIGKILL: {result.get('error')}"
    )
    assert int(result.get("critic_calls", 0) or 0) == 0, (
        "remote critic fired during the post-kill consolidation run"
    )
