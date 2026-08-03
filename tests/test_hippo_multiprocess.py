from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


_LILLI_DRIVER = os.environ.get("LILLI_STORAGE_DRIVER", "stdlib").lower() == "lilli"



_WRITER_SCRIPT = textwrap.dedent("""\
    import os, sys
    from pathlib import Path

    store_root = Path(os.environ["IAI_MCP_STORE"])
    n = int(os.environ.get("IAI_TEST_N_RECORDS", "5"))

    from iai_mcp.direct_write import write_turn_direct

    ok = 0
    for i in range(n):
        result = write_turn_direct(
            store_root,
            text=f"multiprocess writer record {i}",
            session_id="mp-session",
            role="user",
            deferred_embedding=True,
        )
        if result.get("status") in ("inserted", "reinforced"):
            ok += 1
        if i == 0:
            ready = os.environ.get("IAI_TEST_READY_FILE")
            if ready:
                Path(ready).write_text("ready")

    print(f"writer: inserted {ok} records")
    sys.exit(0)
""")

_READER_SCRIPT = textwrap.dedent("""\
    import os, sys
    from pathlib import Path

    store_root = Path(os.environ["IAI_MCP_STORE"])

    from iai_mcp.hippo import AccessMode, HippoDB
    from iai_mcp.store import MemoryStore

    store = MemoryStore(store_root, access_mode=AccessMode.SHARED, read_only=True)
    try:
        records = store.all_records()
        print(f"reader: saw {len(records)} records")
        sys.exit(0)
    finally:
        store.close()
""")

_HNSW_CONCURRENT_LOAD_SCRIPT = textwrap.dedent("""\
    import os, sys
    from pathlib import Path

    store_root = Path(os.environ["IAI_MCP_STORE"])

    from iai_mcp.hippo import load_hnsw_readonly, EMBED_DIM

    idx = load_hnsw_readonly(store_root, EMBED_DIM)
    if idx is None:
        print("hnsw_load: index absent (no .hnsw file)")
        sys.exit(1)
    count = idx.get_current_count()
    print(f"hnsw_load: ok count={count}")
    sys.exit(0)
""")

_HNSW_ATOMIC_SAVE_SCRIPT = textwrap.dedent("""\
    import os, sys, time
    from pathlib import Path

    store_root = Path(os.environ["IAI_MCP_STORE"])

    from iai_mcp.store import MemoryStore

    store = MemoryStore(store_root)
    try:
        time.sleep(0.05)
        import uuid, numpy as np
        from datetime import datetime, timezone
        from iai_mcp.types import EMBED_DIM, MemoryRecord
        rng = np.random.RandomState(seed=999)
        vec = rng.randn(EMBED_DIM).tolist()
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface="daemon-role atomic save probe",
            aaak_index="",
            embedding=vec,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "daemon-save", "role": "user"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["role:user"],
            language="en",
        )
        store.insert(rec)
        print("atomic_save: inserted record ok")
        sys.exit(0)
    finally:
        store.close()
""")


def _child_env(store_root: Path, tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(tmp_path / "no-such.sock")
    env["HOME"] = str(tmp_path)
    return env


def _insert_seed_records(store_root: Path, n: int, seed_base: int = 200) -> None:
    import uuid as _uuid
    import numpy as np
    from datetime import datetime, timezone
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore(store_root)
    try:
        for i in range(n):
            rng = np.random.RandomState(seed=seed_base + i)
            vec = rng.randn(EMBED_DIM).tolist()
            rec = MemoryRecord(
                id=_uuid.uuid4(),
                tier="episodic",
                literal_surface=f"concurrent seed record {i}",
                aaak_index="",
                embedding=vec,
                community_id=None,
                centrality=0.0,
                detail_level=1,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[{"session_id": "seed-session", "role": "user"}],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=["role:user"],
                language="en",
            )
            store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()


def test_multiprocess_serial_write_then_read_no_corruption(
    hermetic_store: Path, tmp_path: Path
) -> None:
    # Verifies post-commit final state: writer completes first, then reader opens.
    # Sequentialized for determinism (the reader's fixed-count assert only makes
    # sense after the writer has committed all records). Does NOT cover WAL
    # cross-process overlap; see test_multiprocess_concurrent_write_read_wal_safe
    # for that.
    n_records = 5
    env = _child_env(hermetic_store, tmp_path)
    env["IAI_TEST_N_RECORDS"] = str(n_records)

    writer = subprocess.Popen(
        [sys.executable, "-c", _WRITER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    writer_out, writer_err = writer.communicate(timeout=30)

    reader = subprocess.Popen(
        [sys.executable, "-c", _READER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    reader_out, reader_err = reader.communicate(timeout=30)

    assert writer.returncode == 0, (
        f"writer process failed (rc={writer.returncode}):\n{writer_err}"
    )
    assert reader.returncode == 0, (
        f"reader process failed (rc={reader.returncode}):\n{reader_err}"
    )
    assert "HippoIntegrityError" not in writer_err, f"writer: HippoIntegrityError\n{writer_err}"
    assert "HippoIntegrityError" not in reader_err, f"reader: HippoIntegrityError\n{reader_err}"
    assert "malformed" not in reader_err.lower(), f"reader: SQLite malformed\n{reader_err}"

    assert f"inserted {n_records}" in writer_out, f"writer output unexpected:\n{writer_out}"
    assert "reader: saw" in reader_out, f"reader output unexpected:\n{reader_out}"

    from iai_mcp.store import MemoryStore
    from iai_mcp.hippo import AccessMode
    store = MemoryStore(hermetic_store, access_mode=AccessMode.SHARED, read_only=True)
    try:
        all_recs = store.all_records()
        assert len(all_recs) == n_records, (
            f"Expected {n_records} records after writer completed, got {len(all_recs)}"
        )
        pending = [r for r in all_recs if getattr(r, "embedding_pending", False)]
        assert len(pending) == n_records, (
            f"Expected all {n_records} records to be embedding_pending; got {len(pending)}"
        )
    finally:
        store.close()


@pytest.mark.skipif(
    _LILLI_DRIVER,
    reason=(
        "asserts a second process opens the store concurrently while a writer "
        "holds it — stdlib sqlite3 WAL admits concurrent readers alongside one "
        "writer. The lilli engine enforces a single-writer exclusive advisory "
        "file lock (the single-writer durability invariant): a concurrent "
        "second open across processes correctly fails with 'store is locked'. "
        "That is expected-correct engine behavior, not a defect. Cross-process "
        "serial open and same-process multi-open are covered by "
        "test_multiprocess_serial_write_then_read_no_corruption and the "
        "registry-reuse path."
    ),
)
def test_multiprocess_concurrent_write_read_wal_safe(
    hermetic_store: Path, tmp_path: Path
) -> None:
    # Restores concurrent-overlap WAL coverage: writer and reader run genuinely
    # overlapping in separate processes. Asserts only the weaker invariants
    # (no crash, no integrity error, no corruption) — NOT a fixed final record
    # count, since the reader may open before or after individual commits.
    # The MCP wrapper reads while the daemon writes — exactly this overlap.
    n_records = 5
    env = _child_env(hermetic_store, tmp_path)
    env["IAI_TEST_N_RECORDS"] = str(n_records)
    ready_file = tmp_path / "writer-ready"
    env["IAI_TEST_READY_FILE"] = str(ready_file)

    writer = subprocess.Popen(
        [sys.executable, "-c", _WRITER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Open the reader only after the writer's FIRST commit — the overlap under
    # test is read-during-write, not read-before-schema-creation. A fixed sleep
    # loses that race under load: the writer subprocess may not even finish its
    # imports in time.
    deadline = time.monotonic() + 30
    while not ready_file.exists():
        if writer.poll() is not None or time.monotonic() > deadline:
            break
        time.sleep(0.01)

    reader = subprocess.Popen(
        [sys.executable, "-c", _READER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    writer_out, writer_err = writer.communicate(timeout=30)
    reader_out, reader_err = reader.communicate(timeout=30)

    assert writer.returncode == 0, (
        f"concurrent writer failed (rc={writer.returncode}):\n{writer_err}"
    )
    assert reader.returncode == 0, (
        f"concurrent reader failed (rc={reader.returncode}):\n{reader_err}"
    )
    assert "HippoIntegrityError" not in writer_err, (
        f"concurrent writer: HippoIntegrityError\n{writer_err}"
    )
    assert "HippoIntegrityError" not in reader_err, (
        f"concurrent reader: HippoIntegrityError\n{reader_err}"
    )
    assert "malformed" not in reader_err.lower(), (
        f"concurrent reader: SQLite malformed\n{reader_err}"
    )
    assert "reader: saw" in reader_out, (
        f"concurrent reader output unexpected:\n{reader_out}"
    )


def test_hnswlib_concurrent_load_index_no_error(
    hermetic_store: Path, tmp_path: Path
) -> None:
    _insert_seed_records(hermetic_store, n=3, seed_base=200)

    hnsw_path = hermetic_store / "hippo" / "records.hnsw"
    assert hnsw_path.exists(), "records.hnsw must exist after seeding for A1 test"

    env = _child_env(hermetic_store, tmp_path)

    p1 = subprocess.Popen(
        [sys.executable, "-c", _HNSW_CONCURRENT_LOAD_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    p2 = subprocess.Popen(
        [sys.executable, "-c", _HNSW_CONCURRENT_LOAD_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    out1, err1 = p1.communicate(timeout=30)
    out2, err2 = p2.communicate(timeout=30)

    assert p1.returncode == 0, f"concurrent load_index process 1 failed (rc={p1.returncode}):\n{err1}"
    assert p2.returncode == 0, f"concurrent load_index process 2 failed (rc={p2.returncode}):\n{err2}"
    assert "hnsw_load: ok" in out1, f"process 1 output unexpected:\n{out1}"
    assert "hnsw_load: ok" in out2, f"process 2 output unexpected:\n{out2}"

    from iai_mcp.hippo import load_hnsw_readonly, EMBED_DIM
    held_idx = load_hnsw_readonly(hermetic_store, EMBED_DIM)
    assert held_idx is not None, "Seeded index must be loadable for Part 2"

    daemon_saver = subprocess.Popen(
        [sys.executable, "-c", _HNSW_ATOMIC_SAVE_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    held_count = held_idx.get_current_count()
    assert held_count >= 3, f"held_idx.get_current_count() should be ≥ 3, got {held_count}"

    daemon_out, daemon_err = daemon_saver.communicate(timeout=30)
    assert daemon_saver.returncode == 0, (
        f"daemon-role saver failed (rc={daemon_saver.returncode}):\n{daemon_err}"
    )
    assert "atomic_save: inserted record ok" in daemon_out, (
        f"daemon_saver output unexpected:\n{daemon_out}"
    )

    assert held_idx.get_current_count() >= 3, (
        "held_idx.get_current_count() degraded after concurrent atomic save"
    )


def test_reader_never_loads_hnsw_tmp(hermetic_store: Path, tmp_path: Path) -> None:
    _insert_seed_records(hermetic_store, n=1, seed_base=300)

    hnsw_tmp = hermetic_store / "hippo" / "records.hnsw.tmp"
    hnsw_tmp.write_bytes(b"CORRUPT_SENTINEL")

    env = _child_env(hermetic_store, tmp_path)

    _READONLY_LOAD_SCRIPT = textwrap.dedent("""\
        import os, sys
        from pathlib import Path

        store_root = Path(os.environ["IAI_MCP_STORE"])

        from iai_mcp.hippo import load_hnsw_readonly, EMBED_DIM

        idx = load_hnsw_readonly(store_root, EMBED_DIM)
        if idx is None:
            print("hnsw_load: FAILED (index is None — corrupt .hnsw.tmp may have been loaded)")
            sys.exit(1)
        count = idx.get_current_count()
        print(f"hnsw_load: ok count={count}")
        sys.exit(0)
    """)

    _CONCURRENT_WRITER_SCRIPT = textwrap.dedent("""\
        import os, sys, time
        from pathlib import Path
        store_root = Path(os.environ["IAI_MCP_STORE"])
        from iai_mcp.store import MemoryStore
        store = MemoryStore(store_root)
        try:
            time.sleep(0.3)
            print("concurrent_writer: ok")
        finally:
            store.close()
    """)

    writer = subprocess.Popen(
        [sys.executable, "-c", _CONCURRENT_WRITER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    time.sleep(0.05)

    reader = subprocess.Popen(
        [sys.executable, "-c", _READONLY_LOAD_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    writer_out, writer_err = writer.communicate(timeout=15)
    reader_out, reader_err = reader.communicate(timeout=15)

    assert writer.returncode == 0, f"writer failed:\n{writer_err}"
    assert reader.returncode == 0, (
        f"reader failed (rc={reader.returncode}):\n{reader_err}\n"
        "load_hnsw_readonly must load records.hnsw independently of flock."
    )
    assert "hnsw_load: ok" in reader_out, f"reader output unexpected:\n{reader_out}"
    assert "CORRUPT" not in reader_err, f"reader may have loaded corrupt .hnsw.tmp:\n{reader_err}"
    assert "FAILED" not in reader_out, f"reader reported failure:\n{reader_out}"
