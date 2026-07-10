from __future__ import annotations

import concurrent.futures
import platform
import threading
from datetime import datetime, timezone
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="threading + POSIX semantics",
)


@pytest.fixture
def iai_home_conc(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    yield tmp_path


def _make_record(session_id: str, i: int):
    from iai_mcp.types import MemoryRecord, EMBED_DIM
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"concurrency test record {i} session {session_id}",
        aaak_index="",
        embedding=[float(i % 100) / 100.0] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": session_id, "role": "user"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["role:user", f"conc-test-{i}"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
    )


def test_concurrent_insert_and_all_records_no_truncation(iai_home_conc, monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=iai_home_conc)
    tbl = store.db.open_table("records")

    N = 30
    session_id = "conc-test-q2"
    records = [_make_record(session_id, i) for i in range(N)]

    rows = [store._to_row(r) for r in records]

    errors: list[str] = []
    truncation_min = [N]

    def write_batch(row_list):
        try:
            tbl.add(row_list)
        except Exception as e:
            errors.append(f"write error: {type(e).__name__}: {e}")

    stop_reader = threading.Event()

    def read_loop():
        import time
        while not stop_reader.is_set():
            try:
                all_rec = store.all_records()
                cnt = len(all_rec)
                if 0 < cnt < truncation_min[0]:
                    truncation_min[0] = cnt
            except Exception as e:
                errors.append(f"read error: {type(e).__name__}: {e}")
            time.sleep(0.0001)

    reader_thread = threading.Thread(target=read_loop, daemon=True)
    reader_thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futs = [executor.submit(write_batch, [row]) for row in rows]
        concurrent.futures.wait(futs)

    stop_reader.set()
    reader_thread.join(timeout=5.0)

    assert not errors, f"Concurrent thread errors: {errors}"

    final = store.all_records()
    session_records = [
        r for r in final
        if (r.provenance or [{}])[0].get("session_id") == session_id
    ]
    assert len(session_records) == N, (
        f"Expected {N} records after all inserts; got {len(session_records)}. "
        f"Min observed during concurrent reads: {truncation_min[0]}"
    )
