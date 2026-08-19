"""Doctor check for non-finite (NaN/inf) coercions at the exact-cosine
authority index. Mirrors tests/test_doctor_centrality_row.py's store-setup
and events-spy patterns.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


def _make_record(vec: list[float], text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
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
        tags=["t"],
        language="en",
    )


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# Monkeypatched query_events cases
# ---------------------------------------------------------------------------


def test_dd_warn_when_coerced_events_present(monkeypatch, tmp_path) -> None:
    # The stubbed-open scenario presumes an existing store file; an absent
    # one short-circuits to the no-store skip row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    from iai_mcp import doctor as _doctor
    import iai_mcp.events as _events

    synthetic = [
        {
            "data": {"action": "coerced", "source": "cue", "count": 1, "total": 3},
            "ts": "2026-08-17T00:00:00",
        },
        {
            "data": {"action": "coerced", "source": "cue", "count": 1, "total": 5},
            "ts": "2026-08-17T00:01:00",
        },
        {
            "data": {
                "action": "coerced",
                "source": "row",
                "context": "build",
                "count": 2,
                "total": 2,
            },
            "ts": "2026-08-17T00:02:00",
        },
    ]
    monkeypatch.setattr(_events, "query_events", lambda *a, **kw: synthetic)
    import iai_mcp.store as _store_mod
    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    result = _doctor.check_dd_exact_index_coercions()
    assert result.status == "WARN"
    assert "cue=5" in result.detail
    assert "row=2" in result.detail
    assert "events query failed" not in result.detail


def test_dd_pass_when_no_coerced_events(monkeypatch, tmp_path) -> None:
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    from iai_mcp import doctor as _doctor
    import iai_mcp.events as _events

    # Only the write-path (severity=error) rejection event — no action=coerced.
    synthetic = [{"data": {"record_id": "x", "nan_count": 1}}]
    monkeypatch.setattr(_events, "query_events", lambda *a, **kw: synthetic)
    import iai_mcp.store as _store_mod
    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    result = _doctor.check_dd_exact_index_coercions()
    assert result.status == "PASS"
    assert "events query failed" not in result.detail


def test_dd_never_calls_exact_top_k_or_build(monkeypatch, tmp_path) -> None:
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    from iai_mcp import doctor as _doctor
    import iai_mcp.events as _events

    monkeypatch.setattr(_events, "query_events", lambda *a, **kw: [])

    class _NeverBuildStore:
        def exact_top_k(self, *a, **kw):
            raise AssertionError("check_dd must never call exact_top_k")

        def _build_exact_index_sync(self, *a, **kw):
            raise AssertionError(
                "check_dd must never call _build_exact_index_sync"
            )

        def close(self):
            pass

    import iai_mcp.store as _store_mod
    monkeypatch.setattr(
        _store_mod, "MemoryStore", lambda *a, **kw: _NeverBuildStore()
    )

    result = _doctor.check_dd_exact_index_coercions()
    assert result.status == "PASS"


def test_dd_no_store_skips(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    from iai_mcp import doctor as _doctor

    result = _doctor.check_dd_exact_index_coercions()
    assert result.status == "PASS"
    assert "no store yet" in result.detail.lower()


# ---------------------------------------------------------------------------
# Real-signature case — no monkeypatch of query_events
# ---------------------------------------------------------------------------


def test_dd_real_signature_shared_store_warns(monkeypatch, tmp_path) -> None:
    """The ONLY case exercising the real query_events signature: emits one
    coerced event through the real feed path, flushes it, then runs the
    check against the SAME store root and key — no query_events monkeypatch.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    from iai_mcp import doctor as _doctor
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.events import flush_event_buffer

    store = MemoryStore()
    rec = _make_record(_seeded_vec(1), "warm-record")
    store.insert(rec)
    flush_record_buffer(store)
    store.exact_top_k(rec.embedding, k=5)  # warms the resident matrix
    assert store._exact_index.is_warm is True

    nan_vec = [float("nan")] * EMBED_DIM
    store._feed_exact_index(str(uuid4()), nan_vec)
    flush_event_buffer(store)
    store.close()

    result = _doctor.check_dd_exact_index_coercions()
    assert result.status == "WARN"
    assert "events query failed" not in result.detail
