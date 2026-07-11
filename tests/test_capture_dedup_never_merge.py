"""RED test for the capture-time half of the never_merge hard lock: it must
hold at the capture-time cosine gate, not just the write-time gate.

Confirmed live gap (capture.py, the cosine loop below the exact-key branch):
the capture-time gate's cosine loop calls ``store.reinforce_record(record.id)``
unconditionally on any ``score >= DEDUP_COS_THRESHOLD`` hit, with NO
``never_merge`` check on the existing neighbour — a pinned record can be
silently reinforced.

Ground truth for the neighbour's cosine score is controlled by monkeypatching
``store.query_similar`` directly (never depending on the real embedder's
actual similarity output — mirrors the eval fixture's synthetic-cosine
discipline), so this test is deterministic regardless of embedder behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.capture import DEDUP_COS_THRESHOLD, capture_turn
from iai_mcp.store import MemoryStore
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


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _make_record(
    rid: UUID,
    surface: str = "topic",
    *,
    tier: str = "episodic",
    embedding: list[float] | None = None,
    never_merge: bool = False,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier=tier,
        literal_surface=surface,
        aaak_index="",
        embedding=list(embedding) if embedding is not None else [0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=never_merge,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pinned_neighbour_never_reinforced_by_capture_turn(driver, store, monkeypatch):
    """A never_merge=True existing neighbour at cos >= DEDUP_COS_THRESHOLD is
    NEVER reinforced by capture_turn — the turn falls through to a normal
    insert instead.

    RED today: capture.py's cosine loop (~lines 390-402) calls
    store.reinforce_record(record.id) unconditionally on any qualifying
    cosine hit, with no never_merge guard.
    """
    _select_driver(driver, monkeypatch)

    pinned_id = uuid4()
    pinned_record = _make_record(
        pinned_id,
        "alice identity anchor — never merge",
        tier="semantic",
        never_merge=True,
    )
    store.insert(pinned_record)

    reinforce_calls: list[UUID] = []
    orig_reinforce = store.reinforce_record

    def _spy_reinforce(record_id, *args, **kwargs):
        reinforce_calls.append(record_id)
        return orig_reinforce(record_id, *args, **kwargs)

    monkeypatch.setattr(store, "reinforce_record", _spy_reinforce)

    # Deterministic neighbour score at/above DEDUP_COS_THRESHOLD, independent
    # of the real embedder's actual cosine output.
    def _fake_query_similar(embedding, k=3, tier=None):
        return [(pinned_record, DEDUP_COS_THRESHOLD + 0.01)]

    monkeypatch.setattr(store, "query_similar", _fake_query_similar)

    result = capture_turn(
        store=store,
        text="alice's identity anchor restated in slightly different words",
        cue="identity probe",
        tier="semantic",
        session_id="s1",
        role="user",
    )

    assert pinned_id not in reinforce_calls, (
        f"[driver={driver}] capture_turn must NEVER reinforce a never_merge=True "
        f"neighbour; reinforce_record was called with {reinforce_calls}"
    )
    assert result["status"] != "reinforced", (
        f"[driver={driver}] capture_turn must not report 'reinforced' against "
        f"a pinned neighbour; got {result}"
    )

    # The pinned record's hebbian self-loop edge weight must be unchanged —
    # reinforce_record was never invoked, so no edge-weight mutation occurred.
    # The turn must fall through to a normal insert (which the write-time
    # gate then independently protects).
    rows = list(store.iter_records())
    surfaces = [r.literal_surface for r in rows]
    assert "alice identity anchor — never merge" in surfaces, (
        "the pinned record's literal_surface must be untouched"
    )
    non_pinned_rows = [r for r in rows if r.id != pinned_id]
    assert len(non_pinned_rows) >= 1, (
        f"[driver={driver}] the incoming turn must fall through to a normal "
        f"insert rather than being silently absorbed into the pinned record; "
        f"rows={[r.literal_surface for r in rows]}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_non_pinned_neighbour_still_reinforced_regression(driver, store, monkeypatch):
    """Regression guard: a NON-pinned neighbour at cos >= DEDUP_COS_THRESHOLD
    is still reinforced and 'reinforced' is returned — unchanged behaviour.
    This must stay GREEN before and after the never_merge guard lands.
    """
    _select_driver(driver, monkeypatch)

    existing_id = uuid4()
    existing_record = _make_record(
        existing_id, "the daemon restarts nightly for consolidation", tier="semantic",
        never_merge=False,
    )
    store.insert(existing_record)

    reinforce_calls: list[UUID] = []
    orig_reinforce = store.reinforce_record

    def _spy_reinforce(record_id, *args, **kwargs):
        reinforce_calls.append(record_id)
        return orig_reinforce(record_id, *args, **kwargs)

    monkeypatch.setattr(store, "reinforce_record", _spy_reinforce)

    def _fake_query_similar(embedding, k=3, tier=None):
        return [(existing_record, DEDUP_COS_THRESHOLD + 0.01)]

    monkeypatch.setattr(store, "query_similar", _fake_query_similar)

    result = capture_turn(
        store=store,
        text="the daemon restarts nightly for consolidation, restated",
        cue="daemon probe",
        tier="semantic",
        session_id="s1",
        role="user",
    )

    assert existing_id in reinforce_calls, (
        f"[driver={driver}] a non-pinned neighbour must still be reinforced "
        f"(unchanged behaviour); reinforce_record was called with {reinforce_calls}"
    )
    assert result["status"] == "reinforced", (
        f"[driver={driver}] expected 'reinforced' for a non-pinned near-dup "
        f"neighbour; got {result}"
    )
