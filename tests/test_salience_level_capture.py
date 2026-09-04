"""Write-path proof for salience_level through capture_turn: default-safety
(omission stays "unflagged", byte-unchanged for existing callers), invalid
value coercion, and monotone-raise-never-lowers across both dedup-fold
branches (exact idem-key and near-dup cosine).

The mark is rank-boost-only: never_merge and pinned must stay untouched by
every path exercised here.
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


def _make_record(rid: UUID, surface: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
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
        tags=[],
        language="en",
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_stores_explicit_salience_level(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    result = capture_turn(
        store=store, cue="c", text="alice's decision to ship on Friday is load-bearing",
        salience_level="critical", session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.salience_level == "critical"
    assert rec.never_merge is False
    assert rec.pinned is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_omitted_defaults_to_unflagged(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    result = capture_turn(
        store=store, cue="c", text="alice attended the weekly standup meeting",
        session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.salience_level == "unflagged"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_invalid_salience_level_coerced_to_unflagged(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    result = capture_turn(
        store=store, cue="c", text="alice's invalid-salience capture attempt here",
        salience_level="banana", session_id="s1", role="user",
    )
    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.salience_level == "unflagged"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_exact_idem_key_fold_raises_salience_level(driver, store, monkeypatch):
    """Two captures with the SAME session_id/role/ts/text hit the exact
    idem-key fold branch. A higher second-call salience_level must raise the
    existing record's stored level, never fork a second record."""
    _select_driver(driver, monkeypatch)

    fixed_ts = "2026-01-01T00:00:00+00:00"
    text = "alice confirmed the migration plan is final"

    result_a = capture_turn(
        store=store, cue="c", text=text, ts=fixed_ts,
        salience_level="unflagged", session_id="s1", role="user",
    )
    assert result_a["status"] == "inserted", result_a
    id_a = result_a["record_id"]
    count_before = sum(1 for _ in store.iter_records())

    result_b = capture_turn(
        store=store, cue="c", text=text, ts=fixed_ts,
        salience_level="critical", session_id="s1", role="user",
    )
    assert result_b["status"] == "reinforced", result_b
    assert result_b["record_id"] == id_a, result_b

    count_after = sum(1 for _ in store.iter_records())
    assert count_after == count_before, "the fold must never fork a second record"

    survivor = store.get(UUID(id_a))
    assert survivor is not None
    assert survivor.salience_level == "critical", (
        f"exact idem-key fold must raise salience_level, got {survivor.salience_level!r}"
    )
    assert survivor.never_merge is False
    assert survivor.pinned is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_exact_idem_key_fold_never_lowers_salience_level(driver, store, monkeypatch):
    """A THIRD capture on the exact idem-key fold path with a LOWER
    salience_level than what's currently stored must leave the stored value
    unchanged (monotone raise, never lower)."""
    _select_driver(driver, monkeypatch)

    fixed_ts = "2026-01-01T00:00:00+00:00"
    text = "alice confirmed the rollback plan is final"

    result_a = capture_turn(
        store=store, cue="c", text=text, ts=fixed_ts,
        salience_level="critical", session_id="s1", role="user",
    )
    assert result_a["status"] == "inserted", result_a
    id_a = result_a["record_id"]

    result_b = capture_turn(
        store=store, cue="c", text=text, ts=fixed_ts,
        salience_level="unflagged", session_id="s1", role="user",
    )
    assert result_b["status"] == "reinforced", result_b
    assert result_b["record_id"] == id_a, result_b

    survivor = store.get(UUID(id_a))
    assert survivor is not None
    assert survivor.salience_level == "critical", (
        f"a lower incoming salience_level must never lower the stored value, "
        f"got {survivor.salience_level!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_near_dup_fold_raises_salience_level(driver, store, monkeypatch):
    """Two near-duplicate (not identical) captures via the cosine fold
    branch (near_dup_gate=True). A higher second-call salience_level must
    raise the existing record's stored level."""
    _select_driver(driver, monkeypatch)

    result_a = capture_turn(
        store=store, cue="c", text="alice's fix lands next week for sure",
        salience_level="unflagged", near_dup_gate=True, session_id="s1", role="user",
    )
    assert result_a["status"] == "inserted", result_a
    id_a = result_a["record_id"]
    record_a = store.get(UUID(id_a))
    assert record_a is not None
    count_before = sum(1 for _ in store.iter_records())

    def _fake_query_similar(embedding, k=3, tier=None):
        return [(record_a, DEDUP_COS_THRESHOLD + 0.01)]

    monkeypatch.setattr(store, "query_similar", _fake_query_similar)

    result_b = capture_turn(
        store=store, cue="c", text="alice's fix will land next week for sure",
        salience_level="notable", near_dup_gate=True, session_id="s1", role="user",
    )
    assert result_b["status"] == "reinforced", result_b
    assert result_b["record_id"] == id_a, result_b
    assert result_b["reason"].startswith("cos="), result_b

    count_after = sum(1 for _ in store.iter_records())
    assert count_after == count_before, "the fold must never fork a second record"

    survivor = store.get(UUID(id_a))
    assert survivor is not None
    assert survivor.salience_level == "notable", (
        f"near-dup fold must raise salience_level, got {survivor.salience_level!r}"
    )
    assert survivor.never_merge is False
    assert survivor.pinned is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_near_dup_fold_never_lowers_salience_level(driver, store, monkeypatch):
    """A THIRD near-dup fold with a LOWER salience_level than what's
    currently stored must leave the stored value unchanged."""
    _select_driver(driver, monkeypatch)

    result_a = capture_turn(
        store=store, cue="c", text="alice's audit closes out this quarter cleanly",
        salience_level="critical", near_dup_gate=True, session_id="s1", role="user",
    )
    assert result_a["status"] == "inserted", result_a
    id_a = result_a["record_id"]
    record_a = store.get(UUID(id_a))
    assert record_a is not None

    def _fake_query_similar(embedding, k=3, tier=None):
        return [(record_a, DEDUP_COS_THRESHOLD + 0.01)]

    monkeypatch.setattr(store, "query_similar", _fake_query_similar)

    result_b = capture_turn(
        store=store, cue="c", text="alice's audit will close out this quarter cleanly",
        salience_level="unflagged", near_dup_gate=True, session_id="s1", role="user",
    )
    assert result_b["status"] == "reinforced", result_b
    assert result_b["record_id"] == id_a, result_b

    survivor = store.get(UUID(id_a))
    assert survivor is not None
    assert survivor.salience_level == "critical", (
        f"a lower incoming salience_level must never lower the stored value, "
        f"got {survivor.salience_level!r}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pinned_neighbour_salience_untouched_by_capture(driver, store, monkeypatch):
    """A never_merge=True neighbour is never reinforced (existing invariant,
    test_capture_dedup_never_merge.py's regression guard) -- confirm the
    salience field on it is likewise never mutated by an incoming capture
    that skips it entirely."""
    _select_driver(driver, monkeypatch)

    pinned_id = uuid4()
    now = datetime.now(timezone.utc)
    pinned_record = MemoryRecord(
        id=pinned_id, tier="semantic",
        literal_surface="alice identity anchor — never merge",
        aaak_index="", embedding=[0.1] * EMBED_DIM, community_id=None,
        centrality=0.0, detail_level=2, pinned=False, stability=0.0,
        difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=True,
        provenance=[], created_at=now, updated_at=now, tags=[], language="en",
        salience_level="notable",
    )
    store.insert(pinned_record)

    def _fake_query_similar(embedding, k=3, tier=None):
        return [(pinned_record, DEDUP_COS_THRESHOLD + 0.01)]

    monkeypatch.setattr(store, "query_similar", _fake_query_similar)

    result = capture_turn(
        store=store,
        text="alice's identity anchor restated in slightly different words",
        cue="identity probe", tier="semantic", salience_level="critical",
        session_id="s1", role="user",
    )

    assert result["status"] != "reinforced", result
    survivor = store.get(pinned_id)
    assert survivor is not None
    assert survivor.salience_level == "notable", (
        f"a skipped (never_merge) neighbour's salience_level must stay untouched, "
        f"got {survivor.salience_level!r}"
    )
    assert survivor.never_merge is True
