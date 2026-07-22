"""contradict × dedup gate: the contradicts edge must never dangle.

When the corrective text of a contradiction is near-identical to an already
stored record, the write-time dedup gate collapses the corrective insert.
The contradicts edge must then target the surviving record — an edge to the
gated-away id would point at a row that never existed.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from iai_mcp import retrieve
from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore


@pytest.fixture(autouse=True)
def _live_dedup_gate(monkeypatch):
    # The pattern-separation gate defaults to dry-run under pytest; the
    # gated-corrective scenario needs the real SKIP behavior.
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _seed(store: MemoryStore, text: str, session: str) -> UUID:
    result = capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=session, role="user",
    )
    assert result["status"] == "inserted", result
    return UUID(result["record_id"])


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_contradict_with_gated_corrective_targets_survivor(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    original_id = _seed(store, "The daemon must gate every recall request.", "s1")
    survivor_text = "The hippocampus answers recall without any daemon."
    survivor_id = _seed(store, survivor_text, "s2")
    survivor = store.get(survivor_id)
    assert survivor is not None

    receipt = retrieve.contradict(
        store, original_id, survivor_text, list(survivor.embedding),
    )

    assert receipt.new_record_id == survivor_id, (
        f"[{driver}] the edge must target the surviving duplicate, "
        f"got {receipt.new_record_id}"
    )
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT src, dst FROM edges WHERE edge_type = 'contradicts'"
        ).fetchall()
    assert rows, "contradicts edge missing"
    for row in rows:
        for endpoint in (str(row[0]), str(row[1])):
            assert store.get(UUID(endpoint)) is not None, (
                f"[{driver}] dangling contradicts edge endpoint {endpoint}"
            )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_contradict_with_fresh_corrective_still_inserts(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    original_id = _seed(store, "Recall must go through the daemon socket.", "s1")

    new_fact = "Recall reads the awake store directly; the daemon only sleeps."
    from iai_mcp.embed import embedder_for_store
    emb = embedder_for_store(store).embed(new_fact)

    receipt = retrieve.contradict(store, original_id, new_fact, list(emb))

    rec = store.get(receipt.new_record_id)
    assert rec is not None
    assert rec.literal_surface == new_fact


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_corrective_folding_into_original_is_rejected(driver, tmp_path, monkeypatch):
    """A corrector near-identical to the CONTRADICTED record itself would wire
    a self-contradiction loop after the dedup fold — it must be rejected with
    a clear error instead."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    text = "The projection matrix rotates on every consolidation cycle."
    original_id = _seed(store, text, "s1")
    original = store.get(original_id)
    assert original is not None

    with pytest.raises(ValueError, match="contradicted record itself"):
        retrieve.contradict(store, original_id, text, list(original.embedding))

    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT src, dst FROM edges WHERE edge_type = 'contradicts'"
        ).fetchall()
    assert all(str(r[0]) != str(r[1]) for r in rows), "self-contradiction edge"
