"""Retiring a directive clears its flag but leaves it searchable (auditable,
no longer injected)."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp import retrieve
from iai_mcp.capture import capture_turn
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore, flush_record_buffer


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _live_directive_ids(store: MemoryStore) -> set[UUID]:
    flush_record_buffer(store)
    return {
        rec.id
        for rec in store.iter_records(where="directive = 1 AND tombstoned_at IS NULL")
    }


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_contradict_clears_directive_flag_and_stays_findable(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    seed = capture_turn(
        store=store, cue="c", text="from now on reply in English",
        directive=True, session_id="s1", role="user",
    )
    assert seed["status"] == "inserted", seed
    original_id = UUID(seed["record_id"])
    original_before = store.get(original_id)
    assert original_before is not None
    assert original_before.directive is True
    assert original_id in _live_directive_ids(store)

    new_fact = "from now on reply in Spanish"
    emb = embedder_for_store(store).embed(new_fact)
    receipt = retrieve.contradict(store, original_id, new_fact, list(emb))

    original_after = store.get(original_id)
    assert original_after is not None
    assert original_after.directive is False
    assert original_after.literal_surface == original_before.literal_surface
    assert original_id not in _live_directive_ids(store)

    corrector = store.get(receipt.new_record_id)
    assert corrector is not None
    assert corrector.directive is False

    cue_emb = embedder_for_store(store).embed(original_before.literal_surface)
    response = retrieve.recall(
        store,
        cue_embedding=list(cue_emb),
        cue_text=original_before.literal_surface,
        session_id="s1",
    )
    hit_ids = {h.record_id for h in response.hits}
    assert original_id in hit_ids, (hit_ids, [h.reason for h in response.hits])


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_contradict_leaves_non_directive_original_untouched(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    seed = capture_turn(
        store=store, cue="c", text="alice's release ships next Tuesday",
        session_id="s1", role="user",
    )
    assert seed["status"] == "inserted", seed
    original_id = UUID(seed["record_id"])
    original_before = store.get(original_id)
    assert original_before is not None
    assert original_before.directive is False

    new_fact = "alice's release actually shipped last Tuesday"
    emb = embedder_for_store(store).embed(new_fact)
    retrieve.contradict(store, original_id, new_fact, list(emb))

    original_after = store.get(original_id)
    assert original_after is not None
    assert original_after.directive is False
    assert original_after.updated_at == original_before.updated_at
