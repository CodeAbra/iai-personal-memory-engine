"""The raw-SQL direct-write path (`write_turn_direct`, used while the daemon
is down) must stamp `directive` and `directive_source` in the SAME INSERT as
the record itself -- never a follow-up UPDATE after the insert has already
committed. A two-step insert-then-update can half-fail: the record lands
with `directive_source=explicit-command` in its provenance but `directive`
stuck at its insert-time default, an orphan state with no reconciliation.

Covers both write branches (`_insert_row_with_embedding` eager-embed and
`insert_pending_row` deferred-embed) on both storage drivers.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp.hippo._table import HippoTable


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _forbid_table_update(monkeypatch) -> None:
    """Any records-table UPDATE during the write proves the write is NOT
    atomic -- a genuinely single-INSERT write never needs a follow-up
    UPDATE call at all."""

    def _raise(self, *args, **kwargs):
        raise AssertionError(
            "direct-write directive stamping must not issue a follow-up "
            "table UPDATE -- directive/directive_source must land in the "
            "same INSERT as the record"
        )

    monkeypatch.setattr(HippoTable, "update", _raise)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_deferred_embed_directive_lands_atomically(
    hermetic_store: Path, monkeypatch, driver
) -> None:
    """insert_pending_row branch: directive=True lands with the insert, and
    no follow-up UPDATE is issued."""
    _select_driver(driver, monkeypatch)
    _forbid_table_update(monkeypatch)

    from iai_mcp.direct_write import write_turn_direct

    result = write_turn_direct(
        store_root=hermetic_store,
        text="atomic deferred directive probe text",
        session_id="s-deferred",
        role="user",
        directive=True,
        deferred_embedding=True,
    )
    assert result["status"] == "inserted", result

    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        rec = store.get(UUID(result["record_id"]))
        assert rec is not None
        assert rec.directive is True, "directive column must be set at insert time"
        stamps = [p.get("directive_source") for p in rec.provenance if isinstance(p, dict)]
        assert "explicit-command" in stamps, rec.provenance
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_eager_embed_directive_lands_atomically(
    hermetic_store: Path, monkeypatch, driver
) -> None:
    """_insert_row_with_embedding branch: directive=True lands with the
    insert, and no follow-up UPDATE is issued."""
    _select_driver(driver, monkeypatch)

    import iai_mcp.direct_write as dw
    from iai_mcp.types import EMBED_DIM

    monkeypatch.setattr(
        dw, "_try_get_embedding_fast", lambda text, cue: [0.01] * EMBED_DIM
    )
    _forbid_table_update(monkeypatch)

    result = dw.write_turn_direct(
        store_root=hermetic_store,
        text="atomic eager directive probe text",
        session_id="s-eager",
        role="user",
        directive=True,
        deferred_embedding=False,
    )
    assert result["status"] == "inserted", result

    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        rec = store.get(UUID(result["record_id"]))
        assert rec is not None
        assert rec.directive is True, "directive column must be set at insert time"
        stamps = [p.get("directive_source") for p in rec.provenance if isinstance(p, dict)]
        assert "explicit-command" in stamps, rec.provenance
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("deferred_embedding", [True, False], ids=["deferred", "eager"])
def test_directive_false_never_stamps_or_sets_flag(
    hermetic_store: Path, monkeypatch, driver, deferred_embedding
) -> None:
    """A plain non-directive write never sets the flag or the provenance
    stamp -- a control case for the atomic-write assertions above."""
    _select_driver(driver, monkeypatch)

    import iai_mcp.direct_write as dw
    from iai_mcp.types import EMBED_DIM

    if not deferred_embedding:
        monkeypatch.setattr(
            dw, "_try_get_embedding_fast", lambda text, cue: [0.01] * EMBED_DIM
        )

    result = dw.write_turn_direct(
        store_root=hermetic_store,
        text="non-directive control probe text",
        session_id="s-control",
        role="user",
        deferred_embedding=deferred_embedding,
    )
    assert result["status"] == "inserted", result

    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        rec = store.get(UUID(result["record_id"]))
        assert rec is not None
        assert rec.directive is False
        stamps = [p.get("directive_source") for p in rec.provenance if isinstance(p, dict)]
        assert "explicit-command" not in stamps, rec.provenance
    finally:
        store.close()
