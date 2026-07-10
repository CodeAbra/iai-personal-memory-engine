"""A chunk two documents share is owned by BOTH: reinforcing unions the new
document's tag onto the record, and superseding one document only disowns
its tag — the fade fires when the LAST owner drops the chunk."""
from __future__ import annotations

import pytest


SHARED = (
    "Licensed under the Apache License Version 2 shared boilerplate header."
)
ONLY_A = "Alpha document unique content about orchard irrigation schedules."
ONLY_B = "Beta document unique content about violin bow maintenance rituals."


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    from iai_mcp.store import MemoryStore

    s = MemoryStore(path=tmp_path)
    yield s
    s.close()


def _fetch_by_surface(store, needle: str):
    return next(
        (r for r in store.iter_records(where="tombstoned_at IS NULL")
         if needle in (r.literal_surface or "")),
        None,
    )


def test_shared_chunk_survives_first_owner_supersede(store):
    from iai_mcp.study import study_document

    # Byte-identical single-chunk documents: the second study reinforces via
    # the exact idem key — the union point under test.
    study_document(store, text=SHARED, source_name="a.md")
    study_document(store, text=SHARED, source_name="b.md")

    shared_rec = _fetch_by_surface(store, "Apache License")
    assert shared_rec is not None
    doc_tags = {t for t in (shared_rec.tags or []) if t.startswith("doc:")}
    assert doc_tags == {"doc:a.md", "doc:b.md"}, (
        f"reinforce must union the second owner's tag; got {doc_tags}"
    )

    # Restudy a.md WITHOUT the shared chunk: a.md disowns it, b.md still
    # owns it — the record must NOT drop into the decay funnel.
    study_document(store, text=ONLY_A, source_name="a.md")
    shared_after = store.get(shared_rec.id)
    assert shared_after is not None
    tags_after = {t for t in (shared_after.tags or []) if t.startswith("doc:")}
    assert tags_after == {"doc:b.md"}, tags_after
    assert shared_after.never_decay == shared_rec.never_decay, (
        "shared chunk faded while still owned by b.md"
    )

    # Now b.md drops it too — LAST owner: the fade fires.
    study_document(store, text=ONLY_B, source_name="b.md")
    shared_final = store.get(shared_rec.id)
    assert shared_final is not None, "supersede never deletes"
    assert float(shared_final.centrality or 0.0) == 0.0
