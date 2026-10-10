"""An AWAKE-path write must remain visible after a full store close and a
fresh reopen from the same on-disk path, with no daemon and no sleep cycle
in between -- both by id and through recall, with verbatim content intact.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make  # noqa: E402

from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

from _recall_helpers import _random_vec  # noqa: E402

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _target_record(text: str, vec: list[float]) -> MemoryRecord:
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
        tags=[],
        language="en",
    )


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_awake_write_visible_by_id_after_restart(
    tmp_path, monkeypatch, driver,
) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)

    from iai_mcp.embed import Embedder

    embedder = Embedder()
    verbatim = "alice's distinctive note for the restart durability check"
    target = _target_record(verbatim, embedder.embed(verbatim))

    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store.insert(target)
    store.close()

    store_b = MemoryStore(path=tmp_path)
    got = store_b.get(target.id)
    assert got is not None, (
        "a write made durable only by store.close() (no per-insert "
        "autoflush) must be present after a fresh reopen from the same path"
    )
    assert got.literal_surface == verbatim, (
        "literal_surface must survive a close-then-reopen byte-identical"
    )


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_awake_write_recallable_after_restart(
    tmp_path, monkeypatch, driver,
) -> None:
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)

    from iai_mcp.embed import Embedder

    embedder = Embedder()

    n_decoys = 20
    decoy_body = (
        "User decoy filler content unrelated to the target note, padded "
        "long enough that not every record fits inside the recall budget "
    )
    for i in range(n_decoys):
        store.insert(_make(text=f"{decoy_body}index {i}", vec=_random_vec(9000 + i)))

    target_text = "alice's distinctive note surfaced by recall after a restart"
    cue_vec = embedder.embed(target_text)
    target = _target_record(target_text, cue_vec)

    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store.insert(target)
    store.close()

    store_b = MemoryStore(path=tmp_path)

    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    graph, assignment, rich_club = _retrieve.build_runtime_graph(store_b)
    _rgc.save(store_b, assignment, rich_club)

    from iai_mcp.pipeline import recall_for_response
    import iai_mcp.pipeline as _pipeline_mod

    _pipeline_mod._last_recall_latency_ms = 0.0

    response = recall_for_response(
        store=store_b,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=embedder,
        cue=target_text,
        session_id="restart-check",
        budget_tokens=150,
        mode="concept",
    )

    hit_ids = {h.record_id for h in response.hits}
    assert target.id in hit_ids, (
        "a write made durable only by store.close() must be surfaced by "
        "recall rebuilt from disk after a fresh reopen, with no daemon and "
        "no sleep cycle in between"
    )
    matching = next(h for h in response.hits if h.record_id == target.id)
    assert matching.literal_surface == target_text, (
        "recall must surface the write's literal_surface byte-identical, "
        "never paraphrased or trimmed"
    )
    assert len(response.hits) < n_decoys + 1, (
        "recall returned the entire corpus -- the target-in-hits assertion "
        "above would pass vacuously"
    )
