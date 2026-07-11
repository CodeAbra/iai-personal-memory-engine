"""Direct teaching end-to-end: a studied file measurably CHANGES the brain.

Proves the weave, not just the append: Hebbian links form from studied
chunks to related PRE-EXISTING records, contradicted prior beliefs gain a
contradicts edge to the correcting chunk, document schemas are induced, the
dedup gate keeps a re-study from flooding memory, and pre-existing content
stays byte-identical (verbatim invariant).

Real embedder, temp stores only, both storage drivers.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore
from iai_mcp.study import study_document


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture(autouse=True)
def _teach_thresholds(monkeypatch):
    # Real-embedding cosines between crafted related sentences land ~0.5-0.9;
    # pin the link floor so the weave assertions are deterministic.
    monkeypatch.setenv("IAI_MCP_PATSEP_LINK_THRESHOLD", "0.45")
    monkeypatch.setenv("IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD", "0.92")


def _seed(store: MemoryStore, text: str, session: str = "prior") -> UUID:
    result = capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=session, role="user",
    )
    assert result["status"] == "inserted", result
    return UUID(result["record_id"])


def _edges(store: MemoryStore) -> list[tuple[str, str, str]]:
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT src, dst, edge_type FROM edges"
        ).fetchall()
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def _record_count(store: MemoryStore) -> int:
    with store.db._conn_lock:
        row = store.db._conn.execute("SELECT COUNT(*) FROM records").fetchone()
    return int(row[0])


_DOC = (
    "Sleep consolidation strengthens hippocampal memory traces overnight. "
    "The hippocampus replays recent experiences while the cortex extracts patterns. "
    "Deep sleep stages are when the brain prunes weak synaptic connections. "
    "Memory retrieval works best after a full consolidation cycle completes."
)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_teach_weaves_links_to_existing_records(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    prior_text = "The hippocampus consolidates memories during sleep."
    prior_id = _seed(store, prior_text)

    report = study_document(store, text=_DOC, source_name="sleep-notes.md")

    assert report["inserted"] >= 1, report
    assert report["edges_formed"] >= 1, (
        f"[{driver}] teach must weave edges to related existing records: {report}"
    )
    hebbian = [e for e in _edges(store) if e[2] == "hebbian"]
    touching_prior = [
        e for e in hebbian if str(prior_id) in (e[0], e[1])
    ]
    assert touching_prior, (
        f"[{driver}] no hebbian edge touches the pre-existing related record; "
        f"edges={hebbian[:10]}"
    )
    # Verbatim invariant: the pre-existing record is untouched byte-for-byte.
    got = store.get(prior_id)
    assert got is not None and got.literal_surface == prior_text


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_reteach_is_idempotent_no_flooding(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    first = study_document(store, text=_DOC, source_name="sleep-notes.md")
    count_after_first = _record_count(store)
    second = study_document(store, text=_DOC, source_name="sleep-notes.md")
    count_after_second = _record_count(store)

    assert first["inserted"] >= 1
    assert second["inserted"] == 0, (
        f"[{driver}] re-study must not insert duplicates: {second}"
    )
    assert count_after_second == count_after_first, (
        f"[{driver}] record count grew on re-study: "
        f"{count_after_first} -> {count_after_second}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_reteach_different_session_does_not_flood(driver, tmp_path, monkeypatch):
    """The cosine near-dup gate must catch what the idem key cannot: the same
    content taught under a different session/source has a different idem key
    (session_id is part of it), so only the cos>=threshold gate stands between
    a re-teach and a full duplicate set."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    first = study_document(
        store, text=_DOC, source_name="sleep-notes.md", session_id="study",
    )
    assert first["inserted"] >= 1
    count_after_first = _record_count(store)

    second = study_document(
        store, text=_DOC, source_name="sleep-notes-copy.md", session_id="other",
    )
    assert second["inserted"] == 0, (
        f"[{driver}] identical content under a new session flooded the store: {second}"
    )
    assert second["reinforced"] >= 1, second
    assert _record_count(store) == count_after_first


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_teach_reconciles_contradicted_belief(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    stale_text = "The project uses LanceDB as its storage backend."
    stale_id = _seed(store, stale_text)

    doc = (
        "The storage backend migrated away from LanceDB entirely. "
        "Persistence now runs on the Hippo store over SQLite with an ANN index."
    )

    def _critic(items):
        return {rid: 0.9 for rid, _surface in items}

    report = study_document(
        store, text=doc, source_name="storage-update.md", critic=_critic,
    )

    assert report["contradiction_candidates"] >= 1, report
    assert report["contradictions_resolved"] >= 1, report
    contradicts = [
        e for e in _edges(store)
        if e[2] == "contradicts" and e[0] == str(stale_id)
    ]
    assert contradicts, (
        f"[{driver}] stale belief must carry a contradicts edge to the "
        f"correcting chunk; edges={_edges(store)[:10]}"
    )
    # The corrector is a real stored chunk, and the stale original is intact.
    corrector = store.get(UUID(contradicts[0][1]))
    assert corrector is not None
    got = store.get(stale_id)
    assert got is not None and got.literal_surface == stale_text


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_teach_without_critic_only_flags_candidates(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed(store, "The project uses LanceDB as its storage backend.")

    doc = "The storage backend migrated away from LanceDB to the Hippo store."
    report = study_document(store, text=doc, source_name="s.md", critic=None)

    assert report["contradictions_resolved"] == 0
    assert not [e for e in _edges(store) if e[2] == "contradicts"]


_TOPICS = (
    "Photosynthesis converts sunlight into chemical energy inside chloroplasts.",
    "Volcanic eruptions reshape coastlines and create new mineral deposits.",
    "The French Revolution transformed European political institutions forever.",
    "Quantum entanglement links particle states across arbitrary distances.",
    "Coral reefs host a quarter of all known marine species on the planet.",
    "The printing press accelerated literacy across fifteenth-century Europe.",
    "Plate tectonics explains earthquake belts along continental boundaries.",
    "Antibiotic resistance emerges through horizontal gene transfer in bacteria.",
    "Glacial cycles carved the fjord landscapes of the northern hemisphere.",
    "Neural networks approximate functions by composing weighted nonlinearities.",
)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_teach_induces_schema_over_combined_material(driver, tmp_path, monkeypatch):
    """Schemas form from OLD + NEW evidence together: 8 pre-existing records
    plus the studied chunks cross the tier-0 auto-induction bar (>=9
    co-occurrences, confidence >=0.85), and the persisted schema's evidence
    includes the freshly studied material."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    for topic in _TOPICS[:8]:
        _seed(store, topic)

    report = study_document(store, text=_DOC, source_name="sleep-notes.md")

    assert report["inserted"] >= 1, report
    assert report["schemas_induced"] >= 1, (
        f"[{driver}] studying must induce a schema over combined material: {report}"
    )
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT literal_surface FROM records WHERE tier = 'semantic'"
        ).fetchall()
    assert rows, f"[{driver}] no semantic schema record persisted"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_teach_e2e_brain_delta(driver, tmp_path, monkeypatch):
    """The whole point in one assertion set: pre-teach snapshot vs post."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    prior_id = _seed(store, "The hippocampus consolidates memories during sleep.")
    records_before = _record_count(store)
    edges_before = len(_edges(store))

    def _critic(items):
        return {rid: 0.9 for rid, _surface in items}

    report = study_document(
        store, text=_DOC, source_name="sleep-notes.md", critic=_critic,
    )

    assert _record_count(store) > records_before, "new knowledge stored"
    assert len(_edges(store)) > edges_before, "links formed"
    hebbian_to_prior = [
        e for e in _edges(store)
        if e[2] == "hebbian" and str(prior_id) in (e[0], e[1])
    ]
    assert hebbian_to_prior, "existing memory demonstrably enriched"
    assert report["chunks_total"] == (
        report["inserted"] + report["reinforced"] + report["skipped"]
    )
    # Closed loop: studied chunks are demonstrably RECALLABLE, not just stored.
    assert report["recall_tested"] >= 1, report
    assert report["recall_verified"] == report["recall_tested"], (
        f"studied material failed the recall self-test: {report}"
    )
    # Targeted reactivation: the studied chunks AND the woven prior record are
    # stamped for the next sleep replay cycle.
    assert report["replay_queued"] >= 1, report
    prior_after = store.get(prior_id)
    assert prior_after is not None and prior_after.last_reviewed is not None, (
        "woven anchor must be queued for sleep replay"
    )

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_teach_cli_end_to_end(driver, tmp_path, monkeypatch, capsys):
    """`iai teach <file>` runs the whole flow against IAI_MCP_STORE."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store-root"))

    doc_file = tmp_path / "notes.md"
    doc_file.write_text(_DOC, encoding="utf-8")

    import argparse

    from iai_mcp.iai_cli import cmd_teach

    args = argparse.Namespace(
        path=str(doc_file), session_id=None, reconcile=False, json=True,
    )
    rc = cmd_teach(args)
    assert rc == 0
    import json as _json

    report = _json.loads(capsys.readouterr().out.strip())
    assert report["source"] == "notes.md"
    assert report["inserted"] >= 1
    assert report["chunks_total"] >= 1

    # Second run through the CLI is idempotent (dedup gate).
    rc = cmd_teach(args)
    assert rc == 0
    report2 = _json.loads(capsys.readouterr().out.strip())
    assert report2["inserted"] == 0, report2

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_restudy_supersedes_vanished_chunks(driver, tmp_path, monkeypatch):
    """Restudying a CHANGED file keeps memory current: chunks that no longer
    exist in the new version are hinted into the forgetting funnel (never
    deleted), surviving chunks are reinforced and left untouched."""
    from uuid import UUID

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    v1 = (
        "The hippocampus consolidates memories during sleep. "
        "Legacy section: the storage engine is LanceDB with columnar files."
    )
    first = study_document(store, text=v1, source_name="arch.md")
    assert first["inserted"] >= 1

    v2 = "The hippocampus consolidates memories during sleep."
    second = study_document(store, text=v2, source_name="arch.md")

    # v2's content differs from the single v1 chunk: the new version inserts,
    # the vanished old chunk supersedes.
    assert second["inserted"] >= 1, second
    assert second["superseded_chunks"] >= 1, second
    # v1 and v2 fit in one chunk each; the v1 chunk (with the legacy section)
    # vanished from v2, so it must now be fading — but still present verbatim.
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, centrality, literal_surface FROM records"
            " WHERE tags_json LIKE ? AND tombstoned_at IS NULL",
            ('%"doc:arch.md"%',),
        ).fetchall()
    surfaces = {str(r[0]): (float(r[1] or 0), r[2]) for r in rows}
    assert len(surfaces) >= 2, surfaces
    stale = [
        (rid, cen) for rid, (cen, surf) in surfaces.items()
        if "LanceDB" in str(store.get(UUID(rid)).literal_surface)
    ]
    assert stale, "the vanished chunk must still exist (never deleted)"
    assert all(cen == 0.0 for _rid, cen in stale), (
        f"[{driver}] vanished chunk must be hinted into the decay funnel: {stale}"
    )
    assert second["superseded_chunks"] >= 1, second


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_study_directory_walks_code_and_docs(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / ".git").mkdir()
    (proj / "README.md").write_text(
        "This project weaves memory like a hippocampus during sleep.",
        encoding="utf-8",
    )
    (proj / "src" / "engine.py").write_text(
        "def consolidate(memories):\n"
        "    return [m for m in memories if m.strength > threshold]\n",
        encoding="utf-8",
    )
    (proj / ".git" / "junk.md").write_text("never ingest vcs internals", encoding="utf-8")

    from iai_mcp.study import study_directory

    totals = study_directory(store, proj)
    assert totals["files"] == 2, totals
    assert totals["inserted"] >= 2

    with store.db._conn_lock:
        row = store.db._conn.execute("SELECT COUNT(*) FROM records").fetchone()
    n_before = int(row[0])

    totals2 = study_directory(store, proj)
    assert totals2["inserted"] == 0, f"[{driver}] restudy must be idempotent: {totals2}"
    with store.db._conn_lock:
        row = store.db._conn.execute("SELECT COUNT(*) FROM records").fetchone()
    assert int(row[0]) == n_before
