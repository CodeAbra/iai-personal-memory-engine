from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.lifecycle_event_log import LifecycleEventLog
from iai_mcp.lifecycle_state import default_state, load_state, save_state
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline, SleepStep
from iai_mcp.lilli.profile.community_names import load_community_names
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _vec(seed: int) -> "list[float]":
    v = [0.0] * EMBED_DIM
    v[seed % EMBED_DIM] = 1.0
    return v


def _rec(
    text: str, seed: int, *, community_id: "UUID | None" = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text, aaak_index="",
        embedding=_vec(seed), community_id=community_id, centrality=0.0,
        detail_level=1, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=["t"],
        language="en",
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-topic-naming")
    from iai_mcp.store import MemoryStore, flush_record_buffer
    s = MemoryStore(path=tmp_path / "lancedb")
    s._flush = lambda: flush_record_buffer(s)
    return s


@pytest.fixture
def pipeline(tmp_path: Path, store):
    return SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=tmp_path / "logs"),
    )


JAZZ_TEXTS = [
    "alice loves jazz music at the downtown club",
    "alice heard the jazz trio play until midnight",
    "alice keeps jazz records on the shelf in the study",
    "alice hums a jazz tune while cooking dinner",
]
OTHER_TEXT = "alice wrote a grocery list for the week ahead"


def _seed_jazz_community(store, *, stored_cid: UUID) -> "list[UUID]":
    recs = [
        _rec(text, i, community_id=stored_cid) for i, text in enumerate(JAZZ_TEXTS)
    ]
    other = _rec(OTHER_TEXT, 99, community_id=None)
    for r in recs + [other]:
        store.insert(r)
    store._flush()
    return [r.id for r in recs]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_sleep_step_registered_at_tail() -> None:
    assert SleepStep.COMMUNITY_NAMING.value == 17
    assert SleepPipeline._STEP_ORDER.index(SleepStep.COMMUNITY_NAMING) == 16
    assert SleepPipeline._STEP_ORDER.index(SleepStep.EMBEDDING_INTEGRITY) == 15
    assert callable(SleepPipeline._step_community_naming)


# ---------------------------------------------------------------------------
# Namespace bridge: same members, two id spaces, one name
# ---------------------------------------------------------------------------


def test_names_both_groupings_with_matching_string(
    pipeline: SleepPipeline, store, monkeypatch,
) -> None:
    stored_cid = uuid4()
    gate_cid = uuid4()
    member_ids = _seed_jazz_community(store, stored_cid=stored_cid)

    import iai_mcp.runtime_graph_cache as rgc_mod
    from iai_mcp.community import CommunityAssignment

    fake_assignment = CommunityAssignment(
        node_to_community={mid: gate_cid for mid in member_ids},
    )
    monkeypatch.setattr(
        rgc_mod, "load_recall_structural",
        lambda _store: (fake_assignment, {}, 0, "test", {}),
    )

    done, payload = pipeline._step_community_naming(interrupt_check=None)
    assert done is True
    assert payload["communities_seen"] == 2
    assert payload["names_persisted"] == 2

    loaded = load_community_names(store)
    reverse_index = loaded["reverse_index"]
    # Both cids wrap the identical member set -- the union-find cluster
    # bridge recognizes them as ONE topic (not a same-word collision
    # between two different topics), so no second term is appended and
    # both namespaces land on the bare top token.
    assert reverse_index[str(stored_cid)] == "jazz"
    assert reverse_index[str(gate_cid)] == "jazz"


def test_overlapping_but_not_identical_members_share_one_name(
    pipeline: SleepPipeline, store, monkeypatch,
) -> None:
    # stored = {r0,r1,r2,r3}; gate = {r0,r1,r2,r4} -- overlapping but NOT
    # identical membership, the realistic shape of two independent
    # clustering runs over the same underlying topic.
    stored_cid = uuid4()
    gate_cid = uuid4()
    member_ids = _seed_jazz_community(store, stored_cid=stored_cid)
    extra = _rec("alice replays a jazz solo she recorded last spring", 50)
    store.insert(extra)
    store._flush()

    import iai_mcp.runtime_graph_cache as rgc_mod
    from iai_mcp.community import CommunityAssignment

    gate_members = member_ids[:3] + [extra.id]
    fake_assignment = CommunityAssignment(
        node_to_community={mid: gate_cid for mid in gate_members},
    )
    monkeypatch.setattr(
        rgc_mod, "load_recall_structural",
        lambda _store: (fake_assignment, {}, 0, "test", {}),
    )

    done, payload = pipeline._step_community_naming(interrupt_check=None)
    assert done is True
    reverse_index = load_community_names(store)["reverse_index"]
    run1_stored = reverse_index[str(stored_cid)]
    run1_gate = reverse_index[str(gate_cid)]
    assert run1_stored == run1_gate == "jazz"

    # A second, independent night must reproduce the identical name in both
    # namespaces via hysteresis, not just by accident of re-deriving the
    # same top token.
    done2, _ = pipeline._step_community_naming(interrupt_check=None)
    assert done2 is True
    reverse_index2 = load_community_names(store)["reverse_index"]
    assert reverse_index2[str(stored_cid)] == run1_stored
    assert reverse_index2[str(gate_cid)] == run1_gate


def test_hysteresis_protects_disambiguated_names_across_reruns(
    pipeline: SleepPipeline, store, monkeypatch,
) -> None:
    # Two genuinely DISJOINT clusters (zero shared members) that both derive
    # "jazz" as their top token, with DIFFERENT second terms -- the exact
    # shape the reverse_index-as-prior bug broke: run 1's compound names
    # ("jazz-trio", "jazz-vinyl") must reproduce identically on run 2, which
    # is only possible when hysteresis reads the bare-token base_index, not
    # the compound reverse_index.
    cid_a = uuid4()
    cid_b = uuid4()
    a_texts = [
        "listening to jazz trio recordings this evening",
        "our jazz trio rehearsed again tonight downtown",
        "jazz plays softly during the morning commute",
        "jazz fills the quiet apartment on weekends",
    ]
    b_texts = [
        "jazz vinyl records line the shelf downstairs",
        "cleaning jazz vinyl sleeves takes forever",
        "jazz drifts from the kitchen radio",
        "jazz hums quietly through the hallway speakers",
    ]
    for i, t in enumerate(a_texts):
        store.insert(_rec(t, 300 + i, community_id=cid_a))
    for i, t in enumerate(b_texts):
        store.insert(_rec(t, 400 + i, community_id=cid_b))
    store._flush()

    import iai_mcp.runtime_graph_cache as rgc_mod
    from iai_mcp.community import CommunityAssignment

    monkeypatch.setattr(
        rgc_mod, "load_recall_structural",
        lambda _store: (CommunityAssignment(), {}, 0, "test", {}),
    )

    def _inflate_n_docs() -> None:
        # Simulate a large surrounding corpus without inserting hundreds of
        # filler records: idf grows with n_docs for every token, and at a
        # large n_docs the shared token's ("jazz", tf=4/cluster) tf
        # advantage outweighs the rarer per-cluster word's ("trio"/"vinyl",
        # tf=2/cluster) idf edge -- the realistic regime where a broadly
        # shared topic word beats an incidental rarer one.
        store.lexical_search("warm", k=1)
        store._lexical_idx._n_docs = 500

    _inflate_n_docs()
    done, _ = pipeline._step_community_naming(interrupt_check=None)
    assert done is True
    reverse_index = load_community_names(store)["reverse_index"]
    name_a_1 = reverse_index[str(cid_a)]
    name_b_1 = reverse_index[str(cid_b)]
    assert name_a_1 != name_b_1, (
        "disjoint clusters sharing a top token must disambiguate"
    )
    assert name_a_1 == "jazz-trio"
    assert name_b_1 == "jazz-vinyl"

    # Shift cluster A's own ranking: two more trio-heavy members push
    # "trio" strictly ahead of "jazz" on a FRESH (no-prior) computation --
    # without hysteresis reading the bare-token base_index, run 2 would
    # flip cluster A's base to "trio" (and, losing its collision with
    # cluster B, drop the compound suffix on both sides entirely).
    store.insert(_rec(
        "the jazz trio added a second trio set this month", 310,
        community_id=cid_a,
    ))
    store.insert(_rec(
        "another trio joined our jazz trio for the finale", 311,
        community_id=cid_a,
    ))
    store._flush()

    _inflate_n_docs()
    done2, _ = pipeline._step_community_naming(interrupt_check=None)
    assert done2 is True
    reverse_index2 = load_community_names(store)["reverse_index"]
    assert reverse_index2[str(cid_a)] == name_a_1, (
        "hysteresis must retain the prior base name despite the fresh "
        "ranking shift"
    )
    assert reverse_index2[str(cid_b)] == name_b_1


def test_hysteresis_protects_disambiguation_term_across_reruns(
    pipeline: SleepPipeline, store, monkeypatch,
) -> None:
    # Two disjoint clusters collide on the base token "jazz" both nights, so
    # each carries a compound display name. Cluster A's OWN second-ranked
    # (non-base) candidate rotates from "trio" to "combo" between nights on
    # a fresh, no-hysteresis computation -- the base token never moves, only
    # the disambiguation term. Without sticky hysteresis on the second term,
    # the compound name (and the monotropism_depth key it backs) churns even
    # though cluster A names the same underlying topic both nights.
    cid_a = uuid4()
    cid_b = uuid4()
    a_texts_1 = [
        "jazz jazz jazz trio trio plays downtown",
        "jazz jazz jazz trio trio returns tonight",
        "jazz jazz jazz combo debuts weekly",
        "jazz jazz jazz combo closes strong",
    ]
    b_texts = [
        "jazz jazz jazz trumpet plays uptown",
        "jazz jazz jazz trumpet returns nightly",
        "jazz jazz jazz trumpet debuts weekly",
        "jazz jazz jazz trumpet closes softly",
    ]
    for i, t in enumerate(a_texts_1):
        store.insert(_rec(t, 300 + i, community_id=cid_a))
    for i, t in enumerate(b_texts):
        store.insert(_rec(t, 400 + i, community_id=cid_b))
    store._flush()

    import iai_mcp.runtime_graph_cache as rgc_mod
    from iai_mcp.community import CommunityAssignment

    monkeypatch.setattr(
        rgc_mod, "load_recall_structural",
        lambda _store: (CommunityAssignment(), {}, 0, "test", {}),
    )

    def _inflate_n_docs() -> None:
        store.lexical_search("warm", k=1)
        store._lexical_idx._n_docs = 500

    _inflate_n_docs()
    done, _ = pipeline._step_community_naming(interrupt_check=None)
    assert done is True
    reverse_index = load_community_names(store)["reverse_index"]
    name_a_1 = reverse_index[str(cid_a)]
    name_b_1 = reverse_index[str(cid_b)]
    assert name_a_1 == "jazz-trio"
    assert name_b_1 == "jazz-trumpet"

    # Push "combo" ahead of "trio" for a FRESH (no-prior) computation on
    # cluster A only -- cluster B is untouched, so its own term must not
    # move either.
    store.insert(_rec(
        "jazz jazz jazz combo combo combo returns strong", 310,
        community_id=cid_a,
    ))
    store.insert(_rec(
        "jazz jazz jazz combo combo combo plays again", 311,
        community_id=cid_a,
    ))
    store._flush()

    _inflate_n_docs()
    done2, _ = pipeline._step_community_naming(interrupt_check=None)
    assert done2 is True
    reverse_index2 = load_community_names(store)["reverse_index"]
    assert reverse_index2[str(cid_a)] == name_a_1, (
        "hysteresis must retain the prior disambiguation term despite the "
        "fresh ranking shift"
    )
    assert reverse_index2[str(cid_b)] == name_b_1


def test_store_without_communities_returns_zero_counts(
    pipeline: SleepPipeline, store,
) -> None:
    r = _rec("a lone record with no community assignment", 1, community_id=None)
    store.insert(r)
    store._flush()

    done, payload = pipeline._step_community_naming(interrupt_check=None)
    assert done is True
    assert payload == {"communities_seen": 0, "names_persisted": 0}


def test_no_store_never_raises(tmp_path: Path) -> None:
    pipeline = SleepPipeline(
        store=None,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=tmp_path / "logs"),
    )
    done, payload = pipeline._step_community_naming(interrupt_check=None)
    assert done is True
    assert payload == {"communities_seen": 0, "names_persisted": 0}


# ---------------------------------------------------------------------------
# Hysteresis across a re-run with drifted corpus
# ---------------------------------------------------------------------------


def test_hysteresis_across_reruns_with_drift(
    pipeline: SleepPipeline, store, monkeypatch,
) -> None:
    stored_cid = uuid4()
    member_ids = _seed_jazz_community(store, stored_cid=stored_cid)

    import iai_mcp.runtime_graph_cache as rgc_mod
    from iai_mcp.community import CommunityAssignment

    monkeypatch.setattr(
        rgc_mod, "load_recall_structural",
        lambda _store: (CommunityAssignment(), {}, 0, "test", {}),
    )

    done, payload = pipeline._step_community_naming(interrupt_check=None)
    assert done is True
    first_name = load_community_names(store)["reverse_index"][str(stored_cid)]
    assert first_name == "jazz"

    # Drift the corpus: many new unrelated records dilute the lexical index
    # without touching the jazz community's own members.
    for i in range(6):
        store.insert(_rec(f"unrelated filler entry number {i} zz", 200 + i))
    store._flush()

    done2, payload2 = pipeline._step_community_naming(interrupt_check=None)
    assert done2 is True
    second_name = load_community_names(store)["reverse_index"][str(stored_cid)]
    assert second_name == first_name, "a still-qualifying name must survive drift"


# ---------------------------------------------------------------------------
# double_empathy: the naming step writes no memory record, no plaintext name
# ---------------------------------------------------------------------------


def test_step_writes_no_memory_records(
    pipeline: SleepPipeline, store, monkeypatch,
) -> None:
    stored_cid = uuid4()
    _seed_jazz_community(store, stored_cid=stored_cid)

    import iai_mcp.runtime_graph_cache as rgc_mod
    from iai_mcp.community import CommunityAssignment

    monkeypatch.setattr(
        rgc_mod, "load_recall_structural",
        lambda _store: (CommunityAssignment(), {}, 0, "test", {}),
    )

    with store.db.ro_conn() as conn:
        before = {
            str(row["id"]): row["updated_at"]
            for row in conn.execute(
                "SELECT id, updated_at FROM records",
            ).fetchall()
        }

    pipeline._step_community_naming(interrupt_check=None)

    with store.db.ro_conn() as conn:
        after = {
            str(row["id"]): row["updated_at"]
            for row in conn.execute(
                "SELECT id, updated_at FROM records",
            ).fetchall()
        }
    assert before == after, "community naming must never touch a memory record"


def test_no_name_string_in_any_event(
    pipeline: SleepPipeline, store, monkeypatch,
) -> None:
    stored_cid = uuid4()
    _seed_jazz_community(store, stored_cid=stored_cid)

    import iai_mcp.runtime_graph_cache as rgc_mod
    from iai_mcp.community import CommunityAssignment

    monkeypatch.setattr(
        rgc_mod, "load_recall_structural",
        lambda _store: (CommunityAssignment(), {}, 0, "test", {}),
    )

    done, payload = pipeline._step_community_naming(interrupt_check=None)
    assert done is True
    name = load_community_names(store)["reverse_index"][str(stored_cid)]

    for k, v in payload.items():
        assert name not in str(v), f"topic name leaked into payload field {k!r}"
    for events in (pipeline._event_log.read_all(),):
        for row in events:
            assert name not in repr(row), "topic name leaked into a lifecycle event"


# ---------------------------------------------------------------------------
# WAL recovery: an old cycle persisted at an index <= 15 (EMBEDDING_INTEGRITY)
# resumes and runs COMMUNITY_NAMING exactly once, without disturbing the
# fixed position of every earlier step.
# ---------------------------------------------------------------------------


def _noop_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_path = tmp_path / "lifecycle_state.json"
    event_log = LifecycleEventLog(log_dir=tmp_path / "logs")
    pipeline = SleepPipeline(store=None, lifecycle_state_path=state_path, event_log=event_log)
    calls: "list[SleepStep]" = []

    def _make_noop(step: SleepStep):
        def _noop(_interrupt_check):
            calls.append(step)
            return True, {}
        return _noop

    for step in SleepPipeline._STEP_ORDER:
        method_name = "_step_" + step.name.lower()
        monkeypatch.setattr(pipeline, method_name, _make_noop(step))
    return pipeline, calls, state_path


def test_wal_recovery_from_curiosity_mine_runs_tail_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, calls, state_path = _noop_pipeline(tmp_path, monkeypatch)
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": SleepPipeline._STEP_ORDER.index(
            SleepStep.CURIOSITY_MINE,
        ),
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    pipeline.run()

    assert calls == [
        SleepStep.EMBEDDING_INTEGRITY,
        SleepStep.COMMUNITY_NAMING,
        SleepStep.RECONSOLIDATION_VALENCE,
        SleepStep.PROC_MINE,
        SleepStep.TRANSCRIPT_SWEEP_BACKSTOP,
    ]
    after = load_state(state_path)
    assert after["sleep_cycle_progress"] is None


def test_wal_recovery_from_embedding_integrity_runs_tail_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, calls, state_path = _noop_pipeline(tmp_path, monkeypatch)
    record = default_state()
    record["sleep_cycle_progress"] = {
        "last_completed_index": SleepPipeline._STEP_ORDER.index(
            SleepStep.EMBEDDING_INTEGRITY,
        ),
        "attempt": 0,
        "last_error": None,
        "started_at": "2026-05-02T00:00:00+00:00",
    }
    save_state(record, state_path)

    pipeline.run()

    assert calls == [
        SleepStep.COMMUNITY_NAMING,
        SleepStep.RECONSOLIDATION_VALENCE,
        SleepStep.PROC_MINE,
        SleepStep.TRANSCRIPT_SWEEP_BACKSTOP,
    ]
    after = load_state(state_path)
    assert after["sleep_cycle_progress"] is None


# ---------------------------------------------------------------------------
# Interrupt-awareness: an interrupt mid-cycle writes nothing.
# ---------------------------------------------------------------------------


def test_interrupt_writes_nothing(
    pipeline: SleepPipeline, store, monkeypatch,
) -> None:
    stored_cid = uuid4()
    _seed_jazz_community(store, stored_cid=stored_cid)

    import iai_mcp.runtime_graph_cache as rgc_mod
    from iai_mcp.community import CommunityAssignment

    monkeypatch.setattr(
        rgc_mod, "load_recall_structural",
        lambda _store: (CommunityAssignment(), {}, 0, "test", {}),
    )

    done, payload = pipeline._step_community_naming(interrupt_check=lambda: True)
    assert done is False
    assert payload == {}
    assert load_community_names(store) == {}
