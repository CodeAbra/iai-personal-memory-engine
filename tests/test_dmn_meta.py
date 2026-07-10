from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.daemon import DmnConfig, _load_dmn_config
from iai_mcp.dmn_reflection import MetaAnalyst, ReflectionAgent
from iai_mcp.events import query_events, write_event
from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord

@pytest.fixture(autouse=True)
def _isolate_iai_dmn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_DMN_REFLECTION_WINDOW_HOURS",
        "IAI_MCP_META_ANALYST_ENABLED",
        "IAI_MCP_DMN_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)

def _make_record(
    *,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
    community_id: uuid.UUID | None = None,
    created_at: datetime | None = None,
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = created_at if created_at is not None else datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * embed_dim,
        community_id=community_id,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["t"],
    )

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )

def _seed_records_for_communities(
    store: MemoryStore,
    community_assignments: list[tuple[str, uuid.UUID]],
) -> list[uuid.UUID]:
    embed_dim = store._embed_dim
    ids: list[uuid.UUID] = []
    for i, (surface, cid) in enumerate(community_assignments):
        emb = [0.0] * embed_dim
        emb[i % embed_dim] = 1.0
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=surface,
            community_id=cid,
            embedding=emb,
        )
        store.insert(rec)
        ids.append(rec.id)
    return ids

def test_reflection_synthesize_returns_semantic_record(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)

    cid_alpha = uuid.uuid4()
    cid_beta = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha second", cid_alpha),
            ("alice topic alpha third", cid_alpha),
            ("bob topic beta", cid_beta),
            ("bob topic beta second", cid_beta),
        ],
    )

    synth = ReflectionAgent().synthesize(store, window_hours=24)

    assert synth.tier == "semantic", (
        f"synth.tier must be 'semantic'; got {synth.tier!r}"
    )

    assert isinstance(synth, MemoryRecord)
    assert isinstance(synth.id, uuid.UUID)

    assert "Daily reflection" in synth.literal_surface, (
        f"literal_surface missing 'Daily reflection' framing; "
        f"got {synth.literal_surface!r}"
    )
    assert "top topics were" in synth.literal_surface, (
        f"literal_surface missing 'top topics were' marker; "
        f"got {synth.literal_surface!r}"
    )
    assert (
        "alice topic alpha" in synth.literal_surface
        or "bob topic beta" in synth.literal_surface
    ), (
        f"literal_surface missing seeded community labels; "
        f"got {synth.literal_surface!r}"
    )

    assert isinstance(synth.provenance, list)
    assert len(synth.provenance) >= 1
    prov = synth.provenance[0]
    assert prov.get("synthesized_by") == "dmn_reflection", (
        f"provenance missing synthesized_by='dmn_reflection'; got {prov!r}"
    )
    assert prov.get("window_hours") == 24, (
        f"provenance missing window_hours echo; got {prov!r}"
    )
    assert isinstance(prov.get("topics"), list)
    assert isinstance(prov.get("captured_count"), int)
    assert isinstance(prov.get("recalled_count"), int)

    assert len(synth.embedding) == store._embed_dim
    # A REAL embedding is required: the promised "re-embed at next REM" never
    # ran, and zero-vector records surface as 0.000-score junk on every
    # degraded recall lane while occupying ANN slots.
    assert any(v != 0.0 for v in synth.embedding), (
        "synthesised record must carry a real embedding of its surface"
    )

def test_meta_analyst_snapshot_counts_correct(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    # The event vocabulary the store actually writes: recall dispatches emit
    # recall_dispatched; every insert passes the pattern-separation gate.
    for _ in range(4):
        write_event(store, "recall_dispatched", {"cue": "alice"})
    for _ in range(3):
        write_event(store, "pattern_separation_pass", {"text": "bob fact"})
    # The one per-completed-cycle event the pipeline actually writes to the
    # store events table (sleep_step_completed goes only to the
    # lifecycle-log sink and never lands in query_events).
    for _ in range(2):
        write_event(
            store,
            "cls_consolidation_run",
            {"mode": "heavy", "tier": "tier0"},
        )
    write_event(
        store,
        "sleep_step_completed",
        {"step": "SCHEMA_MINE"},
    )
    for _ in range(1):
        write_event(
            store,
            "essential_variable_breach",
            {"variable": "richclub_density"},
        )
    for _ in range(5):
        write_event(store, "erasure_agent_pass", {"erased": 1})

    snap = MetaAnalyst().snapshot(store, window_hours=24)

    assert isinstance(snap, dict)
    assert snap["recall_count"] == 4, (
        f"recall_count mismatch: expected 4, got {snap['recall_count']}"
    )
    assert snap["capture_count"] == 3, (
        f"capture_count mismatch: expected 3, got {snap['capture_count']}"
    )
    assert snap["sleep_cycles_count"] == 2, (
        f"sleep_cycles_count mismatch (one per cls_consolidation_run; the "
        f"lifecycle-only sleep_step_completed must NOT count): "
        f"expected 2, got {snap['sleep_cycles_count']}"
    )
    assert snap["breach_count"] == 1, (
        f"breach_count mismatch: expected 1, got {snap['breach_count']}"
    )
    assert snap["erasure_count"] == 5, (
        f"erasure_count mismatch: expected 5, got {snap['erasure_count']}"
    )

    assert snap["average_record_count_delta"] == -2.0, (
        f"average_record_count_delta proxy mismatch: expected -2.0, "
        f"got {snap['average_record_count_delta']!r}"
    )

    assert snap["window_hours"] == 24
    assert isinstance(snap["generated_at"], str)
    parsed = datetime.fromisoformat(snap["generated_at"])
    assert parsed.tzinfo is not None, (
        f"generated_at must be tz-aware ISO; got {snap['generated_at']!r}"
    )

def test_dmn_reflection_step_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_DMN_DRY_RUN", "false")

    store = _make_store(tmp_path)
    cid_alpha = uuid.uuid4()
    cid_beta = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha extended", cid_alpha),
            ("bob topic beta", cid_beta),
        ],
    )

    pre_count = len(store.all_records())

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_dmn_reflection(interrupt_check=None)
    assert done is True
    assert "persist_error" not in payload, (
        f"_step_dmn_reflection fell through to the outer try/except "
        f"silent-failure path; payload={payload!r}"
    )
    assert payload.get("meta_analyst_emitted") is True, (
        f"payload must surface meta_analyst_emitted=True under default "
        f"enabled config; got {payload!r}"
    )
    assert payload.get("reflection_synthesized") is True, (
        f"payload must surface reflection_synthesized=True under "
        f"dry_run=false; got {payload!r}"
    )
    assert payload.get("dry_run_mode") is False, (
        f"payload must echo dry_run_mode=False (we set the env var); "
        f"got {payload!r}"
    )

    post_count = len(store.all_records())
    assert post_count == pre_count + 1, (
        f"store record count must grow by exactly 1 (the synthesised "
        f"semantic record); pre={pre_count}, post={post_count}"
    )

    semantic_records = [
        r for r in store.all_records()
        if r.tier == "semantic"
        and any(
            (p or {}).get("synthesized_by") == "dmn_reflection"
            for p in (r.provenance or [])
        )
    ]
    assert len(semantic_records) >= 1, (
        f"no semantic record with synthesized_by='dmn_reflection' found; "
        f"all records: {[(r.tier, r.provenance) for r in store.all_records()]!r}"
    )

    health_events = query_events(
        store, kind="system_health_report", limit=10,
    )
    assert len(health_events) >= 1, (
        f"_step_dmn_reflection must emit system_health_report event "
        f"under meta_analyst_enabled=True; got {len(health_events)}"
    )
    body = health_events[0]["data"]
    assert "recall_count" in body
    assert "capture_count" in body
    assert "window_hours" in body
    assert body["dry_run_mode"] is False, (
        f"system_health_report body must echo dry_run_mode=False; "
        f"got {body!r}"
    )

def test_dmn_empty_window_inserts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely quiet window writes NOTHING: a real-embedded 'captured 0
    turns' record is a valid ANN neighbor for vague cues — junk that can
    actually surface on degraded recall."""
    monkeypatch.setenv("IAI_MCP_DMN_DRY_RUN", "false")

    store = _make_store(tmp_path)
    pre_count = len(store.all_records())

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_dmn_reflection(interrupt_check=None)
    assert done is True
    assert payload.get("reflection_synthesized") is False, payload
    assert payload.get("reflection_skipped_empty") is True, payload
    assert len(store.all_records()) == pre_count, (
        "an empty window must not grow the store"
    )


def test_dmn_dry_run_no_record_insert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_DMN_DRY_RUN", "true")

    store = _make_store(tmp_path)
    cid_alpha = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha extended", cid_alpha),
        ],
    )

    pre_count = len(store.all_records())

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_dmn_reflection(interrupt_check=None)
    assert done is True
    assert "persist_error" not in payload, (
        f"_step_dmn_reflection fell through to silent-failure path "
        f"under dry_run=true; payload={payload!r}"
    )
    assert payload["dry_run_mode"] is True
    assert payload["reflection_synthesized"] is False, (
        f"dry_run=true must leave reflection_synthesized=False; "
        f"got {payload!r}"
    )
    assert payload["meta_analyst_emitted"] is True, (
        f"meta_analyst_emitted must stay True under dry_run=true "
        f"(independent gate); got {payload!r}"
    )

    post_count = len(store.all_records())
    assert post_count == pre_count, (
        f"dry_run=true must NOT insert; pre={pre_count}, post={post_count}"
    )

    health_events = query_events(
        store, kind="system_health_report", limit=10,
    )
    assert len(health_events) >= 1, (
        f"system_health_report must emit even under dry_run=true; "
        f"got {len(health_events)}"
    )

def test_meta_analyst_disabled_skip_health_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_META_ANALYST_ENABLED", "false")
    monkeypatch.setenv("IAI_MCP_DMN_DRY_RUN", "false")

    store = _make_store(tmp_path)
    cid_alpha = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha extended", cid_alpha),
        ],
    )

    pre_count = len(store.all_records())

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_dmn_reflection(interrupt_check=None)
    assert done is True
    assert "persist_error" not in payload, (
        f"_step_dmn_reflection fell through to silent-failure path; "
        f"payload={payload!r}"
    )
    assert payload["meta_analyst_emitted"] is False, (
        f"meta_analyst_emitted must be False under enabled=false; "
        f"got {payload!r}"
    )
    assert payload["reflection_synthesized"] is True, (
        f"reflection still synthesises under meta_analyst_enabled=false "
        f"+ dry_run=false; got {payload!r}"
    )

    post_count = len(store.all_records())
    assert post_count == pre_count + 1, (
        f"store must grow by 1 under enabled=false+dry_run=false; "
        f"pre={pre_count}, post={post_count}"
    )

    health_events = query_events(
        store, kind="system_health_report", limit=10,
    )
    assert len(health_events) == 0, (
        f"meta_analyst_enabled=false must suppress ALL "
        f"system_health_report emits; got {len(health_events)}"
    )

@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_DMN_REFLECTION_WINDOW_HOURS", "abc"),
        ("IAI_MCP_DMN_REFLECTION_WINDOW_HOURS", "0"),
        ("IAI_MCP_DMN_REFLECTION_WINDOW_HOURS", "99999"),
        ("IAI_MCP_META_ANALYST_ENABLED", "not-a-bool"),
        ("IAI_MCP_META_ANALYST_ENABLED", "maybe"),
        ("IAI_MCP_DMN_DRY_RUN", "bogus"),
        ("IAI_MCP_DMN_DRY_RUN", "perhaps"),
    ],
)
def test_env_var_fail_loud_parametrized(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    monkeypatch.setenv(env_var, bad_value)

    with pytest.raises(ValueError) as excinfo:
        _load_dmn_config()

    assert env_var in str(excinfo.value), (
        f"ValueError must name the offending env var {env_var!r}; "
        f"got {excinfo.value!r}"
    )

def test_dmn_config_defaults_under_pytest() -> None:
    cfg = _load_dmn_config()
    assert isinstance(cfg, DmnConfig)
    assert cfg.reflection_window_hours == 24
    assert cfg.meta_analyst_enabled is True
    assert cfg.dry_run is True

def test_reflection_does_not_re_ingest_prior_reflections(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)

    shared_cid = uuid.uuid4()

    _seed_records_for_communities(
        store,
        [
            ("real user memory about something useful", shared_cid),
            ("another real memory from the user", shared_cid),
            ("a third real memory entry", shared_cid),
        ],
    )

    embed_dim = store._embed_dim
    prior_reflection = MemoryRecord(
        id=uuid.uuid4(),
        tier="semantic",
        literal_surface=(
            "Daily reflection: top topics were "
            "[real user memory about something]; "
            "captured 5 turns; recalled 2 times."
        ),
        aaak_index="",
        embedding=[0.0] * embed_dim,
        community_id=shared_cid,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[
            {
                "synthesized_by": "dmn_reflection",
                "window_hours": 24,
                "topics": ["real user memory about something"],
                "captured_count": 5,
                "recalled_count": 2,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        ],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        language="en",
        tags=[],
    )
    store.insert(prior_reflection)

    synth = ReflectionAgent().synthesize(store, window_hours=24)

    assert "Daily reflection" in synth.literal_surface, (
        "synthesized record must carry the 'Daily reflection' frame; "
        f"got {synth.literal_surface!r}"
    )
    topics_start = synth.literal_surface.find("top topics were [")
    topics_end = synth.literal_surface.find("]", topics_start)
    if topics_start >= 0 and topics_end > topics_start:
        topics_segment = synth.literal_surface[topics_start:topics_end + 1]
        assert "Daily reflection" not in topics_segment, (
            "prior reflection record was re-ingested as a topic label — "
            "nested 'Daily reflection:' found inside topics segment; "
            f"topics_segment={topics_segment!r}; "
            f"full literal_surface={synth.literal_surface!r}"
        )

    prov = synth.provenance[0] if synth.provenance else {}
    assert prov.get("synthesized_by") == "dmn_reflection"
