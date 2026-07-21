from __future__ import annotations

import os
import stat
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.daemon import (
    UserModelConfig,
    _load_user_model_config,
)
from iai_mcp.events import query_events, write_event
from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.lilli.cycle.sleep_pipeline import (
    STEP_PHASE,
    SleepPhase,
    SleepPipeline,
    SleepStep,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord
from iai_mcp.user_model import (
    UserModel,
    UserModelAggregator,
    UserModelPrefetcher,
    default,
    load,
    record_surprise,
    save,
)

@pytest.fixture(autouse=True)
def _isolate_iai_user_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.setenv(
        "IAI_MCP_USER_MODEL_PATH", str(tmp_path / "user_model.json"),
    )
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS",
        "IAI_MCP_USER_MODEL_PREFETCH_TOP_K",
        "IAI_MCP_USER_MODEL_DRY_RUN",
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

def test_R1_persistence_roundtrip_chmod_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "user_model.json"
    assert not target.exists(), "tmp_path must start clean"

    fresh = load()
    assert isinstance(fresh, UserModel)
    assert fresh.top_recent_topics == []
    assert fresh.tool_usage_freq == {}
    assert fresh.time_of_day_pattern == {}
    assert fresh.recent_projects == []
    assert fresh.aggregation_window_days == 30

    d = default()
    assert d.top_recent_topics == []
    assert d.tool_usage_freq == {}
    assert d.time_of_day_pattern == {}
    assert d.recent_projects == []
    assert d.aggregation_window_days == 30

    model = UserModel(
        top_recent_topics=["python async", "torchhd hdc", "alice notes"],
        tool_usage_freq={"memory_recall": 42, "memory_capture": 7},
        time_of_day_pattern={9: 5, 14: 12, 22: 1},
        recent_projects=["project-alpha"],
        last_updated=datetime(2026, 5, 16, 8, 42, 13, tzinfo=timezone.utc),
        aggregation_window_days=14,
    )
    save(model)

    assert target.exists(), "save() must materialise the file at tmp path"

    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode == 0o600, f"file mode must be 0o600, got {oct(mode)}"

    loaded = load()
    assert loaded.top_recent_topics == [
        "python async", "torchhd hdc", "alice notes",
    ]
    assert loaded.tool_usage_freq == {"memory_recall": 42, "memory_capture": 7}
    assert loaded.time_of_day_pattern == {9: 5, 14: 12, 22: 1}
    assert all(isinstance(k, int) for k in loaded.time_of_day_pattern.keys())
    assert loaded.recent_projects == ["project-alpha"]
    assert loaded.aggregation_window_days == 14
    assert loaded.last_updated == datetime(
        2026, 5, 16, 8, 42, 13, tzinfo=timezone.utc,
    )

def test_R2_aggregator_computes_known_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(tmp_path)

    cid_a = uuid.uuid4()
    cid_b = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_a),
            ("alice topic alpha extended", cid_a),
            ("bob topic beta", cid_b),
            ("bob topic beta extended", cid_b),
        ],
    )

    for _ in range(3):
        write_event(store, "retrieval_used", {"tool": "retrieval_used"})

    agg = UserModelAggregator()
    model = agg.aggregate(store, window_days=30)

    assert len(model.top_recent_topics) == 2, (
        f"expected 2 community labels, got {model.top_recent_topics!r}"
    )
    labels = set(model.top_recent_topics)
    assert any("alice topic alpha" in lbl for lbl in labels)
    assert any("bob topic beta" in lbl for lbl in labels)

    assert "retrieval_used" in model.tool_usage_freq, (
        f"retrieval_used missing from tool_usage_freq: "
        f"{model.tool_usage_freq!r}"
    )
    assert model.tool_usage_freq["retrieval_used"] >= 3

    assert len(model.time_of_day_pattern) >= 1
    current_hour = datetime.now(timezone.utc).hour
    assert current_hour in model.time_of_day_pattern, (
        f"current hour {current_hour} missing from "
        f"time_of_day_pattern={model.time_of_day_pattern!r}"
    )
    assert model.time_of_day_pattern[current_hour] >= 3
    assert all(isinstance(k, int) for k in model.time_of_day_pattern.keys())

    assert model.aggregation_window_days == 30

def test_R3_sleep_step_position_and_dispatch(tmp_path: Path) -> None:
    assert SleepStep.USER_MODEL_UPDATE.value == 10
    assert STEP_PHASE[SleepStep.USER_MODEL_UPDATE] == SleepPhase.REM

    order = SleepPipeline._STEP_ORDER
    idx_user = order.index(SleepStep.USER_MODEL_UPDATE)
    idx_recon = order.index(SleepStep.RECONSOLIDATION)
    idx_crisis = order.index(SleepStep.CRISIS_RECLUSTER)
    assert idx_recon < idx_user < idx_crisis, (
        f"USER_MODEL_UPDATE must sit strictly between RECONSOLIDATION and "
        f"CRISIS_RECLUSTER in _STEP_ORDER; got idx_recon={idx_recon}, "
        f"idx_user={idx_user}, idx_crisis={idx_crisis}"
    )
    assert idx_user == idx_recon + 1, (
        f"USER_MODEL_UPDATE must directly follow RECONSOLIDATION; "
        f"got idx_user={idx_user}, idx_recon={idx_recon}"
    )

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=None, lifecycle_state_path=lifecycle_path,
    )
    methods = pipeline._step_methods
    assert SleepStep.USER_MODEL_UPDATE in methods, (
        f"_step_methods missing USER_MODEL_UPDATE entry; "
        f"keys={list(methods.keys())!r}"
    )
    assert methods[SleepStep.USER_MODEL_UPDATE].__func__ is (
        SleepPipeline._step_user_model_update
    ), "USER_MODEL_UPDATE must dispatch to _step_user_model_update"

def test_R4_prefetcher_returns_topic_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(tmp_path)

    cid_match_1 = uuid.uuid4()
    cid_match_2 = uuid.uuid4()
    cid_unmatched = uuid.uuid4()
    ids = _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_match_1),
            ("alice topic alpha second record", cid_match_1),
            ("bob topic beta", cid_match_2),
            ("bob topic beta second record", cid_match_2),
            ("unmatched gamma topic", cid_unmatched),
            ("unmatched gamma topic second record", cid_unmatched),
        ],
    )
    label_match_1 = "alice topic alpha second record"
    label_match_2 = "bob topic beta second record"
    label_unmatched = "unmatched gamma topic second record"

    model = UserModel(
        top_recent_topics=[label_match_1, label_match_2],
        tool_usage_freq={},
        time_of_day_pattern={},
        recent_projects=[],
        last_updated=datetime.now(timezone.utc),
        aggregation_window_days=30,
    )

    result = UserModelPrefetcher().prefetch(store, model, top_k=10)
    assert isinstance(result, list)
    assert len(result) >= 1, "prefetcher must return at least one match"
    assert len(result) <= 10

    matched_ids = {str(rid) for rid in ids[:4]}
    unmatched_ids = {str(rid) for rid in ids[4:]}
    for rid in result:
        assert rid in matched_ids, (
            f"prefetcher returned id {rid} which is not in a matched "
            f"community; matched={matched_ids}, unmatched={unmatched_ids}"
        )
        assert rid not in unmatched_ids, (
            f"prefetcher leaked unmatched community id {rid}; "
            f"label_unmatched={label_unmatched!r}"
        )

def test_R5_session_start_prefetch_integration_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp import core as _core_mod  # noqa: F401
    from iai_mcp.core import dispatch  # noqa: F401
    from iai_mcp.user_model import UserModelPrefetcher as _PF
    from iai_mcp.user_model import load as _load
    from iai_mcp.daemon import _load_user_model_config as _lcfg

    monkeypatch.setenv("IAI_MCP_USER_MODEL_DRY_RUN", "false")

    store = _make_store(tmp_path)
    cid_alpha = uuid.uuid4()
    cid_beta = uuid.uuid4()
    seeded = _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_alpha),
            ("alice topic alpha second record", cid_alpha),
            ("bob topic beta", cid_beta),
            ("bob topic beta second record", cid_beta),
        ],
    )

    label_alpha = "alice topic alpha second record"
    label_beta = "bob topic beta second record"
    model = UserModel(
        top_recent_topics=[label_alpha, label_beta],
        tool_usage_freq={"memory_recall": 5},
        time_of_day_pattern={datetime.now(timezone.utc).hour: 5},
        recent_projects=[],
        last_updated=datetime.now(timezone.utc),
        aggregation_window_days=30,
    )
    save(model)

    cfg = _lcfg()
    loaded = _load()
    assert loaded.top_recent_topics == [label_alpha, label_beta], (
        "save+load round-trip lost the topics; check tmp_path redirect"
    )
    prefetched = _PF().prefetch(store, loaded, top_k=cfg.prefetch_top_k)
    assert isinstance(prefetched, list)
    assert len(prefetched) >= 1, (
        "prefetcher returned no ids despite matching topics + a seeded store; "
        "core.py SessionStart augmentation would silently no-op in production"
    )
    seeded_str = {str(rid) for rid in seeded}
    for rid in prefetched:
        assert rid in seeded_str, (
            f"prefetcher returned non-seeded id {rid}; seeded={seeded_str}"
        )

def test_R6_dry_run_skips_save_but_emits_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(tmp_path)

    cid_a = uuid.uuid4()
    _seed_records_for_communities(
        store,
        [
            ("alice topic alpha", cid_a),
            ("alice topic alpha second", cid_a),
        ],
    )

    target = tmp_path / "user_model.json"
    assert not target.exists(), "tmp_path must start clean"

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store, lifecycle_state_path=lifecycle_path,
    )

    done, payload = pipeline._step_user_model_update(interrupt_check=None)
    assert done is True
    assert payload["dry_run"] is True, (
        f"dry_run pytest-default must surface in payload; got {payload!r}"
    )
    assert payload["topics_count"] >= 1

    assert not target.exists(), (
        "dry_run path must NOT persist user_model.json; "
        f"file appeared at {target}"
    )

    events = query_events(store, kind="user_model_aggregate_pass", limit=10)
    assert len(events) == 1, (
        f"_step_user_model_update must emit exactly one event, "
        f"got {len(events)}"
    )
    body = events[0]["data"]
    assert body["dry_run_mode"] is True, (
        f"event must tag dry_run_mode=True under pytest default; got {body!r}"
    )
    for key in (
        "topics_count",
        "tools_count",
        "hours_count",
        "projects_count",
        "window_days",
        "dry_run_mode",
    ):
        assert key in body, f"event body missing key {key!r}; body={body!r}"
    assert isinstance(body["topics_count"], int)
    assert isinstance(body["window_days"], int)

def test_R6_record_surprise_emits_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(tmp_path)
    record_surprise(store, predicted_topic="alpha", actual_topic="beta")
    events = query_events(store, kind="user_model_surprise", limit=10)
    assert len(events) == 1, (
        f"record_surprise must emit exactly one event, got {len(events)}"
    )
    body = events[0]["data"]
    assert body["predicted_topic"] == "alpha"
    assert body["actual_topic"] == "beta"
    assert "dry_run_mode" in body
    assert isinstance(body["dry_run_mode"], bool)
    assert body["dry_run_mode"] is True

@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS", "0"),
        ("IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS", "366"),
        ("IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS", "nan"),
        ("IAI_MCP_USER_MODEL_PREFETCH_TOP_K", "0"),
        ("IAI_MCP_USER_MODEL_PREFETCH_TOP_K", "200"),
        ("IAI_MCP_USER_MODEL_PREFETCH_TOP_K", "not-an-int"),
        ("IAI_MCP_USER_MODEL_DRY_RUN", "maybe"),
        ("IAI_MCP_USER_MODEL_DRY_RUN", "banana"),
    ],
)
def test_R7_invalid_env_var_raises_ValueError_naming_var(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_user_model_config()

def test_R7_defaults_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS",
        "IAI_MCP_USER_MODEL_PREFETCH_TOP_K",
        "IAI_MCP_USER_MODEL_PATH",
        "IAI_MCP_USER_MODEL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = _load_user_model_config()
    assert isinstance(cfg, UserModelConfig)
    assert cfg.aggregation_window_days == 30
    assert cfg.prefetch_top_k == 10
    assert cfg.user_model_path == "~/.iai-mcp/user_model.json"
    assert cfg.dry_run is True

if __name__ == "__main__":  # pragma: no cover -- direct-run convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
