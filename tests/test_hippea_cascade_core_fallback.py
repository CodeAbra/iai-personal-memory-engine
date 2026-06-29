from __future__ import annotations

import inspect
from pathlib import Path
from unittest import mock
from uuid import UUID, uuid4

import pytest

from iai_mcp import hippea_cascade
from iai_mcp.store import MemoryStore


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


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


@pytest.fixture(autouse=True)
def _reset_daemon_lru():
    hippea_cascade._warm_lru.clear()
    yield
    hippea_cascade._warm_lru.clear()


@pytest.fixture(autouse=True)
def _reset_core_state():
    from iai_mcp import core as _core

    lru = getattr(_core, "_CORE_WARM_LRU", None)
    fired = getattr(_core, "_CORE_CASCADE_FIRED_PER_SESSION", None)
    if lru is not None:
        lru.clear()
    if fired is not None:
        fired.clear()
    yield
    if lru is not None:
        lru.clear()
    if fired is not None:
        fired.clear()


def _make_assignment_with_communities(*community_ids):
    class _A:
        def __init__(self, mid):
            self.mid_regions = mid
            self.top_communities = list(mid.keys())

    return _A({cid: [] for cid in community_ids})


def test_compute_core_side_warm_snapshot_exists_and_is_sync():
    assert hasattr(hippea_cascade, "compute_core_side_warm_snapshot")
    fn = hippea_cascade.compute_core_side_warm_snapshot
    assert not inspect.iscoroutinefunction(fn)


def test_compute_core_side_warm_snapshot_respects_max_records(
    store, monkeypatch
):
    c1, c2, c3 = uuid4(), uuid4(), uuid4()
    assignment = _make_assignment_with_communities(c1, c2, c3)
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c1, c2, c3],
    )
    fake_ids = [uuid4() for _ in range(60)]

    def _per_c(_s, _a, cid, n):
        return fake_ids[:n]

    monkeypatch.setattr(hippea_cascade, "_top_n_records_by_centrality", _per_c)

    result = hippea_cascade.compute_core_side_warm_snapshot(
        store, assignment, top_k=3, max_records=50,
    )
    assert isinstance(result, list)
    assert len(result) <= 50
    assert all(isinstance(r, UUID) for r in result)


def test_compute_core_side_warm_snapshot_empty_when_no_salient(store, monkeypatch):
    assignment = _make_assignment_with_communities()
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [],
    )
    result = hippea_cascade.compute_core_side_warm_snapshot(store, assignment)
    assert result == []


def test_compute_core_side_warm_snapshot_is_read_only(store, monkeypatch):
    c1 = uuid4()
    assignment = _make_assignment_with_communities(c1)
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c1],
    )
    monkeypatch.setattr(
        hippea_cascade, "_top_n_records_by_centrality",
        lambda *a, **kw: [],
    )
    before = store.db.open_table("records").count_rows()
    for _ in range(5):
        hippea_cascade.compute_core_side_warm_snapshot(store, assignment)
    after = store.db.open_table("records").count_rows()
    assert before == after


def test_compute_core_side_warm_snapshot_does_not_touch_daemon_lru(
    store, monkeypatch
):
    c1 = uuid4()
    assignment = _make_assignment_with_communities(c1)
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c1],
    )
    monkeypatch.setattr(
        hippea_cascade, "_top_n_records_by_centrality",
        lambda *a, **kw: [uuid4() for _ in range(5)],
    )
    assert len(hippea_cascade._warm_lru) == 0
    hippea_cascade.compute_core_side_warm_snapshot(store, assignment)
    assert len(hippea_cascade._warm_lru) == 0


def test_compute_core_side_warm_snapshot_honours_topk_ranking(store, monkeypatch):
    c_top = uuid4()
    c_mid = uuid4()
    c_low = uuid4()
    assignment = _make_assignment_with_communities(c_top, c_mid, c_low)
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c_top, c_mid],
    )
    calls: list[UUID] = []

    def _per_c(_s, _a, cid, n):
        calls.append(cid)
        return []

    monkeypatch.setattr(hippea_cascade, "_top_n_records_by_centrality", _per_c)
    hippea_cascade.compute_core_side_warm_snapshot(
        store, assignment, top_k=2, max_records=10,
    )
    assert c_top in calls
    assert c_mid in calls
    assert c_low not in calls


def test_hippea_cascade_module_has_no_anthropic_import():
    source = Path(hippea_cascade.__file__).read_text()
    assert "import anthropic" not in source
    assert "ANTHROPIC_API_KEY" not in source
    assert " from anthropic" not in source


@pytest.mark.perf
def test_compute_core_side_warm_snapshot_is_fast(store, monkeypatch):
    import time

    c1 = uuid4()
    assignment = _make_assignment_with_communities(c1)
    monkeypatch.setattr(
        hippea_cascade, "compute_salient_communities",
        lambda s, a, **kw: [c1],
    )
    monkeypatch.setattr(
        hippea_cascade, "_top_n_records_by_centrality",
        lambda *a, **kw: [uuid4() for _ in range(50)],
    )
    t0 = time.perf_counter()
    result = hippea_cascade.compute_core_side_warm_snapshot(store, assignment)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 100
    assert len(result) == 50


def test_core_warm_lru_module_level_ttlcache():
    from iai_mcp import core as _core

    assert hasattr(_core, "_CORE_WARM_LRU")
    lru = _core._CORE_WARM_LRU
    assert hasattr(lru, "__setitem__")
    assert hasattr(lru, "__getitem__")
    assert getattr(lru, "maxsize", None) == 50


def test_core_cascade_fired_per_session_module_level_set():
    from iai_mcp import core as _core

    assert hasattr(_core, "_CORE_CASCADE_FIRED_PER_SESSION")
    assert isinstance(_core._CORE_CASCADE_FIRED_PER_SESSION, set)


def _invoke_first_turn_hook(session_id="sess-a", cue="hello"):
    from iai_mcp import core as _core

    response: dict = {}
    params = {"session_id": session_id, "cue": cue}

    store = mock.MagicMock()
    store.get = mock.MagicMock(return_value=None)

    with mock.patch("iai_mcp.daemon_state.consume_first_turn", return_value=True), \
         mock.patch("iai_mcp.daemon_state.load_state", return_value={}):
        with mock.patch(
            "iai_mcp.retrieve.recall",
            return_value=mock.MagicMock(hits=[], budget_used=0, anti_hits=[]),
        ), mock.patch(
            "iai_mcp.runtime_graph_cache.load_recall_structural",
            return_value=(_make_assignment_with_communities(), None, 0, "test"),
        ):
            _core._first_turn_recall_hook(response, params=params, store=store)
    return response


def test_empty_daemon_snapshot_triggers_core_cascade():
    from iai_mcp import core as _core

    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4() for _ in range(3)],
    ) as css:
        _invoke_first_turn_hook(session_id="sess-empty")
        assert css.call_count == 1
        assert "sess-empty" in _core._CORE_CASCADE_FIRED_PER_SESSION


def test_same_session_does_not_refire_cascade():
    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4() for _ in range(3)],
    ) as css:
        _invoke_first_turn_hook(session_id="sess-idem")
        _invoke_first_turn_hook(session_id="sess-idem")
        _invoke_first_turn_hook(session_id="sess-idem")
        assert css.call_count == 1


def test_non_empty_daemon_snapshot_skips_core_cascade():
    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[uuid4()]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[],
    ) as css:
        _invoke_first_turn_hook(session_id="sess-daemon-warm")
        assert css.call_count == 0


def test_core_cascade_failure_is_silent():
    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        side_effect=RuntimeError("boom"),
    ):
        response = _invoke_first_turn_hook(session_id="sess-bad-cascade")
    assert "first_turn_recall" in response


def test_m04_regression_fence_cascade_is_read_only():
    observed_results = []

    def _recall_side_effect(**kw):
        r = mock.MagicMock(hits=[mock.MagicMock(record_id=uuid4())], budget_used=10, anti_hits=[])
        observed_results.append(r)
        return r

    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4() for _ in range(5)],
    ), mock.patch(
        "iai_mcp.retrieve.recall", side_effect=_recall_side_effect,
    ), mock.patch(
        "iai_mcp.runtime_graph_cache.load_recall_structural",
        return_value=(_make_assignment_with_communities(), None, 0, "test"),
    ):
        from iai_mcp import core as _core

        for sess in ("s1", "s2", "s3"):
            resp = {}
            params = {"session_id": sess, "cue": "x"}
            store = mock.MagicMock()
            store.get = mock.MagicMock(return_value=None)
            with mock.patch(
                "iai_mcp.daemon_state.consume_first_turn", return_value=True
            ), mock.patch(
                "iai_mcp.daemon_state.load_state", return_value={}
            ):
                _core._first_turn_recall_hook(resp, params=params, store=store)
    assert len(observed_results) == 3


def test_first_turn_core_cascade_does_not_rebuild_runtime_graph():
    from iai_mcp import core as _core

    response: dict = {}
    params = {"session_id": "sess-no-rebuild", "cue": "hello"}
    store = mock.MagicMock()
    store.get = mock.MagicMock(return_value=None)

    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4()],
    ), mock.patch(
        "iai_mcp.runtime_graph_cache.load_recall_structural",
        return_value=(_make_assignment_with_communities(), None, 0, "test"),
    ) as load_structural, mock.patch(
        "iai_mcp.retrieve.build_runtime_graph",
        side_effect=AssertionError("runtime graph rebuild must stay off recall hot path"),
    ), mock.patch(
        "iai_mcp.daemon_state.consume_first_turn", return_value=True
    ), mock.patch(
        "iai_mcp.daemon_state.load_state", return_value={}
    ), mock.patch(
        "iai_mcp.retrieve.recall",
        return_value=mock.MagicMock(hits=[], budget_used=0, anti_hits=[]),
    ):
        _core._first_turn_recall_hook(response, params=params, store=store)

    assert "first_turn_recall" in response
    assert load_structural.call_count == 1


def test_response_carries_warm_lru_source():
    with mock.patch(
        "iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[]
    ), mock.patch(
        "iai_mcp.hippea_cascade.compute_core_side_warm_snapshot",
        return_value=[uuid4() for _ in range(2)],
    ):
        response = _invoke_first_turn_hook(session_id="sess-obs")
    assert "first_turn_recall" in response
    assert "warm_lru_source" in response["first_turn_recall"]
    assert response["first_turn_recall"]["warm_lru_source"] in (
        "daemon", "core_fallback", "none",
    )
