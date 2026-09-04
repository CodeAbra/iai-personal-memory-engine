"""Tracer proving stored `valence` reaches the rank-time multiplier via the
DOMINANT graph-pool candidate path (`records_cache` built from
`graph.get_payload(...)`), not only the store-fallback decode. Also covers
the `raise_valence` writer: it persists, clamps, and honors its kill-switch
so an unwritten store recalls byte-identically to today.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_recall_stage_profile import _monkeypatch_env  # noqa: E402

from tests._synthetic_cue_corpus import (  # noqa: E402
    build_corpus_records,
    build_cue_set,
    insert_corpus,
)
from tests.test_exact_authority_index import _assert_top_k_tie_tolerant  # noqa: E402

from iai_mcp.embed import Embedder  # noqa: E402
from iai_mcp.lifecycle_state import default_state, save_state  # noqa: E402
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline  # noqa: E402
from iai_mcp.lilli.ops.reconsolidation import STABILITY_BOOST_ON_RECALL  # noqa: E402
from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.types import MemoryRecord  # noqa: E402

# Byte-identity contract: the live critic step must stay frozen. A
# committed accidental edit would pass a base-less `git diff`; a
# content-hash fence does not.
_RECONSOLIDATION_STEP_SHA256 = (
    "64eedfe0a0caa315eaf42d7c2f447690c9dcb8beaecbd8d17ca2fccd9db010ce"
)
_RECONSOLIDATION_CRITIC_SHA256 = (
    "e71f7a5e79adb45b3bedb33db0d84ba1e44e5164887a8be59546bb0ab50d5047"
)

_SEED = 0
_TOP_K = 10

# One home_renovation sentence -- not in the corpus's verbatim-topic set, so
# a plain "concept" cue reaches it without a verbatim-quote wrapper.
_TARGET_SURFACE = (
    "Alice's kitchen renovation contractor recommended quartz countertops "
    "over granite."
)
_TARGET_CUE_TEXT = (
    "Did Alice's kitchen renovation contractor recommend quartz countertops "
    "over granite?"
)


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_graph_pool_payload_carries_stored_valence(tmp_path, monkeypatch, driver):
    """A positive valence written directly to the records column is decoded
    by the cold graph-build stream, carried into the persistent graph's node
    payload, and resolves via `graph.get_payload(rid)` for a graph-pool
    candidate -- proving the DOMINANT read path (not only the store
    fallback) actually carries the value pipeline.py's rank multiplier
    reads.
    """
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / f"valence-graph-pool-{driver}"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    records = build_corpus_records(seed=_SEED, embedder=embedder)
    store = MemoryStore(path=store_root)
    insert_corpus(store, records)

    target = records[0]
    tbl = store.db.open_table("records")
    tbl.update(where=f"id = '{target.id}'", values={"valence": 0.5})

    graph, _assignment, _rich_club = build_runtime_graph(store)

    payload = graph.get_payload(target.id)
    assert payload is not None, f"graph carries no payload for {target.id}"
    assert payload.get("valence") == pytest.approx(0.5), (
        f"graph-pool payload valence for {target.id} did not resolve the "
        f"stored write: {payload.get('valence')!r}"
    )


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _build_valence_store(driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / f"valence-writer-{driver}"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    records = build_corpus_records(seed=_SEED, embedder=embedder)
    store = MemoryStore(path=store_root)
    insert_corpus(store, records)
    graph, assignment, rich_club = build_runtime_graph(store)
    return store, graph, assignment, rich_club, embedder, records


def _find_target_record(records: "list[object]"):
    for rec in records:
        if rec.literal_surface == _TARGET_SURFACE:
            return rec
    raise AssertionError(
        "fixture precondition: target home_renovation record not found in corpus"
    )


def _find_target_cue():
    for cue in build_cue_set(seed=_SEED)["specific"]:
        if cue.text == _TARGET_CUE_TEXT:
            return cue
    raise AssertionError("fixture precondition: target specific cue not found")


def _hit_for(hits, record_id):
    for h in hits:
        if h.record_id == record_id:
            return h
    return None


def _rebuild_graph_after_write(store: MemoryStore):
    """Mirrors the production write-then-recall sequence: a warm graph
    bundle is never live-patched by a valence write, so the cache must be
    explicitly invalidated and rebuilt for the new column value to reach a
    fresh graph payload."""
    from iai_mcp import runtime_graph_cache

    runtime_graph_cache.invalidate(store)
    assert getattr(store, "_warm_graph_bundle", None) is None, (
        "runtime_graph_cache.invalidate did not clear the warm bundle -- "
        "the next build would silently re-serve the stale (pre-write) graph"
    )
    return build_runtime_graph(store)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_raise_valence_persists_moves_score_and_fires_xval(tmp_path, monkeypatch, driver):
    store, graph, assignment, rich_club, embedder, records = _build_valence_store(
        driver, tmp_path, monkeypatch,
    )
    target = _find_target_record(records)
    cue = _find_target_cue()

    baseline = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue.text, session_id="valence-writer-baseline",
        budget_tokens=1500, mode=cue.mode,
    )
    before = _hit_for(baseline.hits, target.id)
    assert before is not None, (
        "target record absent from the unwritten baseline recall -- fixture "
        "precondition failed"
    )
    assert "xval" not in before.reason, (
        f"unwritten record already carries an xval reason marker: {before.reason!r}"
    )

    wrote = store.raise_valence(target.id, 0.5)
    assert wrote is True, "raise_valence must report a successful write"

    graph2, assignment2, rich_club2 = _rebuild_graph_after_write(store)
    after_response = recall_for_response(
        store=store, graph=graph2, assignment=assignment2, rich_club=rich_club2,
        embedder=embedder, cue=cue.text, session_id="valence-writer-after",
        budget_tokens=1500, mode=cue.mode,
    )
    after = _hit_for(after_response.hits, target.id)
    assert after is not None, "target record absent from post-write recall"
    assert "xval" in after.reason, f"xval marker missing from reason: {after.reason!r}"
    assert after.score > before.score, (
        f"score did not rise after writing a positive valence: "
        f"before={before.score} after={after.score}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_raise_valence_live_patches_warm_graph_without_rebuild(
    tmp_path, monkeypatch, driver,
):
    """Production propagation path: raise_valence must reach the graph
    object already built by build_runtime_graph in place, via the
    registered graph-sync hook, without an explicit
    runtime_graph_cache.invalidate() + rebuild in between."""
    from iai_mcp import runtime_graph_cache

    store, graph, assignment, rich_club, embedder, records = _build_valence_store(
        driver, tmp_path, monkeypatch,
    )
    target = _find_target_record(records)
    cue = _find_target_cue()

    before_payload = graph.get_payload(target.id)
    assert before_payload is not None
    assert before_payload.get("valence") == pytest.approx(0.0)

    dirty_before = runtime_graph_cache.get_dirty_counter()
    wrote = store.raise_valence(target.id, 0.5)
    assert wrote is True, "raise_valence must report a successful write"

    after_payload = graph.get_payload(target.id)
    assert after_payload is not None
    assert after_payload.get("valence") == pytest.approx(0.5), (
        "raise_valence did not live-patch the already-built graph object "
        "in place -- the graph-sync hook did not fire"
    )
    assert runtime_graph_cache.get_dirty_counter() > dirty_before, (
        "raise_valence did not bump the dirty counter that gates on-disk "
        "cache regeneration"
    )

    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue.text, session_id="valence-live-patch",
        budget_tokens=1500, mode=cue.mode,
    )
    hit = _hit_for(response.hits, target.id)
    assert hit is not None, "target record absent from post-write recall"
    assert "xval" in hit.reason, f"xval marker missing from reason: {hit.reason!r}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_raise_valence_write_off_kill_switch_leaves_recall_byte_identical(
    tmp_path, monkeypatch, driver,
):
    monkeypatch.setenv("IAI_MCP_VALENCE_WRITE_OFF", "1")
    store, graph, assignment, rich_club, embedder, records = _build_valence_store(
        driver, tmp_path, monkeypatch,
    )
    target = _find_target_record(records)
    cue = _find_target_cue()

    baseline = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue.text, session_id="valence-write-off-baseline",
        budget_tokens=1500, mode=cue.mode,
    )
    baseline_top_k = [(str(h.record_id), h.score) for h in baseline.hits[:_TOP_K]]

    wrote = store.raise_valence(target.id, 0.5)
    assert wrote is False, "IAI_MCP_VALENCE_WRITE_OFF must make raise_valence a no-op"

    persisted = store.get(target.id)
    assert persisted is not None and persisted.valence == 0.0, (
        f"write-off leg must persist nothing: got valence={persisted.valence!r}"
    )

    graph2, assignment2, rich_club2 = _rebuild_graph_after_write(store)
    after_response = recall_for_response(
        store=store, graph=graph2, assignment=assignment2, rich_club=rich_club2,
        embedder=embedder, cue=cue.text, session_id="valence-write-off-after",
        budget_tokens=1500, mode=cue.mode,
    )
    after_top_k = [(str(h.record_id), h.score) for h in after_response.hits[:_TOP_K]]

    _assert_top_k_tie_tolerant(baseline_top_k, after_top_k, k=_TOP_K, cue_seed=0)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_raise_valence_refuses_to_lower_existing_value(tmp_path, monkeypatch, driver):
    """Monotone-raise guard: a call with a new_value at or below the current
    stored valence must be a no-op, not a silent lower."""
    store, _graph, _assignment, _rich_club, _embedder, records = _build_valence_store(
        driver, tmp_path, monkeypatch,
    )
    target = _find_target_record(records)

    raised = store.raise_valence(target.id, 0.5)
    assert raised is True

    lowered = store.raise_valence(target.id, 0.2)
    assert lowered is False, "raise_valence must refuse a lower new_value"

    same = store.raise_valence(target.id, 0.5)
    assert same is False, "raise_valence must refuse an equal new_value"

    persisted = store.get(target.id)
    assert persisted is not None
    assert persisted.valence == pytest.approx(0.5), (
        f"a refused lower/equal write must leave the stored value untouched: "
        f"got {persisted.valence!r}"
    )


def test_live_reconsolidation_step_files_byte_identical() -> None:
    import iai_mcp.lilli.cycle.sleep_pipeline._reconsolidation as _live_step
    import iai_mcp.reconsolidation_critic as _live_critic

    step_hash = hashlib.sha256(Path(_live_step.__file__).read_bytes()).hexdigest()
    critic_hash = hashlib.sha256(Path(_live_critic.__file__).read_bytes()).hexdigest()

    assert step_hash == _RECONSOLIDATION_STEP_SHA256, (
        f"_reconsolidation.py changed since being pinned: sha256={step_hash}"
    )
    assert critic_hash == _RECONSOLIDATION_CRITIC_SHA256, (
        f"reconsolidation_critic.py changed since being pinned: sha256={critic_hash}"
    )


def _make_valence_test_record(embed_dim: int, literal: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal,
        aaak_index="",
        embedding=[0.01] * embed_dim,
        community_id=None,
        centrality=0.0,
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
    )


def _make_valence_facade_store(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> MemoryStore:
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / f"valence-facade-{driver}"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    return MemoryStore(
        path=store_root, read_consistency_interval=timedelta(seconds=0),
    )


def _mark_labile(store: MemoryStore, record_id: uuid.UUID) -> None:
    tbl = store.db.open_table("records")
    tbl.update(
        where=f"id = '{record_id}'",
        values={"labile_until": datetime.now(timezone.utc) + timedelta(seconds=300)},
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_sleep_step_live_patches_warm_graph_without_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str,
) -> None:
    """Production caller path: the sleep step (the only real caller of
    raise_valence) must live-patch an already-built graph object in place,
    not just a direct store.raise_valence call driven by a test."""
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / f"valence-sleep-step-live-{driver}"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    embedder = Embedder()
    records = build_corpus_records(seed=_SEED, embedder=embedder)
    store = MemoryStore(path=store_root, read_consistency_interval=timedelta(seconds=0))
    insert_corpus(store, records)
    graph, _assignment, _rich_club = build_runtime_graph(store)

    target = _find_target_record(records)
    _mark_labile(store, target.id)

    before_payload = graph.get_payload(target.id)
    assert before_payload is not None
    assert before_payload.get("valence") == pytest.approx(0.0)

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)
    done, payload = pipeline._step_reconsolidation_valence(interrupt_check=None)
    assert done is True
    assert payload["valence_writes"] >= 1

    after_payload = graph.get_payload(target.id)
    assert after_payload is not None
    assert after_payload.get("valence") == pytest.approx(STABILITY_BOOST_ON_RECALL), (
        "the sleep step's raise_valence write did not live-patch the "
        "already-built graph object -- the production caller path did not "
        "propagate to the warm bundle"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_step_writes_only_the_valence_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str,
) -> None:
    """Single-column-write fence: every other field, including
    literal_surface, is byte-identical before and after the step runs."""
    store = _make_valence_facade_store(driver, tmp_path, monkeypatch)
    embed_dim = store._embed_dim
    rec = _make_valence_test_record(embed_dim, "alice single column record")
    store.insert(rec)
    _mark_labile(store, rec.id)

    before = store.get(rec.id)
    assert before is not None

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)
    done, payload = pipeline._step_reconsolidation_valence(interrupt_check=None)
    assert done is True
    assert payload["valence_writes"] >= 1

    after = store.get(rec.id)
    assert after is not None
    assert after.valence != before.valence, (
        "non-vacuity: the write must actually move valence"
    )
    after_with_before_valence = dataclasses.replace(after, valence=before.valence)
    assert after_with_before_valence == before, (
        "the step wrote a field other than valence"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_facade_persists_bounded_delta_to_labile_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str,
) -> None:
    store = _make_valence_facade_store(driver, tmp_path, monkeypatch)
    embed_dim = store._embed_dim
    labile_rec = _make_valence_test_record(embed_dim, "alice labile record")
    stable_rec = _make_valence_test_record(embed_dim, "alice untouched record")
    store.insert(labile_rec)
    store.insert(stable_rec)
    _mark_labile(store, labile_rec.id)

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)

    done, payload = pipeline._step_reconsolidation_valence(interrupt_check=None)
    assert done is True
    assert payload["candidates_labile"] >= 1
    assert payload["valence_writes"] >= 1

    after_labile = store.get(labile_rec.id)
    assert after_labile is not None
    assert after_labile.valence == pytest.approx(STABILITY_BOOST_ON_RECALL)
    assert 0.0 < after_labile.valence <= 1.0

    after_stable = store.get(stable_rec.id)
    assert after_stable is not None
    assert after_stable.valence == 0.0, (
        "a non-labile record must not receive a valence write"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_facade_kill_switch_persists_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str,
) -> None:
    store = _make_valence_facade_store(driver, tmp_path, monkeypatch)
    embed_dim = store._embed_dim
    rec = _make_valence_test_record(embed_dim, "alice kill switch record")
    store.insert(rec)
    _mark_labile(store, rec.id)

    monkeypatch.setenv("IAI_MCP_VALENCE_WRITE_OFF", "1")

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)
    done, payload = pipeline._step_reconsolidation_valence(interrupt_check=None)

    assert done is True
    assert payload == {
        "candidates_labile": 0, "valence_writes": 0, "valence_saturated": 0,
    }

    after = store.get(rec.id)
    assert after is not None
    assert after.valence == 0.0
