"""Write-path proof for epistemic_status through capture_turn, contradict,
and their RPC dispatch entry points.

Guards default-safety (omission stays "unknown", byte-unchanged for existing
callers), invalid-value coercion to "unknown", and first-set-wins across a
near-dup fold.
"""

from __future__ import annotations

import dataclasses
from uuid import UUID

import pytest

from iai_mcp import retrieve
from iai_mcp.capture import DEDUP_COS_THRESHOLD, capture_turn
from iai_mcp.core import dispatch
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_stores_explicit_epistemic_status(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    result = capture_turn(
        store, cue="c", text="alice thinks the fix will land Friday",
        epistemic_status="estimate", session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.epistemic_status == "estimate"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_omitted_defaults_to_unknown(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    result = capture_turn(
        store, cue="c", text="alice attended the weekly standup",
        session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.epistemic_status == "unknown"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_dispatch_memory_capture_threads_epistemic_status(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    result = dispatch(
        store, "memory_capture",
        {
            "text": "alice guesses the release slips a week",
            "session_id": "s1",
            "epistemic_status": "hypothesis",
        },
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.epistemic_status == "hypothesis"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_contradict_threads_epistemic_status_direct_and_rpc(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    seed = capture_turn(
        store, cue="c", text="alice's release ships next Tuesday",
        session_id="s1", role="user",
    )
    assert seed["status"] == "inserted", seed
    original_id = UUID(seed["record_id"])

    new_fact = "alice's release actually shipped last Tuesday"
    emb = embedder_for_store(store).embed(new_fact)
    receipt = retrieve.contradict(
        store, original_id, new_fact, list(emb), epistemic_status="fact",
    )
    corrected = store.get(receipt.new_record_id)
    assert corrected is not None
    assert corrected.epistemic_status == "fact"

    seed2 = capture_turn(
        store, cue="c", text="alice's second release ships next Wednesday",
        session_id="s1", role="user",
    )
    assert seed2["status"] == "inserted", seed2
    original_id2 = UUID(seed2["record_id"])

    new_fact2 = "alice's second release actually shipped last Wednesday"
    emb2 = embedder_for_store(store).embed(new_fact2)
    rpc_result = dispatch(
        store, "memory_contradict",
        {
            "id": str(original_id2),
            "new_fact": new_fact2,
            "cue_embedding": list(emb2),
            "epistemic_status": "fact",
        },
    )
    corrected2 = store.get(UUID(rpc_result["new_record_id"]))
    assert corrected2 is not None
    assert corrected2.epistemic_status == "fact"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_invalid_epistemic_status_coerced_to_unknown_direct_and_rpc(driver, tmp_path, monkeypatch):
    """An out-of-enum caller value must never abort a capture -- a
    non-enum-enforcing MCP host forwarding garbage can otherwise cost a
    whole Stop-hook batch turn. capture_turn and contradict coerce to
    'unknown' before MemoryRecord construction, symmetric with the
    read-path coercion in _from_row; the dataclass __post_init__ raise
    stays as a last-resort internal invariant, never reached from here."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    result = capture_turn(
        store, cue="c", text="alice's invalid-status capture attempt",
        epistemic_status="banana", session_id="s1", role="user",
    )
    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.epistemic_status == "unknown"

    rpc_result = dispatch(
        store, "memory_capture",
        {
            "text": "alice's invalid-status rpc capture attempt",
            "session_id": "s1",
            "epistemic_status": "banana",
        },
    )
    assert rpc_result["status"] == "inserted", rpc_result
    rpc_rec = store.get(UUID(rpc_result["record_id"]))
    assert rpc_rec is not None
    assert rpc_rec.epistemic_status == "unknown"

    seed = capture_turn(
        store, cue="c", text="alice's release ships next Tuesday",
        session_id="s1", role="user",
    )
    assert seed["status"] == "inserted", seed
    original_id = UUID(seed["record_id"])
    new_fact = "alice's release actually shipped last Tuesday"
    emb = embedder_for_store(store).embed(new_fact)
    receipt = retrieve.contradict(
        store, original_id, new_fact, list(emb), epistemic_status="banana",
    )
    corrected = store.get(receipt.new_record_id)
    assert corrected is not None
    assert corrected.epistemic_status == "unknown"

    seed2 = capture_turn(
        store, cue="c", text="alice's second release ships next Wednesday",
        session_id="s1", role="user",
    )
    assert seed2["status"] == "inserted", seed2
    original_id2 = UUID(seed2["record_id"])
    new_fact2 = "alice's second release actually shipped last Wednesday"
    emb2 = embedder_for_store(store).embed(new_fact2)
    rpc_contradict = dispatch(
        store, "memory_contradict",
        {
            "id": str(original_id2),
            "new_fact": new_fact2,
            "cue_embedding": list(emb2),
            "epistemic_status": "banana",
        },
    )
    corrected2 = store.get(UUID(rpc_contradict["new_record_id"]))
    assert corrected2 is not None
    assert corrected2.epistemic_status == "unknown"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_memoryrecord_post_init_rejects_invalid_epistemic_status(driver, tmp_path, monkeypatch):
    """The dataclass __post_init__ raise stays reachable for a direct
    MemoryRecord construction, even though capture_turn/contradict now
    coerce before it. This is the sole remaining coverage for that
    last-resort internal invariant."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    result = capture_turn(
        store, cue="c", text="alice's post-init guard baseline capture",
        session_id="s1", role="user",
    )
    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None

    with pytest.raises(ValueError):
        dataclasses.replace(rec, epistemic_status="banana")


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_near_dup_fold_never_overwrites_survivor_epistemic_status(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    result_a = capture_turn(
        store, cue="c", text="alice's fix lands next week",
        epistemic_status="fact", near_dup_gate=True, session_id="s1", role="user",
    )
    assert result_a["status"] == "inserted", result_a
    id_a = result_a["record_id"]
    record_a = store.get(UUID(id_a))
    assert record_a is not None

    count_after_a = sum(1 for _ in store.iter_records())

    # Deterministic neighbour score at/above DEDUP_COS_THRESHOLD, independent
    # of the real embedder's actual cosine output on the B paraphrase.
    def _fake_query_similar(embedding, k=3, tier=None):
        return [(record_a, DEDUP_COS_THRESHOLD + 0.01)]

    monkeypatch.setattr(store, "query_similar", _fake_query_similar)

    result_b = capture_turn(
        store, cue="c", text="alice's fix will land next week",
        epistemic_status="hypothesis", near_dup_gate=True, session_id="s1", role="user",
    )

    # Hard precondition: the fold must be PROVEN to have materialized before
    # the status-preservation assertion means anything.
    assert result_b["status"] == "reinforced", result_b
    assert result_b["record_id"] == id_a, result_b
    assert result_b["reason"].startswith("cos="), result_b

    survivor = store.get(UUID(id_a))
    assert survivor is not None
    assert survivor.epistemic_status == "fact"

    count_after_b = sum(1 for _ in store.iter_records())
    assert count_after_b == count_after_a


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_ambient_capture_tagged_by_lexical_epistemic_classifier(driver, tmp_path, monkeypatch):
    """An ambient (omitted-status) episodic user/assistant capture is
    auto-tagged by the lexical classifier; an explicit caller value is
    still never second-guessed."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)

    hedge_result = capture_turn(
        store, cue="c", text="alice thinks the release lands roughly next week",
        session_id="s1", role="user",
    )
    assert hedge_result["status"] == "inserted", hedge_result
    hedge_rec = store.get(UUID(hedge_result["record_id"]))
    assert hedge_rec is not None
    assert hedge_rec.epistemic_status in {"estimate", "hypothesis"}

    fact_result = capture_turn(
        store, cue="c", text="the team confirmed alice's release shipped Tuesday",
        session_id="s1", role="user",
    )
    assert fact_result["status"] == "inserted", fact_result
    fact_rec = store.get(UUID(fact_result["record_id"]))
    assert fact_rec is not None
    assert fact_rec.epistemic_status == "fact"

    explicit_result = capture_turn(
        store, cue="c", text="the team confirmed alice's release shipped Wednesday",
        epistemic_status="opinion", session_id="s1", role="user",
    )
    assert explicit_result["status"] == "inserted", explicit_result
    explicit_rec = store.get(UUID(explicit_result["record_id"]))
    assert explicit_rec is not None
    assert explicit_rec.epistemic_status == "opinion"
