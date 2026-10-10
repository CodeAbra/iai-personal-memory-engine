"""Off-by-default per-candidate admission trace on the foresight pack path.

`refresh_pack`'s public report previously exposed only aggregate
`skipped_*` counters, never which specific candidate was admitted at what
raw cosine/rank or which lane packed or dropped it (see the "unattributed"
comment in `test_baseline_knowledge_vs_chatter.py`). This instrument closes
that gap for diagnostic use, entirely behind an opt-in env flag: unset, the
report shape and every other code path stay byte-identical to today.
"""
from __future__ import annotations

import pytest

from iai_mcp import foresight
from iai_mcp.capture import capture_turn
from iai_mcp.embed import embedder_for_store
from iai_mcp.store import MemoryStore, flush_record_buffer
from tests._helpers import stub_embedder_for_store
from tests.test_foresight_served_order_fence import _flagged_rec
from tests.test_value_signal_astra_probe import _MappedEmbedder, _mix, _unit

_KNOWLEDGE_SESSION = "admission-trace-knowledge"
_QUERY_SESSION = "admission-trace-query"

_OUTCOMES = {
    "packed", "not_examined", "skipped_low_cos", "skipped_same_session",
    "skipped_already_injected", "skipped_unconfirmed",
    "skipped_duplicate_content", "skipped_budget_exceeded",
}


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _quiet_reserves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic single-lane admission: no assistant-tail spool reads,
    no derived-cue reserve, no goal blend -- isolates the primary window
    the trace instrument targets."""
    monkeypatch.setenv(foresight.FORESIGHT_ASSISTANT_TAIL_OFF_ENV, "1")
    monkeypatch.setenv(foresight.FORESIGHT_MULTI_CUE_OFF_ENV, "1")
    monkeypatch.setenv(foresight.FORESIGHT_GOAL_WEIGHT_ENV, "0")


def _seed(store: MemoryStore, text: str) -> None:
    capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=_KNOWLEDGE_SESSION, role="user", live_turn=True,
    )
    flush_record_buffer(store)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_admission_trace_absent_by_default(driver, tmp_path, monkeypatch):
    """The flag unset -> the report carries no admission_trace at all, not
    an empty list -- an empty list would still let a caller pay for
    dict-building it never asked for; None is the explicit off signal."""
    _select_driver(driver, monkeypatch)
    _quiet_reserves(monkeypatch)
    monkeypatch.delenv(foresight.FORESIGHT_TRACE_ENV, raising=False)

    store = MemoryStore(path=tmp_path)
    _seed(store, "alice's cat naps on the porch every afternoon.")

    embedder = embedder_for_store(store)
    cue_text = "does alice's cat nap in the afternoon"
    cue_embedding = list(embedder.embed(cue_text))

    report = foresight.refresh_pack(
        store, cue_text=cue_text, cue_embedding=cue_embedding,
        session_id=_QUERY_SESSION,
    )
    assert report["admission_trace"] is None, (
        f"[{driver}] trace must stay absent when the flag is unset"
    )
    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_admission_trace_captures_per_candidate_cosine_rank_lane_when_enabled(
    driver, tmp_path, monkeypatch,
):
    _select_driver(driver, monkeypatch)
    _quiet_reserves(monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, "0.0")
    monkeypatch.setenv(foresight.FORESIGHT_TRACE_ENV, "1")

    store = MemoryStore(path=tmp_path)
    texts = [
        "alice's cat naps on the porch every afternoon.",
        "alice's dog chased a squirrel around the yard this morning.",
        "alice went to the market to buy fresh vegetables yesterday.",
    ]
    for text in texts:
        _seed(store, text)

    embedder = embedder_for_store(store)
    cue_text = "what did alice's cat do this afternoon"
    cue_embedding = list(embedder.embed(cue_text))

    report = foresight.refresh_pack(
        store, cue_text=cue_text, cue_embedding=cue_embedding,
        session_id=_QUERY_SESSION,
    )
    trace = report["admission_trace"]
    assert trace, f"[{driver}] the instrument must capture every admitted candidate"
    assert len(trace) == len(texts), (
        f"[{driver}] every seeded record fits inside the overfetched window "
        "and must get a trace entry"
    )

    for entry in trace:
        assert {"record_id", "raw_cos", "admission_rank", "lane", "outcome"} <= set(entry), (
            f"[{driver}] trace entry missing a required field: {entry}"
        )
        assert isinstance(entry["raw_cos"], float)
        assert isinstance(entry["admission_rank"], int), (
            f"[{driver}] a candidate returned by store.query_similar must carry "
            f"its window rank: {entry}"
        )
        assert entry["lane"] in {"primary_window", "primary"}
        assert entry["outcome"] in _OUTCOMES

    by_rank = sorted(trace, key=lambda e: e["admission_rank"])
    assert [e["admission_rank"] for e in by_rank] == list(range(len(trace))), (
        f"[{driver}] admission_rank must be a dense 0..N-1 index into the raw "
        "ANN window, not an arbitrary counter"
    )
    cosines = [e["raw_cos"] for e in by_rank]
    assert cosines == sorted(cosines, reverse=True), (
        f"[{driver}] admission_rank must reflect store.query_similar's own "
        "cosine-descending order, captured before any boost-sort reorder"
    )

    packed = [e for e in trace if e["outcome"] == "packed"]
    assert packed, f"[{driver}] at least the top candidate must clear a zeroed floor"
    assert all(e["lane"] == "primary" for e in packed), (
        f"[{driver}] a packed candidate's lane must reflect the pass that "
        "actually admitted it into the pack, not just the raw window"
    )
    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_admission_trace_reserve_lane_pack_is_not_overwritten_by_later_primary_skip(
    driver, tmp_path, monkeypatch,
):
    """The reserve lane runs first and packs a salience-flagged record that
    also clears the primary window; the primary lane then re-examines the
    same candidate and skips it as a duplicate. The final trace entry must
    keep the reserve lane's "packed" verdict, not the primary lane's later
    duplicate-content skip -- a record that IS in packed_ids must never be
    traced as skipped."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv(foresight.FORESIGHT_TRACE_ENV, "1")

    query_text = "so hey, anything new worth flagging on your end lately"
    flagged_text = (
        "alice's standing rule: always confirm the deploy window with the "
        "on-call engineer before shipping."
    )
    chatter_texts = [
        "yeah so anyway, nothing much new on my end, just been keeping busy",
        "honestly just the usual stuff, no real updates from me today",
    ]

    query_vec = _unit(0)
    flagged_vec = _mix(0, 1, 0.75)
    chatter_vecs = [_mix(0, 2, 0.62), _mix(0, 3, 0.61)]
    mapping = {query_text: query_vec, flagged_text: flagged_vec}
    for text, vec in zip(chatter_texts, chatter_vecs):
        mapping[text] = vec
    stub_embedder_for_store(monkeypatch, _MappedEmbedder(mapping))

    capture_session = f"trace-reserve-capture-{driver}"

    store = MemoryStore(path=tmp_path / f"trace-reserve-{driver}")
    try:
        flagged = _flagged_rec(
            flagged_text, flagged_vec,
            salience_level="notable", session_id=capture_session,
        )
        store.insert(flagged)
        for text, vec in zip(chatter_texts, chatter_vecs):
            store.insert(_flagged_rec(
                text, vec, salience_level="unflagged", session_id=capture_session,
            ))
        flush_record_buffer(store)

        report = foresight.refresh_pack(
            store, cue_text=query_text, cue_embedding=query_vec,
            session_id=_QUERY_SESSION,
        )
        trace = report["admission_trace"]
        assert trace, f"[{driver}] the trace must capture the flagged record"

        flagged_id = str(flagged.id)
        packed_ids = [pid for pid in report["packed_ids"] if not pid.startswith("q:")]
        assert flagged_id in packed_ids, (
            f"[{driver}] the flagged record must actually be served: {report}"
        )

        entry = next((e for e in trace if e["record_id"] == flagged_id), None)
        assert entry is not None, f"[{driver}] the served record must have a trace entry"
        assert entry["outcome"] == "packed", (
            f"[{driver}] a record in packed_ids must not have its trace "
            f"entry downgraded by a later lane's duplicate-content skip: "
            f"{entry}"
        )
        assert entry["lane"] == "reserve_lane", (
            f"[{driver}] the reserve lane admitted this record first; its "
            f"verdict must win over the primary lane's later re-examination "
            f"of the same candidate: {entry}"
        )
    finally:
        store.close()
