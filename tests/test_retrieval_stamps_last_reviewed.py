"""A retrieved record must be measurably protected from decay.

Composed gates: retrieval -> `last_reviewed` stamp -> erasure sweep exclusion,
plus whether cluster replay can ever see a recently used record. Every
downstream assertion is preceded by a positive gate proving the retrieval
happened and the sweep actually selected the arm it was supposed to.

Gates are collected rather than raised immediately, so a single test run
observes every gate's outcome even when an earlier one fails — that is what
lets the same run prove both "the recall reached the record" and "the sweep
really selected its untouched twin" without one masking the other.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pandas as pd
import pytest

from iai_mcp.core import dispatch
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

# Real wall clock, patched into the pipeline below. A hardcoded past constant
# would make the erasure assertion pass for the wrong reason and would push
# the cluster-replay stamp outside its lookback window.
FROZEN_NOW = datetime.now(timezone.utc)
CREATED_AT = FROZEN_NOW - timedelta(days=60)


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param(
        "lilli",
        id="lilli",
        marks=pytest.mark.skipif(
            not _native_available(),
            reason="iai_mcp_native.engine not installed (build the native "
            "wheel to run engine-driver tests)",
        ),
    ),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _make_record(
    *,
    literal_surface: str,
    embedding: list[float],
    centrality: float = 0.005,
    pinned: bool = False,
    never_decay: bool = False,
    created_at: datetime = CREATED_AT,
    last_reviewed: datetime | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=centrality,
        detail_level=1,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=last_reviewed,
        never_decay=never_decay,
        never_merge=False,
        provenance=[],
        created_at=created_at,
        updated_at=created_at,
        tags=[],
        language="en",
    )


def _read_last_reviewed(store: MemoryStore, record_id: UUID):
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT last_reviewed FROM records WHERE id = ?",
            (str(record_id),),
        ).fetchone()
    return row[0] if row else None


def _row_for(df: pd.DataFrame, rid: UUID) -> dict | None:
    sub = df[df["id"] == str(rid)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _is_tombstoned(row: dict) -> bool:
    val = row.get("tombstoned_at")
    return val is not None and not pd.isna(val)


def _make_pipeline(store: MemoryStore, tmp_path, monkeypatch) -> SleepPipeline:
    monkeypatch.setattr(
        "iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: FROZEN_NOW,
    )
    return SleepPipeline(
        store=store, lifecycle_state_path=tmp_path / "lifecycle_state.json",
    )


def _record_gate(failures: list[str], condition: bool, message: str) -> bool:
    """Collect a gate outcome instead of raising, so later gates still run."""
    if not condition:
        failures.append(message)
    return condition


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_retrieved_record_survives_erasure_sweep_that_tombstones_its_twin(
    tmp_path, monkeypatch, driver,
):
    """A recalled record must outlive an untouched twin the sweep would drop.

    The reconsolidation dry-run guard is left at its pytest default
    (unset -> True under `PYTEST_CURRENT_TEST`) on purpose: this proves the
    stamp under test does not depend on that unrelated write site.

    Parametrized over both storage drivers -- the stamp's text form and the
    sweep's cutoff comparison must agree on the driver production actually
    runs, not only on the default one a fresh store resolves to under test.
    """
    _set_driver(monkeypatch, driver)
    monkeypatch.setenv("IAI_MCP_ERASURE_DRY_RUN", "false")

    store = MemoryStore(path=tmp_path)
    pipe = _make_pipeline(store, tmp_path, monkeypatch)

    failures: list[str] = []

    treatment = _make_record(
        literal_surface="alice's note kept alive by being retrieved",
        embedding=[0.9] + [0.01] * (EMBED_DIM - 1),
    )
    store.insert(treatment)

    resp = dispatch(
        store,
        "memory_recall",
        {
            "cue": "alice's note kept alive by being retrieved",
            "session_id": "s1",
            "cue_embedding": treatment.embedding,
        },
    )

    # Gate 1: the recall actually reached the treatment record. A recall that
    # returns nothing enqueues nothing, and every later assertion would pass
    # for free.
    hit_ids = {h["record_id"] for h in resp["hits"]}
    _record_gate(
        failures,
        str(treatment.id) in hit_ids,
        "the recall must return the treatment record before its stamp can be "
        f"asserted; got hits={hit_ids}",
    )

    # The control is inserted only now, after the recall, so it cannot be
    # picked up by the same recall call.
    control = _make_record(
        literal_surface="bob's note never retrieved, identical age otherwise",
        embedding=[0.01] * (EMBED_DIM - 1) + [0.9],
        created_at=CREATED_AT,
    )
    store.insert(control)

    # Gate 2 (primary red signal): the stamp landed on the retrieved record
    # and only on it.
    treatment_stamp = _read_last_reviewed(store, treatment.id)
    control_stamp = _read_last_reviewed(store, control.id)
    stamp_landed = _record_gate(
        failures,
        treatment_stamp is not None,
        "retrieving a record must stamp last_reviewed on it; got None",
    )
    _record_gate(
        failures,
        control_stamp is None,
        "a record never retrieved must not carry a last_reviewed stamp; "
        f"got {control_stamp!r}",
    )
    if stamp_landed:
        stamp_text = str(treatment_stamp)
        _record_gate(
            failures,
            stamp_text.startswith(FROZEN_NOW.strftime("%Y-%m-%d")),
            "the stamp's on-disk text form must sort correctly against the "
            f"sweep's cutoff strings; got {stamp_text!r}",
        )

    ok, payload = pipe._step_erasure_agent(None)
    assert ok is True, payload
    assert payload.get("dry_run") is False, (
        f"the sweep must run live for this gate to mean anything; got {payload}"
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()

    # Gate 3: the control really was selected by the sweep. A count_rows
    # failure degrades to zero eligible rows, under which "the treatment
    # survives" would pass vacuously.
    control_row = _row_for(df, control.id)
    control_present = _record_gate(
        failures, control_row is not None, "control record disappeared from the table",
    )
    if control_present:
        _record_gate(
            failures,
            _is_tombstoned(control_row),
            "the untouched control record must be selected by the erasure sweep; "
            f"got tombstoned_at={control_row.get('tombstoned_at')!r}",
        )
    _record_gate(
        failures,
        payload.get("count_quarantined", 0) >= 1,
        f"the sweep must have selected at least the control; got {payload}",
    )

    # Gate 4: the retrieved record must NOT be selected.
    treatment_row = _row_for(df, treatment.id)
    treatment_present = _record_gate(
        failures, treatment_row is not None, "treatment record disappeared from the table",
    )
    if treatment_present:
        _record_gate(
            failures,
            not _is_tombstoned(treatment_row),
            "a recently retrieved record must be excluded from the erasure sweep; "
            f"got tombstoned_at={treatment_row.get('tombstoned_at')!r}",
        )

    if failures:
        pytest.fail("\n".join(failures))


def test_two_recently_retrieved_records_form_a_replay_cluster(tmp_path, monkeypatch):
    """Cluster replay must see records retrieved moments ago.

    Both records are inserted before the single recall that touches them, the
    opposite ordering from the erasure gate above: here both stamps are meant
    to land inside the same lookback window.

    Only `clusters_replayed` is asserted. `avg_cluster_size` lives in the
    emitted event body, not in this payload, and asserting it would error the
    test rather than fail it for the right reason. `edges_boosted` is not
    trustworthy either: under the pytest-default dry-run it reports the count
    of edges it would have written, not what it wrote, so a positive value
    there proves nothing about reachability.
    """
    store = MemoryStore(path=tmp_path)
    pipe = _make_pipeline(store, tmp_path, monkeypatch)

    failures: list[str] = []

    first = _make_record(
        literal_surface="alice's cluster-replay record one",
        embedding=[0.9] + [0.01] * (EMBED_DIM - 1),
    )
    second = _make_record(
        literal_surface="alice's cluster-replay record two",
        embedding=[0.01] * (EMBED_DIM - 1) + [0.9],
    )
    store.insert(first)
    store.insert(second)

    resp = dispatch(
        store,
        "memory_recall",
        {
            "cue": "alice's cluster-replay record one",
            "session_id": "s1",
            "cue_embedding": first.embedding,
        },
    )

    # Gate: the recall reached both records. Cluster replay only forms
    # clusters of size >= 2, so a single retrieved record would pass every
    # downstream assertion vacuously.
    hit_ids = {h["record_id"] for h in resp["hits"]}
    _record_gate(
        failures,
        len(resp["hits"]) >= 2,
        f"the recall must return both records for a cluster to form; got {resp['hits']}",
    )
    _record_gate(
        failures,
        {str(first.id), str(second.id)} <= hit_ids,
        f"both records must be among the recall hits; got {hit_ids}",
    )

    first_stamp = _read_last_reviewed(store, first.id)
    second_stamp = _read_last_reviewed(store, second.id)
    _record_gate(
        failures,
        first_stamp is not None and second_stamp is not None,
        "both retrieved records must carry a last_reviewed stamp before "
        f"replay can see them; got {first_stamp!r}, {second_stamp!r}",
    )

    ok, payload = pipe._step_cluster_replay(None)
    assert ok is True, payload
    _record_gate(
        failures,
        payload.get("clusters_replayed", 0) >= 1,
        "cluster replay must count a cluster over two records retrieved "
        f"inside its lookback window; got {payload}",
    )

    if failures:
        pytest.fail("\n".join(failures))


def test_cluster_replay_forms_no_cluster_over_records_never_retrieved(
    tmp_path, monkeypatch,
):
    """The negative control: replay must stay empty when nothing was recalled.

    This distinguishes "replay sees retrieved records" from "replay sees
    everything".
    """
    store = MemoryStore(path=tmp_path)
    pipe = _make_pipeline(store, tmp_path, monkeypatch)

    first = _make_record(
        literal_surface="bob's untouched record one",
        embedding=[0.9] + [0.01] * (EMBED_DIM - 1),
    )
    second = _make_record(
        literal_surface="bob's untouched record two",
        embedding=[0.01] * (EMBED_DIM - 1) + [0.9],
    )
    store.insert(first)
    store.insert(second)

    ok, payload = pipe._step_cluster_replay(None)
    assert ok is True, payload
    assert payload.get("clusters_replayed", 0) == 0, (
        f"records that were never retrieved must not form a replay cluster; got {payload}"
    )


def test_retrieval_reinforce_issues_exactly_one_records_update(tmp_path, monkeypatch):
    """One retrieval reinforce must carry both stamps in a single statement.

    `boost_edges` opens the edges table before any records write, and the
    conftest autouse fixture flushes buffers through both `insert` and
    `boost_edges`; the spy is installed after the insert and filtered to the
    records table so neither of those touches inflate the count.
    """
    from iai_mcp.hippo._table import HippoTable

    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")

    store = MemoryStore(path=tmp_path)
    record = _make_record(
        literal_surface="alice's record reinforced exactly once",
        embedding=[0.9] + [0.01] * (EMBED_DIM - 1),
    )
    store.insert(record)

    calls: list[tuple[str | None, str, dict]] = []
    orig_update = HippoTable.update

    def _spy_update(self, where, values):
        calls.append((getattr(self, "_name", None), where, dict(values)))
        return orig_update(self, where, values)

    monkeypatch.setattr(HippoTable, "update", _spy_update)

    store.reinforce_record(record.id, is_retrieval=True)

    records_calls = [c for c in calls if c[0] == "records"]
    assert len(records_calls) == 1, (
        "a retrieval reinforce must issue exactly one records-table update; "
        f"got {len(records_calls)}: {records_calls}"
    )
    values = records_calls[0][2]
    assert "last_reviewed" in values, (
        f"the single records update must carry last_reviewed; got {values!r}"
    )
    assert "labile_until" in values, (
        "when reconsolidation is live, the same update must also carry "
        f"labile_until rather than issuing a second statement; got {values!r}"
    )


def test_reinforce_without_retrieval_does_not_stamp(tmp_path, monkeypatch):
    """Reinforcement not driven by a retrieval must not stamp last_reviewed.

    The nightly reconsolidation step reinforces records with
    `is_retrieval=False`; if that stamped the column, a sleep cycle would
    protect its own records from the erasure step running in the same cycle.
    """
    store = MemoryStore(path=tmp_path)
    record = _make_record(
        literal_surface="bob's record reinforced without being retrieved",
        embedding=[0.9] + [0.01] * (EMBED_DIM - 1),
    )
    store.insert(record)

    store.reinforce_record(record.id)

    assert _read_last_reviewed(store, record.id) is None, (
        "reinforcement with is_retrieval=False must not stamp last_reviewed"
    )


def test_malformed_reconsolidation_config_does_not_cancel_the_stamp(
    tmp_path, monkeypatch,
):
    """A broken reconsolidation config must not take the stamp down with it.

    The config load lives in its own nested error handler; removing that
    handler lets the ValueError escape past the records update and the
    stamp is lost.
    """
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "not-a-number")

    store = MemoryStore(path=tmp_path)
    record = _make_record(
        literal_surface="alice's record reinforced under a broken config",
        embedding=[0.9] + [0.01] * (EMBED_DIM - 1),
    )
    store.insert(record)

    store.reinforce_record(record.id, is_retrieval=True)

    assert _read_last_reviewed(store, record.id) is not None, (
        "a malformed reconsolidation config must not cancel the last_reviewed stamp"
    )


def test_missing_column_error_degrades_without_raising(tmp_path, monkeypatch):
    """A records-UPDATE that fails on a missing column must not escape.

    Drives the driver's own missing-column error (`OperationalError`, the
    PEP-249 class `iai_mcp.errors.DatabaseError` subclasses) through the
    records-table update: the retrieval reinforce must swallow it and simply
    leave the stamp unwritten, not propagate it out of `reinforce_record`.
    """
    from iai_mcp.errors import OperationalError
    from iai_mcp.hippo._table import HippoTable

    store = MemoryStore(path=tmp_path)
    record = _make_record(
        literal_surface="alice's record whose table lacks last_reviewed",
        embedding=[0.9] + [0.01] * (EMBED_DIM - 1),
    )
    store.insert(record)

    orig_update = HippoTable.update

    def _raise_missing_column(self, where, values):
        if getattr(self, "_name", None) == "records":
            raise OperationalError("no such column: last_reviewed")
        return orig_update(self, where, values)

    monkeypatch.setattr(HippoTable, "update", _raise_missing_column)

    store.reinforce_record(record.id, is_retrieval=True)  # must not raise

    assert _read_last_reviewed(store, record.id) is None, (
        "a missing-column error must degrade without ever writing the stamp"
    )


def test_unrelated_records_update_error_still_propagates(tmp_path, monkeypatch):
    """A records-UPDATE error unrelated to a missing column must still raise.

    Proves the exception-type widening that lets the missing-column case
    degrade does not turn into a blanket swallow: a real driver error of the
    same caught type, whose message does not match the column-missing
    matcher, must still escape `reinforce_record`.
    """
    from iai_mcp.errors import OperationalError
    from iai_mcp.hippo._table import HippoTable

    store = MemoryStore(path=tmp_path)
    record = _make_record(
        literal_surface="bob's record whose table update fails for an unrelated reason",
        embedding=[0.9] + [0.01] * (EMBED_DIM - 1),
    )
    store.insert(record)

    orig_update = HippoTable.update

    def _raise_unrelated(self, where, values):
        if getattr(self, "_name", None) == "records":
            raise OperationalError("disk I/O error")
        return orig_update(self, where, values)

    monkeypatch.setattr(HippoTable, "update", _raise_unrelated)

    with pytest.raises(OperationalError):
        store.reinforce_record(record.id, is_retrieval=True)
