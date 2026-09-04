"""Process-exit and daemon-absent durability guards for the reinforce queue.

All four recall-answer-path write types are best-effort-drop (see
257-DURABILITY.md). These tests prove that classification is honored, not
merely declared: a hard process kill must never lose a write with zero
signal anywhere queryable, and the queue must never depend on a separate
daemon process to ever run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _record(text: str = "n") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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


def _edge_weight(store, a: UUID, b: UUID, edge_type: str) -> "float | None":
    key = sorted([str(a), str(b)])
    df = store.db.open_table(EDGES_TABLE).to_pandas()
    if df.empty:
        return None
    mask = (
        (df["src"] == key[0])
        & (df["dst"] == key[1])
        & (df["edge_type"] == edge_type)
    )
    if not mask.any():
        return None
    return float(df.loc[mask, "weight"].iloc[0])


# Subprocess body: insert one record, enqueue a reinforcement write for it,
# then either explicitly flush-and-close (landed) or hard-kill via os._exit
# with NO flush signal (not landed).  coalesce_ms is set far larger than the
# process lifetime in the not-landed case so the worker never gets a chance
# to run the actual DB write before the process dies -- deterministic, not
# a timing race.
_SUBPROCESS_TEMPLATE = textwrap.dedent("""
    import os
    import sys
    from datetime import datetime, timezone
    from uuid import UUID

    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    if {suppress_marker}:
        class _NullStream:
            def write(self, *a, **k):
                pass
            def flush(self):
                pass
        sys.stderr = _NullStream()

    now = datetime.now(timezone.utc)
    store = MemoryStore(path={store_path!r})
    rec = MemoryRecord(
        id=UUID({rec_id!r}), tier="episodic", literal_surface="n",
        aaak_index="", embedding=[0.1] * EMBED_DIM, community_id=None,
        centrality=0.0, detail_level=2, pinned=False, stability=0.0,
        difficulty=0.0, last_reviewed=None, never_decay=False,
        never_merge=False, provenance=[], created_at=now, updated_at=now,
        tags=[], language="en",
    )
    store.insert(rec)
    # The record buffer is a separate, pre-existing durability mechanism for
    # verbatim content, out of this suite's scope -- flush it explicitly so
    # only the reinforce queue's write is exposed to the hard kill below.
    flush_record_buffer(store)

    store.enable_reinforce_queue(coalesce_ms={coalesce_ms})
    store._reinforce_queue.enqueue([rec.id])

    if {flush_before_exit}:
        store._reinforce_queue.flush(timeout=5.0)
        store.close()

    os._exit(0)
""")


def _run_subprocess(
    store_path, rec_id, *, flush_before_exit, coalesce_ms, suppress_marker=False,
):
    script = _SUBPROCESS_TEMPLATE.format(
        store_path=str(store_path),
        rec_id=str(rec_id),
        flush_before_exit=flush_before_exit,
        coalesce_ms=coalesce_ms,
        suppress_marker=suppress_marker,
    )
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def _json_markers(text: str, event: str) -> "list[dict]":
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == event:
            out.append(obj)
    return out


def test_process_exit_landed_write_present_and_drained_marker_seen(tmp_path):
    """Fast in-window flush before a process exit: the reinforcement write
    IS present after reopening the store, and the drain-completion marker
    confirms it -- the positive half of the non-vacuous pair."""
    store_path = tmp_path / "landed"
    rec_id = uuid4()

    proc = _run_subprocess(
        store_path, rec_id, flush_before_exit=True, coalesce_ms=50,
    )
    assert proc.returncode == 0, proc.stderr

    deferred = _json_markers(proc.stderr, "reinforce_deferred")
    drained = _json_markers(proc.stderr, "reinforce_drained")
    assert deferred and deferred[0]["n_ids"] == 1, proc.stderr
    assert drained and drained[0]["n_drained"] == 1 and drained[0]["n_total"] == 1, (
        proc.stderr
    )

    store = MemoryStore(path=store_path)
    got = store.get(rec_id)
    assert got is not None
    assert got.literal_surface == "n", "verbatim content unaffected either way"

    w = _edge_weight(store, rec_id, rec_id, "hebbian")
    assert w is not None, "landed write must be present after reopening the store"
    assert w == pytest.approx(0.1, abs=1e-3)


def test_process_exit_dropped_write_absent_but_accounted_for(tmp_path):
    """Hard kill (os._exit, no flush) before the coalesce window elapses:
    the reinforcement write is genuinely NOT present, but it is accounted
    for by the enqueue-time marker -- never a silent, signal-free vanish.
    coalesce_ms=60000 makes the drop deterministic: the worker cannot reach
    the DB-write step before the process dies."""
    store_path = tmp_path / "dropped"
    rec_id = uuid4()

    proc = _run_subprocess(
        store_path, rec_id, flush_before_exit=False, coalesce_ms=60_000,
    )
    assert proc.returncode == 0, proc.stderr

    deferred = _json_markers(proc.stderr, "reinforce_deferred")
    drained = _json_markers(proc.stderr, "reinforce_drained")
    assert deferred and deferred[0]["n_ids"] == 1, (
        f"a queued write must always be accounted for at enqueue time; stderr={proc.stderr!r}"
    )
    assert not drained, (
        "the coalesce window must not have elapsed before the hard kill -- "
        "a drain marker here means the test setup failed to force the drop"
    )

    store = MemoryStore(path=store_path)
    got = store.get(rec_id)
    assert got is not None, "the explicitly-flushed record buffer is unaffected by the queue"
    assert got.literal_surface == "n", "verbatim content unaffected by a dropped boost"

    w = _edge_weight(store, rec_id, rec_id, "hebbian")
    assert w is None, (
        "the reinforcement edge boost must genuinely NOT have landed -- "
        "otherwise this scenario is not testing the drop path"
    )


def test_process_exit_marker_absence_is_itself_detectable(tmp_path):
    """Non-vacuity control: if the enqueue-time marker were silently
    disabled (the exact 'write vanished with zero signal' regression this
    suite guards against), that absence is directly observable here -- which
    is precisely what the accounted-for assertion in the sibling test above
    depends on to fail loudly instead of passing vacuously."""
    store_path = tmp_path / "suppressed"
    rec_id = uuid4()

    proc = _run_subprocess(
        store_path, rec_id, flush_before_exit=False, coalesce_ms=60_000,
        suppress_marker=True,
    )
    assert proc.returncode == 0, proc.stderr

    deferred = _json_markers(proc.stderr, "reinforce_deferred")
    assert not deferred, (
        "control precondition: the marker write must be genuinely suppressed "
        "for this scenario to demonstrate detectability"
    )
    # This is the exact condition under which
    # test_process_exit_dropped_write_absent_but_accounted_for's
    # `assert deferred and ...` line would raise -- proving that assertion
    # is sensitive to a real regression, not tautologically true.


def test_daemon_absent_apply_or_drop_without_error(tmp_path, capsys):
    """The reinforce queue is same-process (threading.Thread), never a
    separate daemon process -- this whole test runs with no daemon anywhere
    on the machine. A queued profile_modulates write must land after an
    explicit flush, or be accounted for by the overflow marker; it must
    never raise and never require a daemon process to exist."""
    store = MemoryStore(path=tmp_path)
    rec_a = _record("a")
    rec_b = _record("b")
    store.insert(rec_a)
    store.insert(rec_b)

    store.enable_reinforce_queue(coalesce_ms=10)
    try:
        store.queue_profile_modulate([(rec_a.id, rec_b.id)], [0.3])
        store._reinforce_queue.flush(timeout=5.0)
    finally:
        store.disable_reinforce_queue()

    stderr = capsys.readouterr().err
    dropped = bool(_json_markers(stderr, "reinforce_queue_pairs_overflow"))
    w = _edge_weight(store, rec_a.id, rec_b.id, "profile_modulates")

    assert (w is not None) or dropped, (
        "a queue-live, no-daemon profile_modulates write must land after an "
        "explicit flush, or be accounted for by the overflow marker -- never "
        "silently vanish"
    )
    if w is not None:
        assert w == pytest.approx(0.3, abs=1e-3)
