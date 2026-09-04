"""Sustained-load overflow probe for the reinforce queue.

Pushes enqueue throughput past the queue's drain rate to force overflow,
asserting the drop is observable via the existing stderr JSON marker
convention (extended to the profile_modulates edge_type) and that the queue
degrades gracefully under sustained pressure: no raise, no store corruption,
no unbounded spill. Drives the real `MemoryStore.queue_profile_modulate`
entry point with several distinct delta values per call, matching the
production multi-distinct-delta fan-out (pipeline.py sums a per-hit gain, so
a recall with several hits groups into several separate queue puts, not
one). Opt-in (--runslow) throughput probe, not a correctness unit -- does
not run in the default gate.
"""
from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stderr
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.reinforce_queue import ReinforceWriteQueue
from iai_mcp.store import MemoryStore
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


@pytest.mark.slow
def test_sustained_load_forces_observable_overflow_and_degrades_gracefully(
    tmp_path, monkeypatch,
):
    store = MemoryStore(path=tmp_path)
    recs = [_record(f"r{i}") for i in range(20)]
    for r in recs:
        store.insert(r)

    real_boost_edges = store.boost_edges

    def _slow_boost_edges(*args, **kwargs):
        # Deliberately slower than the producer loop below, so the queue's
        # bounded size overflows deterministically within the probe window
        # rather than depending on machine-speed timing luck.
        time.sleep(0.05)
        return real_boost_edges(*args, **kwargs)

    monkeypatch.setattr(store, "boost_edges", _slow_boost_edges)

    q = ReinforceWriteQueue(store, coalesce_ms=20, max_queue_size=4, max_batch=8)
    q.start()
    # Route the real production entry point through this bounded queue, same
    # as MemoryStore.enable_reinforce_queue() wires it in the daemon.
    monkeypatch.setattr(store, "_reinforce_queue", q)

    real_enqueue_pairs = q.enqueue_pairs
    n_enqueue_calls = 0

    def _counting_enqueue_pairs(*args, **kwargs):
        nonlocal n_enqueue_calls
        n_enqueue_calls += 1
        return real_enqueue_pairs(*args, **kwargs)

    monkeypatch.setattr(q, "enqueue_pairs", _counting_enqueue_pairs)

    # Distinct delta values per call, modeling pipeline.py's per-hit
    # total_gain: a recall with several hits produces several distinct
    # deltas, so queue_profile_modulate's internal by-delta grouping fans
    # out into one enqueue_pairs call per distinct delta, not one per recall.
    deltas_per_recall = [0.02, 0.05, 0.08, 0.11]

    buf = io.StringIO()
    n_recalls = 0
    n_pairs_total = 0
    try:
        with redirect_stderr(buf):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                pairs = []
                for _ in deltas_per_recall:
                    a = recs[n_pairs_total % 20].id
                    b = recs[(n_pairs_total + 1) % 20].id
                    pairs.append((a, b))
                    n_pairs_total += 1
                store.queue_profile_modulate(pairs, list(deltas_per_recall))
                n_recalls += 1
    finally:
        q.flush(timeout=10.0)
        q.stop()

    stderr_text = buf.getvalue()
    overflow_markers = _json_markers(stderr_text, "reinforce_queue_pairs_overflow")

    assert n_enqueue_calls == n_recalls * len(deltas_per_recall), (
        f"each queue_profile_modulate call with {len(deltas_per_recall)} distinct "
        f"deltas must fan out into {len(deltas_per_recall)} enqueue_pairs calls; "
        f"got {n_enqueue_calls} calls across {n_recalls} recalls"
    )
    assert overflow_markers, (
        f"sustained enqueue ({n_recalls} recalls / {n_enqueue_calls} enqueue_pairs "
        f"calls against a slowed drain) must produce at least one overflow marker; "
        f"got none. stderr={stderr_text!r}"
    )
    assert all(m.get("edge_type") == "profile_modulates" for m in overflow_markers), (
        f"overflow marker must carry the profile_modulates edge_type; got {overflow_markers!r}"
    )

    drop_rate = len(overflow_markers) / n_enqueue_calls
    # Evidence, not a threshold: the assertion above is graceful-degradation
    # + observability, not a specific drop percentage.
    print(
        f"sustained-load drop rate: {drop_rate:.2%} "
        f"({len(overflow_markers)}/{n_enqueue_calls} enqueue_pairs calls, "
        f"{n_recalls} recalls x {len(deltas_per_recall)} distinct deltas)"
    )

    # Graceful degrade: no raise reached this point (a raise from enqueue_pairs
    # or the producer loop would have failed the test above), the store is
    # not corrupted -- a follow-up recall-shaped read still returns valid hits.
    got = store.get(recs[0].id)
    assert got is not None
    assert got.literal_surface == recs[0].literal_surface

    # ReinforceWriteQueue has no disk-spill path (unlike ProvenanceWriteQueue) --
    # confirm sustained overflow did not create unbounded spill artifacts.
    spill_paths = [
        p for p in tmp_path.rglob("*")
        if "overflow" in p.name.lower() or "spill" in p.name.lower()
    ]
    assert spill_paths == [], (
        f"ReinforceWriteQueue must not create overflow/spill files on disk; "
        f"found {spill_paths}"
    )
