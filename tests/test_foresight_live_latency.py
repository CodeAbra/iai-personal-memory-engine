"""Live-daemon per-turn latency A/B for the assistant-tail counter-evidence
lane: capture-path added cost (ON minus OFF) against a stated ceiling, and
memory_recall p95 unchanged. Each arm spawns its OWN isolated daemon
subprocess (off-switch env set only in the OFF arm's subprocess env) -- the
user's live/production daemon is never touched, restarted, or read.

Ordering: the hermetic in-process capture-path delta already ran in
test_foresight_pack.py::test_three_lane_added_latency_report -- this live
measurement runs after that hermetic bound, never before it.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from _live_harness import _send_jsonrpc, spawn_live_daemon
from iai_mcp.capture import write_deferred_event
from iai_mcp.foresight import FORESIGHT_ASSISTANT_TAIL_OFF_ENV

pytestmark = pytest.mark.live

# Stated per-turn capture-path budget: refresh_pack's assistant-tail lane
# runs synchronously inside the memory_capture RPC, so its added cost must
# stay under this p95 delta versus the byte-identical lane-off baseline
# (off-switch-integrity: primary cap 4, ANN window 20, unchanged).
# Hermetic sanity check before this live run
# (test_foresight_pack.py::test_three_lane_added_latency_report): primary+
# derived+tail delta ~97ms -- comfortably under.
CAPTURE_ADDED_P95_CEILING_MS = 250.0

# The lane never touches memory_recall; this is the pre-existing warm-recall
# SLA (test_warm_recall_prodscale_isolated_daemon.py::_WARM_SLA_SEC), not a
# new bound -- recall p95 with the lane ON must stay inside it.
RECALL_P95_SLA_MS = 1000.0

_N_CAPTURE_REPS = 8  # first sample discarded as cold; p95 over the remaining warm reps
_N_RECALL_REPS = 6

_TAIL_SESSION = "live-latency-tail-session"
_TAIL_CLAIM_TEXT = (
    "alice reply: the dashboard feature is already queued for next week's release."
)
_TAIL_COUNTER_TEXT = (
    "alice note: the dashboard feature was pulled from the release plan last week."
)
_RECALL_CUE = "alice dashboard release plan status"


def _p95(samples: "list[float]") -> float:
    ordered = sorted(samples)
    idx = max(0, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


async def _measure_capture_reps(sock_path: Path, session_id: str, n: int) -> "list[float]":
    samples: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        resp = await _send_jsonrpc(
            sock_path, "memory_capture",
            {
                "text": f"alice latency probe turn {i} for session {session_id}",
                "session_id": session_id,
            },
            req_id=i + 1, timeout=30.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert "error" not in resp, f"memory_capture RPC errored: {resp.get('error')}"
        samples.append(elapsed_ms)
    return samples


async def _measure_recall_reps(sock_path: Path, cue: str, n: int) -> "list[float]":
    samples: list[float] = []
    for i in range(n):
        resp = await _send_jsonrpc(
            sock_path, "memory_recall",
            {"cue": cue, "session_id": f"live-latency-recall-{i}"},
            req_id=1000 + i, timeout=30.0,
        )
        assert "error" not in resp, f"memory_recall RPC errored: {resp.get('error')}"
        result = resp.get("result") or {}
        latency = result.get("_recall_latency_ms")
        assert latency is not None, (
            f"memory_recall response missing _recall_latency_ms: {result!r}"
        )
        samples.append(float(latency))
    return samples


def _seed_counter_evidence(store, _cue_vec) -> None:
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.embed import Embedder
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    vec = Embedder().embed(_TAIL_COUNTER_TEXT)
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=_TAIL_COUNTER_TEXT,
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[{"session_id": "seed", "role": "user"}],
        created_at=now, updated_at=now, tags=["role:user"], language="en",
        s5_trust_score=0.5, profile_modulation_gain={},
    )
    store.insert(rec)
    for i in range(20):
        filler_vec = [0.0] * EMBED_DIM
        filler_vec[i % EMBED_DIM] = 1.0
        store.insert(MemoryRecord(
            id=uuid4(), tier="episodic",
            literal_surface=f"alice filler record {i} unrelated to the dashboard",
            aaak_index="", embedding=filler_vec, community_id=None, centrality=0.0,
            detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
            last_reviewed=None, never_decay=False, never_merge=False,
            provenance=[{"session_id": "seed", "role": "user"}],
            created_at=now, updated_at=now, tags=["role:user"], language="en",
            s5_trust_score=0.5, profile_modulation_gain={},
        ))


def test_assistant_tail_capture_added_latency_and_recall_sla_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv(FORESIGHT_ASSISTANT_TAIL_OFF_ENV, "1")
    off_gen = spawn_live_daemon(tmp_path / "off-arm", monkeypatch, cue=_RECALL_CUE)
    off_ns = next(off_gen)
    try:
        off_samples = asyncio.run(
            _measure_capture_reps(off_ns.sock_path, "live-latency-off-session", _N_CAPTURE_REPS)
        )
    finally:
        try:
            next(off_gen)
        except StopIteration:
            pass

    monkeypatch.delenv(FORESIGHT_ASSISTANT_TAIL_OFF_ENV, raising=False)
    on_gen = spawn_live_daemon(
        tmp_path / "on-arm", monkeypatch, cue=_RECALL_CUE, seed=_seed_counter_evidence,
    )
    on_ns = next(on_gen)
    try:
        monkeypatch.setenv("HOME", str(on_ns.store_dir.parent))
        write_deferred_event(_TAIL_SESSION, "assistant", _TAIL_CLAIM_TEXT)

        on_samples = asyncio.run(
            _measure_capture_reps(on_ns.sock_path, _TAIL_SESSION, _N_CAPTURE_REPS)
        )
        recall_samples = asyncio.run(
            _measure_recall_reps(on_ns.sock_path, _RECALL_CUE, _N_RECALL_REPS)
        )
    finally:
        try:
            next(on_gen)
        except StopIteration:
            pass

    # Warm repeats only: drop the first (cold, model/index warmup) sample.
    off_warm = off_samples[1:]
    on_warm = on_samples[1:]

    off_p95 = _p95(off_warm)
    on_p95 = _p95(on_warm)
    delta_p95_ms = on_p95 - off_p95
    recall_p95 = _p95(recall_samples)

    print(
        f"\nlive capture-path latency: off_p95={off_p95:.1f}ms "
        f"on_p95={on_p95:.1f}ms delta={delta_p95_ms:.1f}ms "
        f"(n={len(off_warm)} warm reps/arm)"
    )
    print(f"live recall p95 (lane ON): {recall_p95:.1f}ms (n={len(recall_samples)} reps)")

    assert delta_p95_ms <= CAPTURE_ADDED_P95_CEILING_MS, (
        f"assistant-tail lane added {delta_p95_ms:.1f}ms p95 to the capture "
        f"path (off={off_p95:.1f}ms, on={on_p95:.1f}ms) -- over the stated "
        f"{CAPTURE_ADDED_P95_CEILING_MS}ms ceiling"
    )
    assert recall_p95 <= RECALL_P95_SLA_MS, (
        f"memory_recall p95 ({recall_p95:.1f}ms) with the lane ON exceeded "
        f"the existing warm-recall SLA ({RECALL_P95_SLA_MS}ms) -- the lane "
        f"must never touch the recall path"
    )
