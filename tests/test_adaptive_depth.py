"""Tests for the graded adaptive-depth controller.

Contract:
  - Spread depth is a function of a CONFIDENCE signal (top-hit cosine + hit
    count above threshold), not a function of prior wall-clock latency.
  - High confidence (top-hit cosine high, several hits above threshold) keeps
    spread shallow-and-fast.
  - Low confidence (weak top hit, few hits) escalates to one bounded step.
  - Escalation is bounded: widened candidate count <= cap (e.g. 2000), single
    step (not a loop), never calls store.all_records().
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.types import EMBED_DIM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _build_simple_store(tmp_path: Path, n_records: int = 50):
    from iai_mcp.store import MemoryStore
    store = MemoryStore(path=tmp_path / "depth_store")
    for i in range(n_records):
        store.insert(_make(text=f"User record {i}", vec=_unit_vec(i + 100)))
    return store


# ---------------------------------------------------------------------------
# Test: depth is driven by confidence, not wall-clock latency
# ---------------------------------------------------------------------------


def test_spread_depth_driven_by_confidence_not_wallclock(tmp_path):
    """High-confidence signal keeps depth shallow; low-confidence escalates.

    The controller must return different depths based on confidence scalars
    (top-hit cosine + hit count) while the wall-clock latency global is held
    constant, proving the depth decision is decoupled from wall-clock.
    """
    from iai_mcp.pipeline import compute_spread_depth

    import iai_mcp.pipeline as _pl

    # Fix the wall-clock latency global at a high value (normally would trigger
    # the old F2 binary degrade).  The NEW controller must ignore this.
    _pl._last_recall_latency_ms = 3000.0

    # High-confidence scenario: top cosine 0.95, 8 hits above threshold
    high_conf_depth = compute_spread_depth(top_cosine=0.95, hit_count_above_threshold=8)

    # Low-confidence scenario: top cosine 0.35, 1 hit above threshold
    low_conf_depth = compute_spread_depth(top_cosine=0.35, hit_count_above_threshold=1)

    # Reset
    _pl._last_recall_latency_ms = 0.0

    # High confidence -> shallow (short spread path is enough)
    assert high_conf_depth >= 0, "depth must be non-negative"
    # Low confidence -> escalate (must be strictly deeper than high-confidence)
    assert low_conf_depth > high_conf_depth, (
        "low-confidence recall must escalate spread depth beyond the "
        "high-confidence depth; confidence, not wall-clock, drives depth"
    )
    # Depth must never be negative
    assert low_conf_depth >= 0


def test_bounded_escalation(tmp_path, monkeypatch):
    """Escalated recall is bounded: cap <= 2000, single step, no all_records scan.

    The bounded-escalation constraint: hard candidate cap, single widening step,
    never store.all_records() — matches the recency-buffer precedent (maxlen=200
    hard cap) and the SYNTHESIS.md never-scan invariant.
    """
    from iai_mcp.pipeline import (
        ADAPTIVE_ESCALATION_CAP,
        escalate_recall_candidates,
    )
    from iai_mcp.store import MemoryStore

    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "d.sock"))

    store = _build_simple_store(tmp_path)

    # The cap must not exceed 2000 (per the bounded-escalation constraint)
    assert ADAPTIVE_ESCALATION_CAP <= 2000, (
        f"escalation cap {ADAPTIVE_ESCALATION_CAP} exceeds the 2000 maximum; "
        "a larger cap risks reintroducing the scan-latency regime"
    )

    # Track whether all_records was called (must never be)
    all_records_called = []
    real_all_records = store.all_records

    def _spy_all_records(*args, **kwargs):
        all_records_called.append(True)
        return real_all_records(*args, **kwargs)

    monkeypatch.setattr(store, "all_records", _spy_all_records)

    # Run escalated recall: expects a widened candidate dict, bounded by the cap
    result = escalate_recall_candidates(
        store=store,
        base_candidate_ids=[uuid4() for _ in range(5)],
        cue_vec=[0.1] * EMBED_DIM,
        cap=ADAPTIVE_ESCALATION_CAP,
    )

    store.close()

    assert not all_records_called, (
        "escalate_recall_candidates called store.all_records() — "
        "escalation must be a bounded widened index read, not a corpus scan"
    )
    assert len(result) <= ADAPTIVE_ESCALATION_CAP, (
        f"escalated candidate count {len(result)} exceeds cap "
        f"{ADAPTIVE_ESCALATION_CAP}"
    )
