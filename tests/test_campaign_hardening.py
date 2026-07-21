"""Regression coverage for the mosaic + Hippo hardening campaign.

Each test pins one defect the campaign fixed so a future refactor that
re-introduces it fails loudly rather than silently:

* mosaic ``compute_delta_q_cpm`` self-loop exclusion — a super-node's own
  internal mass must not inflate its links-to-current-community sum.
* ``MemoryGraph.two_hop_neighborhood`` determinism — a hash-randomized set
  sweep made the recall candidate set differ across processes.
* Hippo ``_validate_columns`` — caller-supplied column names are checked to
  the identifier grammar as a defense-in-depth guard against injection.
* Hippo ``_is_snapshot_fence`` — a checkpoint-invalidated read-only snapshot
  is a retryable fence, distinct from a genuine operational error.
* ``runtime_graph_cache.save_with_generation`` — the module generation is
  advanced only after a durable snapshot is on disk, never on a failed save.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest


def test_delta_q_cpm_excludes_self_loop() -> None:
    """A self-loop on the moving node must not count toward the
    links-to-current-community sum: its mass is degree, not an inter-node link.
    Two CSRs identical except for a node-0 self-loop must yield the same ΔQ."""
    from iai_mcp.mosaic import compute_delta_q_cpm

    # Node 0 in community 0, node 1 in community 1; a single real edge 0<->1.
    # CSR_B: no self-loop.
    indptr_b = np.array([0, 1, 2], dtype=np.int64)
    indices_b = np.array([1, 0], dtype=np.int64)
    data_b = np.array([1.0, 1.0], dtype=np.float64)

    # CSR_A: node 0 additionally carries a self-loop (0,0) with heavy weight.
    indptr_a = np.array([0, 2, 3], dtype=np.int64)
    indices_a = np.array([0, 1, 0], dtype=np.int64)
    data_a = np.array([5.0, 1.0, 1.0], dtype=np.float64)

    partition = np.array([0, 1], dtype=np.int64)
    sigma_tot = np.array([1.0, 1.0], dtype=np.float64)
    k_i = 1.0
    gamma = 1.0
    two_m = 2.0

    dq_no_loop = compute_delta_q_cpm(
        0, 0, 1, indptr_b, indices_b, data_b,
        partition, sigma_tot, k_i, gamma, two_m,
    )
    dq_with_loop = compute_delta_q_cpm(
        0, 0, 1, indptr_a, indices_a, data_a,
        partition, sigma_tot, k_i, gamma, two_m,
    )
    assert abs(dq_with_loop - dq_no_loop) < 1e-9, (
        "self-loop mass leaked into the ΔQ neighbor sums: "
        f"{dq_with_loop} != {dq_no_loop}"
    )


def test_two_hop_neighborhood_is_deterministic_and_sorted() -> None:
    from iai_mcp.graph import MemoryGraph

    seed = uuid.uuid4()
    neighbours = [uuid.uuid4() for _ in range(6)]

    def build() -> MemoryGraph:
        g = MemoryGraph()
        g.add_node(seed, None, [0.0])
        for n in neighbours:
            g.add_node(n, None, [0.0])
        # Insert edges in a shuffled-but-fixed order; equal weights force the
        # tie-break to carry the determinism, not the weight ordering.
        for n in reversed(neighbours):
            g.add_edge(seed, n, weight=1.0)
        return g

    runs = [build().two_hop_neighborhood([seed], top_k=5) for _ in range(5)]
    for r in runs[1:]:
        assert r == runs[0], "two_hop_neighborhood output is not reproducible"
    assert runs[0] == sorted(runs[0]), "result must be returned in sorted order"


def test_validate_columns_rejects_injection() -> None:
    from iai_mcp.hippo._table import _validate_columns

    assert _validate_columns(["id", "vec_label", "payload_v5"]) == [
        "id", "vec_label", "payload_v5",
    ]
    for bad in ["a b", "1col", "col;DROP TABLE x", 'x") --', "col-name", ""]:
        with pytest.raises(ValueError):
            _validate_columns([bad])


def test_is_snapshot_fence_discriminates_marker() -> None:
    from iai_mcp import errors
    from iai_mcp.hippo._ro_pool import (
        _RO_SNAPSHOT_FENCE_MARKER,
        _is_snapshot_fence,
    )

    fence = errors.OperationalError(
        f"{_RO_SNAPSHOT_FENCE_MARKER}: checkpoint generation advanced"
    )
    assert _is_snapshot_fence(fence) is True

    genuine = errors.OperationalError("database is locked")
    assert _is_snapshot_fence(genuine) is False


def test_save_with_generation_no_advance_on_failed_save(monkeypatch) -> None:
    from iai_mcp import runtime_graph_cache as rgc

    before = rgc.get_current_generation()
    monkeypatch.setattr(rgc, "save", lambda *a, **k: False)

    result = rgc.save_with_generation(
        store=object(), assignment=None, rich_club=None,
    )
    assert result is False
    assert rgc.get_current_generation() == before, (
        "generation advanced despite a failed save — snapshot/generation "
        "would be out of sync"
    )


def test_save_with_generation_snapshot_epoch_matches_module(monkeypatch) -> None:
    """The epoch save() stamps into the snapshot must equal the module epoch
    after the save commits — a mismatch makes every overlay read bypass on
    epoch_mismatch."""
    from iai_mcp import runtime_graph_cache as rgc

    written: list[int] = []

    def fake_save(*a, **k):
        written.append(rgc._snapshot_generation())
        return True

    monkeypatch.setattr(rgc, "save", fake_save)
    result = rgc.save_with_generation(
        store=object(), assignment=None, rich_club=None,
    )
    assert result is True
    assert written == [rgc.get_current_generation()], (
        f"snapshot epoch {written} != module epoch "
        f"{rgc.get_current_generation()} after a successful save"
    )
    # Override must not leak past the call: a later plain save() stamps the
    # (now committed) module generation, not a stale next_gen.
    assert rgc._snapshot_generation() == rgc.get_current_generation()
