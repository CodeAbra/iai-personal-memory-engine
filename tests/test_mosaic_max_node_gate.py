from __future__ import annotations

import time
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.community import CommunityAssignment, _compute_centroid
from iai_mcp.pipeline import _community_gate
from iai_mcp.types import EMBED_DIM

pytestmark = pytest.mark.perf


def _unit_axis(i: int, dim: int = EMBED_DIM) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _build_drift_fixture() -> tuple[
    CommunityAssignment, dict[UUID, list[float]], list[float], UUID, UUID
]:
    dim = EMBED_DIM
    cue = _unit_axis(0, dim)

    a_perfect = _unit_axis(0, dim)
    a_orthogonals = [_unit_axis(2 + i, dim) for i in range(10)]

    b_template = [0.0] * dim
    b_template[0] = 0.5
    b_template[1] = (3.0 ** 0.5) / 2.0
    b_members = [list(b_template) for _ in range(11)]

    a_member_ids = [uuid4() for _ in range(11)]
    b_member_ids = [uuid4() for _ in range(11)]
    comm_A = uuid4()
    comm_B = uuid4()

    member_embeddings: dict[UUID, list[float]] = {}
    for mid, vec in zip(a_member_ids, [a_perfect, *a_orthogonals]):
        member_embeddings[mid] = vec
    for mid, vec in zip(b_member_ids, b_members):
        member_embeddings[mid] = vec

    centroid_A = _compute_centroid([a_perfect, *a_orthogonals])
    centroid_B = _compute_centroid(b_members)

    mid_regions = {
        comm_A: a_member_ids,
        comm_B: b_member_ids,
    }
    node_to_community = {}
    for mid in a_member_ids:
        node_to_community[mid] = comm_A
    for mid in b_member_ids:
        node_to_community[mid] = comm_B

    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids={comm_A: centroid_A, comm_B: centroid_B},
        modularity=0.0,
        backend="leiden-test-drift-witness",
        top_communities=[comm_A, comm_B],
        mid_regions=mid_regions,
    )
    return assignment, member_embeddings, cue, comm_A, comm_B


def _build_one_per_community(
    n: int,
) -> tuple[CommunityAssignment, dict[UUID, list[float]]]:
    dim = EMBED_DIM
    member_ids = [uuid4() for _ in range(n)]
    comm_ids = [uuid4() for _ in range(n)]
    node_to_community: dict[UUID, UUID] = {}
    centroids: dict[UUID, list[float]] = {}
    mid_regions: dict[UUID, list[UUID]] = {}
    member_embeddings: dict[UUID, list[float]] = {}
    for i in range(n):
        vec = _unit_axis(i, dim)
        node_to_community[member_ids[i]] = comm_ids[i]
        centroids[comm_ids[i]] = list(vec)
        mid_regions[comm_ids[i]] = [member_ids[i]]
        member_embeddings[member_ids[i]] = vec
    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=centroids,
        modularity=0.0,
        backend="leiden-test-fragment",
        top_communities=comm_ids[:3],
        mid_regions=mid_regions,
    )
    return assignment, member_embeddings


def test_max_node_gate_finds_correct_community_when_centroid_drifts():
    assignment, member_embeddings, cue, comm_A, comm_B = _build_drift_fixture()

    centroid_order = _community_gate(cue, assignment, top_n=2)
    assert centroid_order[0] == comm_B, (
        "Drift-fixture invariant violated: centroid-cosine should pick "
        f"comm_B first (centroid(B).cue=0.5, centroid(A).cue~0.30). "
        f"Got centroid_order[0]={centroid_order[0]}. Recompute fixture."
    )

    max_node_order = _community_gate(
        cue, assignment, top_n=2, member_embeddings=member_embeddings,
    )
    assert max_node_order[0] == comm_A, (
        "B* contract violated: max-node-cosine MUST pick comm_A first "
        "(max(A members . cue) = 1.0 vs max(B members . cue) = 0.5). "
        f"Got max_node_order={max_node_order}. "
        "If this fails, _community_gate did NOT switch to max-node when "
        "member_embeddings was passed."
    )


def test_max_node_gate_robust_to_fragmentation():
    n = 50
    assignment, member_embeddings = _build_one_per_community(n)
    target_member: UUID | None = None
    for mid, emb in member_embeddings.items():
        if emb[5] == 1.0 and all(
            emb[i] == 0.0 for i in range(EMBED_DIM) if i != 5
        ):
            target_member = mid
            break
    assert target_member is not None, "fixture lookup failed"
    target_comm = assignment.node_to_community[target_member]

    cue = _unit_axis(5)
    top3 = _community_gate(
        cue, assignment, top_n=3, member_embeddings=member_embeddings,
    )
    assert target_comm in top3, (
        "Fragmentation-robustness violated: the community holding the "
        "cos-1.0 record is NOT in top-3. With 1-record-per-community "
        "geometry on orthogonal axes, max-node-cosine MUST pick the "
        "axis-aligned community first. "
        f"Got top3={top3}, target_comm={target_comm}."
    )
    assert top3[0] == target_comm, (
        f"Max-node top-1 should be the cos-1.0 community; got {top3[0]}."
    )


def test_max_node_gate_backwards_compat_without_member_embeddings():
    n = 50
    assignment, _ = _build_one_per_community(n)
    cue = _unit_axis(5)

    out_no_kwarg = _community_gate(cue, assignment, top_n=5)
    out_explicit_none = _community_gate(
        cue, assignment, top_n=5, member_embeddings=None,
    )
    assert out_no_kwarg == out_explicit_none, (
        "Backwards-compat broken: omitting member_embeddings vs passing "
        "None must yield bit-identical results.\n"
        f"omit:     {out_no_kwarg}\n"
        f"explicit: {out_explicit_none}"
    )
    assert len(out_no_kwarg) == 5


def test_max_node_gate_determinism():
    n = 30
    assignment, member_embeddings = _build_one_per_community(n)
    cue = _unit_axis(7)

    runs = [
        _community_gate(
            cue, assignment, top_n=5, member_embeddings=member_embeddings,
        )
        for _ in range(5)
    ]
    first = runs[0]
    for i, r in enumerate(runs[1:], start=1):
        assert r == first, (
            f"Determinism violated at run {i}: {r} != run0 {first}"
        )

    tied = first[1:]
    tied_strs = [str(u) for u in tied]
    assert tied_strs == sorted(tied_strs), (
        "Stable tie-break must be ascending UUID-str within a tied score "
        f"bucket. Got: {tied_strs}, expected: {sorted(tied_strs)}"
    )


def test_max_node_gate_perf_under_5ms_at_n5000():
    rng = np.random.default_rng(seed=42)
    dim = EMBED_DIM
    n_communities = 100
    members_per_community = 50
    member_ids: list[UUID] = []
    member_embeddings: dict[UUID, np.ndarray] = {}
    mid_regions: dict[UUID, list[UUID]] = {}
    centroids: dict[UUID, list[float]] = {}
    node_to_community: dict[UUID, UUID] = {}
    top_communities: list[UUID] = []
    for c_idx in range(n_communities):
        comm_id = uuid4()
        top_communities.append(comm_id)
        mid_regions[comm_id] = []
        member_vecs: list[list[float]] = []
        for _ in range(members_per_community):
            v = rng.standard_normal(dim).astype(np.float32)
            n_v = float(np.linalg.norm(v))
            if n_v > 0:
                v = v / n_v
            else:
                v = np.zeros(dim, dtype=np.float32)
            m_id = uuid4()
            member_ids.append(m_id)
            mid_regions[comm_id].append(m_id)
            node_to_community[m_id] = comm_id
            member_embeddings[m_id] = v
            member_vecs.append(v.tolist())
        centroids[comm_id] = _compute_centroid(member_vecs)
    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=centroids,
        modularity=0.0,
        backend="leiden-test-perf",
        top_communities=top_communities[:7],
        mid_regions=mid_regions,
    )
    cue = rng.standard_normal(dim).astype(np.float32)
    cn = float(np.linalg.norm(cue))
    cue = (cue / cn).tolist() if cn > 0 else cue.tolist()

    _community_gate(
        cue, assignment, top_n=3, member_embeddings=member_embeddings,
    )

    n_runs = 20
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _community_gate(
            cue, assignment, top_n=3, member_embeddings=member_embeddings,
        )
    elapsed = time.perf_counter() - t0
    mean_ms = (elapsed / n_runs) * 1000.0
    assert mean_ms < 5.0, (
        f"Max-node gate perf regression: mean wall time {mean_ms:.3f} ms "
        f"exceeds 5.0 ms budget at N=5000 members over 100 communities. "
        f"Total {n_runs} runs took {elapsed*1000:.1f} ms."
    )


def test_max_node_gate_scans_candidate_embeddings_not_full_assignment():
    member_ids = [uuid4() for _ in range(1000)]
    comm_ids = [uuid4() for _ in range(1000)]
    node_to_community = {
        member_id: comm_id
        for member_id, comm_id in zip(member_ids, comm_ids)
    }
    assignment = CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids={comm_id: _unit_axis(0) for comm_id in comm_ids},
        modularity=0.0,
        backend="leiden-test-wide",
        top_communities=comm_ids[:3],
        mid_regions={
            comm_id: [member_id]
            for member_id, comm_id in zip(member_ids, comm_ids)
        },
    )
    candidate = member_ids[123]
    out = _community_gate(
        _unit_axis(4),
        assignment,
        top_n=3,
        member_embeddings={candidate: _unit_axis(4)},
    )

    assert out == [node_to_community[candidate]]


def test_max_node_gate_tolerates_non_finite_embeddings():
    member_a = uuid4()
    member_b = uuid4()
    comm_a = uuid4()
    comm_b = uuid4()
    assignment = CommunityAssignment(
        node_to_community={
            member_a: comm_a,
            member_b: comm_b,
        },
        community_centroids={
            comm_a: _unit_axis(0),
            comm_b: _unit_axis(1),
        },
        modularity=0.0,
        backend="leiden-test-non-finite",
        top_communities=[comm_a, comm_b],
        mid_regions={
            comm_a: [member_a],
            comm_b: [member_b],
        },
    )

    out = _community_gate(
        [float("nan")] + [0.0] * (EMBED_DIM - 1),
        assignment,
        top_n=2,
        member_embeddings={
            member_a: np.asarray([float("inf")] + [0.0] * (EMBED_DIM - 1)),
            member_b: _unit_axis(1),
        },
    )

    assert out == sorted([comm_a, comm_b], key=str)
