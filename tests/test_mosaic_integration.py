from __future__ import annotations

import inspect
import random
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.community import (
    CommunityAssignment,
    detect_communities,
)
from iai_mcp.mosaic_lineage import LineageReport
from iai_mcp.graph import MemoryGraph


REPO_ROOT = Path(__file__).resolve().parent.parent


def _random_emb(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(384)]


def _make_two_clique_graph(n_per_clique: int = 150) -> MemoryGraph:
    g = MemoryGraph()
    clique_a = [uuid4() for _ in range(n_per_clique)]
    clique_b = [uuid4() for _ in range(n_per_clique)]
    for i, n in enumerate(clique_a):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    for i, n in enumerate(clique_b):
        g.add_node(n, community_id=None, embedding=_random_emb(10_000 + i))
    for i in range(n_per_clique):
        for j in range(i + 1, n_per_clique):
            g.add_edge(clique_a[i], clique_a[j])
            g.add_edge(clique_b[i], clique_b[j])
    return g


def test_detect_communities_uses_mosaic_backend() -> None:
    g = _make_two_clique_graph()
    a = detect_communities(g)
    assert a.backend == "leiden-custom"
    assert a.modularity >= 0.20


def test_detect_communities_accepts_prior_mode_seeded() -> None:
    g = _make_two_clique_graph()
    a = detect_communities(g, prior=None, prior_mode="seeded")
    assert isinstance(a, CommunityAssignment)


def test_detect_communities_accepts_prior_mode_cold() -> None:
    g = _make_two_clique_graph()
    first = detect_communities(g, prior=None, prior_mode="seeded")
    second = detect_communities(g, prior=first, prior_mode="cold")
    prior_uuids = set(first.node_to_community.values())
    new_uuids = set(second.node_to_community.values())
    assert prior_uuids.isdisjoint(new_uuids), (
        "cold mode must discard prior UUIDs; "
        f"overlap = {prior_uuids & new_uuids}"
    )


def test_detect_communities_default_prior_mode_is_seeded() -> None:
    sig = inspect.signature(detect_communities)
    assert "prior_mode" in sig.parameters
    assert sig.parameters["prior_mode"].default == "seeded"


def test_community_assignment_lineage_report_field_exists() -> None:
    assert "lineage_report" in CommunityAssignment.__dataclass_fields__
    fld = CommunityAssignment.__dataclass_fields__["lineage_report"]
    assert fld.default is None


def test_lineage_report_populated_on_leiden_path() -> None:
    g = _make_two_clique_graph()
    a = detect_communities(g)
    assert a.lineage_report is not None
    assert isinstance(a.lineage_report, LineageReport)


def test_lineage_report_empty_on_flat_fallback() -> None:
    g = MemoryGraph()
    for i in range(50):
        g.add_node(uuid4(), community_id=None, embedding=_random_emb(i))
    a = detect_communities(g)
    assert a.backend == "flat"
    assert a.lineage_report is not None
    assert isinstance(a.lineage_report, LineageReport)


def test_aggregate_uses_pick_merge_survivor() -> None:
    src = (REPO_ROOT / "src" / "iai_mcp" / "mosaic.py").read_text()
    assert "pick_merge_survivor" in src, (
        "_aggregate must call lineage.pick_merge_survivor(...)"
    )


def test_detect_communities_uses_cpm_floor_not_legacy_0_2() -> None:
    src = (REPO_ROOT / "src" / "iai_mcp" / "community.py").read_text()
    assert "from iai_mcp.mosaic_policy import" in src, (
        "community.py must import the CPM-calibrated floor"
    )
    assert "CPM_MODULARITY_FLOOR" in src, (
        "community.py must reference CPM_MODULARITY_FLOOR in the mid-N guard"
    )


def test_sleep_pipeline_crisis_mode_uses_prior_mode_cold() -> None:
    src = (REPO_ROOT / "src" / "iai_mcp" / "lilli" / "cycle" / "sleep_pipeline" / "_crisis.py").read_text()
    assert 'prior_mode="cold"' in src, (
        "sleep_pipeline.py crisis_recluster must call detect_communities "
        "with prior_mode=\"cold\""
    )


def test_sleep_pipeline_does_not_use_run_leiden_directly() -> None:
    pkg = REPO_ROOT / "src" / "iai_mcp" / "lilli" / "cycle" / "sleep_pipeline"
    for module in sorted(pkg.glob("*.py")):
        src = module.read_text()
        assert "from iai_mcp.community import _run_leiden" not in src, (
            f"{module.name} must NOT import _run_leiden directly; use "
            "detect_communities with prior_mode='cold' instead"
        )


def test_retrieve_uses_seeded_mode() -> None:
    src = (REPO_ROOT / "src" / "iai_mcp" / "retrieve.py").read_text()
    assert 'prior_mode="seeded"' in src, (
        "retrieve.py must call detect_communities with prior_mode=\"seeded\""
    )


def test_sigma_uses_seeded_mode() -> None:
    src = (REPO_ROOT / "src" / "iai_mcp" / "sigma.py").read_text()
    assert 'prior_mode="seeded"' in src, (
        "sigma.py must call detect_communities with prior_mode=\"seeded\""
    )


def test_retrieve_continuity_preserves_uuids_across_unchanged_graph() -> None:
    g = _make_two_clique_graph()
    first = detect_communities(g, prior=None, prior_mode="seeded")
    second = detect_communities(g, prior=first, prior_mode="seeded")
    for node, comm_first in first.node_to_community.items():
        assert second.node_to_community[node] == comm_first, (
            f"node {node} community drifted: {comm_first} -> "
            f"{second.node_to_community[node]}"
        )


def test_sigma_community_count_stable_across_calls() -> None:
    g = _make_two_clique_graph()
    first = detect_communities(g, prior=None, prior_mode="seeded")
    second = detect_communities(g, prior=first, prior_mode="seeded")
    assert (
        len(set(second.node_to_community.values()))
        == len(set(first.node_to_community.values()))
    ), "community count must be stable across re-runs of an unchanged graph"
