from __future__ import annotations

import numpy as np


def test_build_csr_from_edge_lists_round_trip() -> None:
    from iai_mcp.sigma import _build_csr_from_edge_lists

    u_list = [0, 1, 0]
    v_list = [1, 2, 2]
    indptr, indices, data = _build_csr_from_edge_lists(u_list, v_list, 3)
    assert indptr.tolist() == [0, 2, 4, 6]
    assert sorted(indices[indptr[0]:indptr[1]].tolist()) == [1, 2]
    assert sorted(indices[indptr[1]:indptr[2]].tolist()) == [0, 2]
    assert sorted(indices[indptr[2]:indptr[3]].tolist()) == [0, 1]
    assert np.allclose(data, 1.0)


def test_build_csr_from_empty_edge_lists() -> None:
    from iai_mcp.sigma import _build_csr_from_edge_lists

    indptr, indices, data = _build_csr_from_edge_lists([], [], 5)
    assert indptr.tolist() == [0, 0, 0, 0, 0, 0]
    assert indices.tolist() == []
    assert data.tolist() == []


def test_induced_csr_extracts_largest_component_correctly() -> None:
    from iai_mcp.sigma import _build_csr_from_edge_lists, _induced_csr_from_component
    from iai_mcp_native import graph as lilli_graph

    edges_k5 = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edges_k3 = [(u, v) for u in range(5, 8) for v in range(u + 1, 8)]
    all_edges = edges_k5 + edges_k3
    u_list = [u for u, _v in all_edges]
    v_list = [v for _u, v in all_edges]
    indptr, indices, data = _build_csr_from_edge_lists(u_list, v_list, 8)

    components = lilli_graph.connected_components(indptr, indices, 8)
    largest = max(components, key=len)
    assert sorted(largest) == [0, 1, 2, 3, 4]

    sub_indptr, sub_indices, sub_data, sub_n = _induced_csr_from_component(
        indptr, indices, data, list(largest)
    )
    assert sub_n == 5
    assert sub_indptr[-1] == 20
    for u in range(5):
        row_len = sub_indptr[u + 1] - sub_indptr[u]
        assert row_len == 4, f"K5 row {u} degree should be 4, got {row_len}"


def test_largest_component_csr_on_connected_graph_is_identity() -> None:
    from iai_mcp.sigma import (
        _build_csr_from_edge_lists,
        _largest_component_csr,
    )

    edges = [(u, v) for u in range(4) for v in range(u + 1, 4)]
    u_list = [u for u, _v in edges]
    v_list = [v for _u, v in edges]
    indptr, indices, data = _build_csr_from_edge_lists(u_list, v_list, 4)

    sub_indptr, sub_indices, sub_data, sub_n = _largest_component_csr(
        indptr, indices, data, 4
    )
    assert sub_n == 4
    assert np.array_equal(sub_indptr, indptr)
    assert np.array_equal(sub_indices, indices)
    assert np.array_equal(sub_data, data)
