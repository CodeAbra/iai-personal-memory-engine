
from __future__ import annotations

from typing import Callable, Iterable

import pytest

pytest.importorskip("networkx")
pytest.importorskip("hypothesis")
pytest.importorskip("hypothesis_networkx")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis_networkx import graph_builder  # noqa: E402


_GRAPH_STRATEGY = graph_builder(
    min_nodes=20,
    max_nodes=200,
    connected=True,
    self_loops=False,
)


def differential_check(
    g: "nx.Graph",
    our_metric_fn: Callable[["nx.Graph"], float],
    oracle_metric_fn: Callable[["nx.Graph"], float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-12,
) -> None:
    our_value = our_metric_fn(g)
    oracle_value = oracle_metric_fn(g)
    np.testing.assert_allclose(our_value, oracle_value, rtol=rtol, atol=atol)


def make_failure_list_pattern(
    g_iter: Iterable["nx.Graph"],
    our_fn: Callable[["nx.Graph"], float],
    oracle_fn: Callable[["nx.Graph"], float],
) -> list[tuple[int, float, str]]:
    failures: list[tuple[int, float, str]] = []
    for idx, g in enumerate(g_iter):
        try:
            our_value = float(our_fn(g))
            oracle_value = float(oracle_fn(g))
        except (RuntimeError, ValueError, ZeroDivisionError) as exc:
            failures.append((idx, float("inf"), f"raised={type(exc).__name__}: {exc!s}"))
            continue
        drift = abs(our_value - oracle_value)
        if drift > 0:
            descriptor = (
                f"n={g.number_of_nodes()} m={g.number_of_edges()} "
                f"ours={our_value!r} oracle={oracle_value!r}"
            )
            failures.append((idx, drift, descriptor))
    return failures


@given(g=_GRAPH_STRATEGY)
@settings(deadline=None, max_examples=20)
def test_harness_imports_and_strategy_yields_a_graph(g: "nx.Graph"):
    assert g.number_of_nodes() > 0, "graph_builder returned an empty graph"
    assert nx.is_connected(g), (
        f"strategy promised connected=True but got disconnected graph "
        f"(n={g.number_of_nodes()}, m={g.number_of_edges()})"
    )


__all__ = [
    "differential_check",
    "make_failure_list_pattern",
    "test_harness_imports_and_strategy_yields_a_graph",
]
