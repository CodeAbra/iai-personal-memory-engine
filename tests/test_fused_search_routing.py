"""Routing test for `search_fused`: base lanes pass through the
`_normalize_hit_path` choke point before fusion, and the query-embed
subprocess fires ONLY when the gate reports the base fused top result is
weak. The native extension and the embed subprocess are both mocked --
this test needs no extension rebuild and no brain load.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from iai_mcp.tools import _normalize_hit_path, search_fused, search_fused_detailed


class _FakeNative:
    """Minimal stand-in for iai_mcp_tools_core: `search`/`search_symbol`/
    `search_xref` return fixed fixtures, `rrf_merge` mirrors the real RRF
    math closely enough to exercise the routing path, and `gate_is_weak`
    is scripted so a test can force either branch deterministically.
    """

    def __init__(self, whole, symbol, xref, weak, *, vector_error: "Exception | None" = None):
        self._whole = whole
        self._symbol = symbol
        self._xref = xref
        self._weak = weak
        self._vector_error = vector_error
        self.gate_calls: "list[tuple]" = []
        self.search_vector_calls = 0

    def search(self, root, query, k):
        return self._whole

    def search_symbol(self, root, query, k):
        return self._symbol

    def search_xref(self, root, xref_root, query, k):
        return self._xref

    def search_vector(self, root, query_vector, k):
        self.search_vector_calls += 1
        if self._vector_error is not None:
            raise self._vector_error
        return [("vec_hit.py", 0.9)]

    def rrf_merge(self, lanes):
        K = 60.0
        scores: "dict[str, float]" = {}
        order: "list[str]" = []
        for lane in lanes:
            for idx, path in enumerate(lane):
                if path not in scores:
                    order.append(path)
                    scores[path] = 0.0
                scores[path] += 1.0 / (K + idx + 1)
        ordered = sorted(order, key=lambda p: -scores[p])
        return [(p, scores[p]) for p in ordered]

    def gate_is_weak(self, base_fused, threshold):
        self.gate_calls.append((tuple(base_fused), threshold))
        return self._weak


def test_normalize_hit_path_canonicalizes_absolute_and_relative_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "src").mkdir()
    abs_path = repo / "src" / "mod.py"

    assert _normalize_hit_path(str(abs_path)) == "src/mod.py"
    assert _normalize_hit_path("src/mod.py") == "src/mod.py"


def test_search_fused_never_invokes_embed_subprocess_when_gate_does_not_fire(monkeypatch) -> None:
    fake = _FakeNative(
        whole=[("a.py", 5.0), ("b.py", 3.0)],
        symbol=[("a.py", 2.0)],
        xref=[("a.py", 5.0), ("c.py", 1.0)],
        weak=False,
    )
    monkeypatch.setitem(sys.modules, "iai_mcp_tools_core", fake)

    calls: "list[object]" = []

    def _fail_if_called(*args, **kwargs):
        calls.append(args)
        raise AssertionError("embed subprocess must not run when the gate does not fire")

    monkeypatch.setattr("subprocess.run", _fail_if_called)

    hits = search_fused("some query", k=5)

    assert calls == []
    assert fake.search_vector_calls == 0
    assert len(fake.gate_calls) == 1
    assert hits
    assert any(path == "a.py" for path, _ in hits)

    detailed = search_fused_detailed("some query", k=5)
    assert detailed["vector_lane"] == "not_attempted"
    assert detailed["reason"] is None


def test_search_fused_invokes_embed_subprocess_only_when_gate_fires(monkeypatch) -> None:
    fake = _FakeNative(
        whole=[("a.py", 5.0)],
        symbol=[],
        xref=[],
        weak=True,
    )
    monkeypatch.setitem(sys.modules, "iai_mcp_tools_core", fake)

    completed = types.SimpleNamespace(returncode=0, stdout="[0.1, 0.2, 0.3]\n", stderr="")
    calls: "list[list[str]]" = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return completed

    monkeypatch.setattr("subprocess.run", _fake_run)

    hits = search_fused("some query", k=5)

    assert len(calls) == 1
    assert "iai_mcp.tools.embed_worker" in calls[0]
    assert "query" in calls[0]
    assert fake.search_vector_calls == 1
    assert hits
    assert any(path == "vec_hit.py" for path, _ in hits)

    detailed = search_fused_detailed("some query", k=5)
    assert detailed["vector_lane"] == "ok"
    assert detailed["reason"] is None


def test_search_fused_degrades_to_base_lanes_when_vector_lane_errors(monkeypatch, capsys) -> None:
    """A stale vector store (dimension mismatch after an embedder change)
    must not take down the whole fused query -- the gate fires precisely
    when base results are weakest, so search_fused falls back to
    base-lanes-only and warns to stderr instead of propagating."""
    fake = _FakeNative(
        whole=[("a.py", 5.0)],
        symbol=[],
        xref=[],
        weak=True,
        vector_error=RuntimeError("tools vector index dimension mismatch: index built at dim 384, query is dim 8"),
    )
    monkeypatch.setitem(sys.modules, "iai_mcp_tools_core", fake)

    completed = types.SimpleNamespace(returncode=0, stdout="[0.1, 0.2, 0.3]\n", stderr="")
    monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: completed)

    hits = search_fused("some query", k=5)

    assert fake.search_vector_calls == 1
    assert hits
    assert any(path == "a.py" for path, _ in hits)
    assert "vec_hit.py" not in [p for p, _ in hits]
    assert "dimension mismatch" in capsys.readouterr().err

    detailed = search_fused_detailed("some query", k=5)
    assert detailed["vector_lane"] == "degraded"
    assert detailed["reason"] == "index_dimension_mismatch"
    assert any(path == "a.py" for path, _ in detailed["hits"])
    capsys.readouterr()  # drain the second call's stderr warning


def test_search_fused_degrades_when_embed_worker_itself_fails(monkeypatch, capsys) -> None:
    """A failing query-embed subprocess (not a stale vector store) must
    degrade the same way -- distinguishable by `reason`, never conflated
    with an index dimension mismatch."""
    fake = _FakeNative(
        whole=[("a.py", 5.0)],
        symbol=[],
        xref=[],
        weak=True,
    )
    monkeypatch.setitem(sys.modules, "iai_mcp_tools_core", fake)

    failed = types.SimpleNamespace(returncode=1, stdout="", stderr="embed worker crashed")
    monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: failed)

    hits = search_fused("some query", k=5)

    assert fake.search_vector_calls == 0
    assert hits
    assert any(path == "a.py" for path, _ in hits)

    detailed = search_fused_detailed("some query", k=5)
    assert detailed["vector_lane"] == "degraded"
    assert detailed["reason"] == "embed_unavailable"
    assert any(path == "a.py" for path, _ in detailed["hits"])
    assert "vector lane unavailable" in capsys.readouterr().err


def test_search_fused_xref_lane_is_derived_not_verbatim(monkeypatch) -> None:
    """The xref lane must be the SET DIFFERENCE beyond the whole-file base
    hits, not `search_xref`'s full output -- verified by checking the gate
    receives a base fusion where the injected-only row (`injected.py`) was
    fused as a genuinely separate lane contribution, not merely re-counted
    from the whole-file lane."""
    fake = _FakeNative(
        whole=[("a.py", 5.0), ("b.py", 3.0)],
        symbol=[],
        xref=[("a.py", 5.0), ("b.py", 3.0), ("injected.py", 1.0)],
        weak=False,
    )
    monkeypatch.setitem(sys.modules, "iai_mcp_tools_core", fake)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not embed")),
    )

    hits = search_fused("some query", k=10)
    paths = [p for p, _ in hits]

    assert "injected.py" in paths
    # a.py/b.py are stripped from the xref lane by derivation (they are
    # already in the whole-file base lane); injected.py becomes the xref
    # lane's OWN rank-1 entry, not a rank-3 entry copied from the raw
    # search_xref output -- confirms the derivation, not verbatim feed.
    injected_score = dict(hits)["injected.py"]
    assert abs(injected_score - 1.0 / (60.0 + 1.0)) < 1e-9
