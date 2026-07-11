"""Regression: the isolated daemon boot proof must compare the daemon's recall
against a STRUCTURE-INFORMED stdlib reference, not a structure-blind one.

The daemon serves recall with the runtime graph cache present (community
assignment -> structure-informed ranking). A reverse-built stdlib reference that
lacks the graph cache dispatches with an empty community assignment
("cold-structural-degrade") and ranks the k-boundary by raw cosine alone. Comparing
the two then diverges at the k-boundary for reasons unrelated to the storage
engine, spuriously failing the recall gate.

These tests pin the fidelity contract that fixes that:
  1. the reverse copy carries runtime_graph_cache.json when the source has one,
  2. the reference recall reports whether it was structure-informed,
  3. the verdict gate fails when the reference was cold-degraded, even if every
     per-cue hit set happens to match.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from bench import isolated_daemon_boot_proof as proof


def test_reverse_copy_copies_graph_cache_when_present(monkeypatch, tmp_path):
    """The reverse-copy stage must carry the runtime graph cache with the corpus,
    so the reference dispatches with the same community structure the daemon uses.

    The DB copy loop is stubbed out -- this test pins only the graph-cache-copy
    branch, which is the load-bearing fidelity fix and is pure filesystem work.
    """
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    (src_root / ".crypto.key").write_bytes(b"0" * 32)
    cache_bytes = b'{"cache_version":"x","key":[],"assignment":{}}'
    (src_root / "runtime_graph_cache.json").write_bytes(cache_bytes)

    # Stub the DB-heavy body: after the key + graph-cache copy, short-circuit.
    class _StopAfterCopies(RuntimeError):
        pass

    def _boom(*_a, **_k):
        raise _StopAfterCopies

    # HippoDB is imported lazily inside the worker; block the first DB open so
    # the worker aborts right after the filesystem copies.
    import iai_mcp.hippo as hippo

    monkeypatch.setattr(hippo, "HippoDB", _boom, raising=True)

    with pytest.raises(_StopAfterCopies):
        proof._worker_reverse_copy(
            {"src_root": str(src_root), "dst_root": str(dst_root)}
        )

    copied = dst_root / "runtime_graph_cache.json"
    assert copied.exists(), "reverse copy must carry runtime_graph_cache.json"
    assert copied.read_bytes() == cache_bytes


def test_reverse_copy_no_graph_cache_is_tolerated(monkeypatch, tmp_path):
    """A source without a graph cache must not crash the copy -- the copy is
    best-effort (a fresh corpus that never built a cache is still valid)."""
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()
    (src_root / ".crypto.key").write_bytes(b"0" * 32)

    class _StopAfterCopies(RuntimeError):
        pass

    import iai_mcp.hippo as hippo

    monkeypatch.setattr(
        hippo, "HippoDB", lambda *a, **k: (_ for _ in ()).throw(_StopAfterCopies())
    )

    with pytest.raises(_StopAfterCopies):
        proof._worker_reverse_copy(
            {"src_root": str(src_root), "dst_root": str(dst_root)}
        )
    assert not (dst_root / "runtime_graph_cache.json").exists()


def _synthetic_recalls(ids_per_cue):
    return [
        {
            "index": i,
            "cue": f"cue{i}",
            "elapsed_ms": 1.0,
            "top10_ids": ids,
            "well_formed": True,
        }
        for i, ids in enumerate(ids_per_cue)
    ]


def _stub_prod_evidence(monkeypatch):
    ev = {
        "brain_sqlite3": None,
        "daemon_state_json": None,
        "wake_signal_exists": False,
        "independent_prod_daemon_pid": None,
        "independent_prod_daemon_alive": False,
    }
    monkeypatch.setattr(proof, "_prod_evidence", lambda: dict(ev))


def test_verdict_fails_when_reference_is_cold_degraded(monkeypatch):
    """Even with IDENTICAL per-cue hit sets, the recall comparison is not a valid
    truth bar if the reference was structure-blind. The gate must catch it."""
    _stub_prod_evidence(monkeypatch)
    ids = [[f"id{i}-{j}" for j in range(10)] for i in range(3)]
    daemon_cold = {"socket_up_sec": 1.0, "recalls": _synthetic_recalls(ids)}
    ref_recall = {
        "recalls": _synthetic_recalls(ids),  # identical hit sets
        "reference_structure_informed": False,  # but reference was cold-degraded
    }
    warm = {"socket_up_sec": 1.0, "well_formed": True, "hit_count": 10}
    smoke = {"smoke_pass": True, "llm_free": True, "sidecar_re_persisted": True}

    verdict = proof.stage_verdict(
        {}, daemon_cold, ref_recall, warm, smoke, {"ok": True}
    )
    assert verdict["gates"]["reference_structure_informed"] is False
    assert verdict["verdict"] == "FAIL"


def test_verdict_passes_when_reference_is_structure_informed(monkeypatch):
    _stub_prod_evidence(monkeypatch)
    ids = [[f"id{i}-{j}" for j in range(10)] for i in range(3)]
    daemon_cold = {"socket_up_sec": 1.0, "recalls": _synthetic_recalls(ids)}
    # first-recall steady band + latency gates would fail on real numbers, but
    # elapsed_ms=1.0 keeps this test focused on the fidelity gate specifically.
    ref_recall = {
        "recalls": _synthetic_recalls(ids),
        "reference_structure_informed": True,
    }
    warm = {"socket_up_sec": 1.0, "well_formed": True, "hit_count": 10}
    smoke = {"smoke_pass": True, "llm_free": True, "sidecar_re_persisted": True}

    verdict = proof.stage_verdict(
        {}, daemon_cold, ref_recall, warm, smoke, {"ok": True}
    )
    assert verdict["gates"]["reference_structure_informed"] is True
    assert verdict["gates"]["recall_at_10_mean_ge_0_98"] is True
