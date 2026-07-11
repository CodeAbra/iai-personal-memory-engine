"""Smoke wrapper for the RSS attribution harness.

Drives the harness at N=200 records × 1 cycle to gate the import shape, the
corpus builder, the one-cycle end-to-end path (including the rendered
RCA-FINDINGS.md), and the hermeticity contract. The heavy N=22000 × 3-cycle
run is invoked manually via ``python tests/rss_rca/harness.py``.

Single-file pytest target: ``pytest tests/test_rss_rca_smoke.py -x``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.rss_rca import corpus, harness, surfaces
from tests.rss_rca.harness import _real_store_signature  # type: ignore[attr-defined]


def test_harness_imports() -> None:
    """The module surface MUST be importable independent of order."""
    assert hasattr(harness, "run")
    assert hasattr(corpus, "build_corpus")
    assert hasattr(corpus, "make_record")
    assert hasattr(surfaces, "SURFACES")
    assert hasattr(surfaces, "classify_all")
    assert hasattr(surfaces, "render_findings_md")
    assert len(surfaces.SURFACES) == 9


def test_corpus_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_corpus(store, n=200)`` inserts 200 records into a tmp store."""
    # Hermetic env: HOME and IAI_MCP_STORE under the pytest tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))
    store_root = tmp_path / ".iai-mcp" / "hippo"
    store_root.mkdir(parents=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-rss-rca-smoke")
    monkeypatch.setenv(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring",
    )

    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=store_root)
    try:
        inserted = corpus.build_corpus(
            store, n=200, seed=42, varied_tags=True,
        )
        assert inserted == 200
        assert store.active_records_count() >= 200
    finally:
        try:
            store.close()
        except Exception:
            pass


def test_one_cycle_runs(tmp_path: Path) -> None:
    """``harness.run(n=200, cycles=1)`` writes a populated RCA-FINDINGS.md."""
    out = tmp_path / "out"
    out.mkdir()

    findings = harness.run(n=200, cycles=1, out=out, seed=42,
                           deferred_events_per_cycle=50)
    assert findings.exists(), f"findings file missing at {findings}"

    body = findings.read_text(encoding="utf-8")
    assert "## Executive Summary" in body
    assert "## Per-Surface Findings Table" in body
    assert "## Ranked Fix List" in body
    assert "## Threshold Suggestion" in body
    assert "## Sources of Per-Cycle Data" in body

    # All 9 surface rows appear in the per-surface table.
    for i in range(1, 10):
        assert f"| {i} |" in body, f"surface {i} missing from findings table"

    # rca-log.jsonl has exactly one cycle record.
    log = out / "rca-log.jsonl"
    assert log.exists(), f"rca-log.jsonl missing at {log}"
    cycle_records = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("kind") == "cycle":
            cycle_records.append(obj)
    assert len(cycle_records) == 1, (
        f"expected 1 cycle record, got {len(cycle_records)}"
    )


def test_hermeticity(tmp_path: Path) -> None:
    """The harness MUST NOT touch the real ``~/.iai-mcp/``."""
    out = tmp_path / "out"
    out.mkdir()
    sig_before = _real_store_signature()
    harness.run(n=200, cycles=1, out=out, seed=42,
                deferred_events_per_cycle=50)
    sig_after = _real_store_signature()
    assert sig_before == sig_after, (
        "harness must not modify the real ~/.iai-mcp/ store"
    )
