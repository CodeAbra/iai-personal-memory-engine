"""Foresight against the labelled REAL-thread fixture (local-only, opt-in).

The fixture and the store copy carry the operator's real memory content and
are NEVER committed; absent fixture -> graceful skip. Heavy (copies the real
store, builds the ANN index), so it is opt-in via --runslow like the other
subprocess-heavy gates.

Gate rationale: first measured run scored pack_hit_rate=1.0 and
labelled_precision_floor=0.825 over 40 cues; the gate sits safely below so
only a real regression trips it, not measurement noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _local_eval_env() -> "tuple[str, str] | None":
    """The suite sandboxes HOME, so this LOCAL-ONLY gate activates only when
    both the fixture and a pre-copied store are handed in explicitly."""
    import os

    fixture = os.environ.get("IAI_MCP_EVAL_FIXTURE_PATH")
    store_copy = os.environ.get("IAI_MCP_FORESIGHT_EVAL_STORE")
    if not fixture or not Path(fixture).expanduser().exists():
        return None
    if not store_copy or not (Path(store_copy).expanduser() / "hippo").exists():
        return None
    return fixture, store_copy


@pytest.mark.slow
def test_foresight_pack_hit_rate_on_real_threads():
    if _local_eval_env() is None:
        pytest.skip(
            "local-only gate: set IAI_MCP_EVAL_FIXTURE_PATH and "
            "IAI_MCP_FORESIGHT_EVAL_STORE (a pre-made read-only store copy) "
            "to run against the operator's real threads"
        )
    from bench.foresight_real_eval import run_eval

    summary = run_eval()
    assert summary["cues"] >= 30
    assert summary["pack_hit_rate"] >= 0.90, summary
    assert summary["empty_rate"] <= 0.20, summary
    precision = summary["labelled_precision_floor"]
    assert precision is None or precision >= 0.60, summary
