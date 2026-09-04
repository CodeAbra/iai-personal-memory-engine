"""Structural guards proving the procedural-mine sleep step stays sleep-only.

Two guards:
  1. The awake recall dispatch surface (pipeline.py, core/__init__.py,
     retrieve.py, session.py) carries zero references to the procedural-mine
     step symbols — a forward-looking regression guard against a future
     refactor pulling chunk/mine symbols onto the latency-sensitive recall
     path.
  2. The step's own source carries no reference to the dead
     ``temporal_sequence`` edge type.

Pure text scan — no store, no embedder.
"""

from __future__ import annotations

from pathlib import Path

_FORBIDDEN_PROC_MINE_SYMBOLS = (
    "proc_mine",
    "PROC_MINE",
    "persist_proc_chunk",
    "mine_cofired_pairs",
)


def test_awake_recall_surface_has_no_proc_mine_symbols():
    import iai_mcp.core as core_mod
    import iai_mcp.pipeline as pipeline_mod
    import iai_mcp.retrieve as retrieve_mod
    import iai_mcp.session as session_mod

    modules = (pipeline_mod, core_mod, retrieve_mod, session_mod)
    for mod in modules:
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_PROC_MINE_SYMBOLS:
            assert symbol not in src, (
                f"{mod.__name__} ({mod.__file__}) references forbidden "
                f"symbol {symbol!r} — PROC_MINE must stay sleep-only"
            )


def test_proc_mine_step_never_references_temporal_sequence():
    from iai_mcp.lilli.cycle.sleep_pipeline import _proc_mine

    src = Path(_proc_mine.__file__).read_text(encoding="utf-8")
    assert "temporal_sequence" not in src
