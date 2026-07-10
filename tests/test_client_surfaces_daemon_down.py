from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _seed_store_with_drained_turn(store_root: Path, text: str) -> None:
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(store_root)
    try:
        rng = np.random.RandomState(seed=77)
        vec = rng.randn(EMBED_DIM).tolist()
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=vec,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "h2-session", "role": "user"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["role:user"],
            language="en",
        )
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()


def _make_args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_cmd_capture_daemon_down_writes_direct_to_store(
    hermetic_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iai_mcp.cli as _cli_mod

    monkeypatch.setattr(_cli_mod, "_send_jsonrpc_request", lambda *a, **k: None)

    from iai_mcp.iai_cli import cmd_capture

    args = _make_args(
        text="h2 capture probe text",
        session_id="h2-cap-session",
    )
    rc = cmd_capture(args)

    assert rc == 0, (
        f"H2 cmd_capture with daemon down must return 0 (success), got {rc}; "
        "the direct-write fallback is not yet wired"
    )

    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        records = store.all_records()
        surfaces = [r.literal_surface or "" for r in records]
        assert any("h2 capture probe text" in s for s in surfaces), (
            "H2: captured turn not found in the tmp Hippo store after daemon-down cmd_capture"
        )
    finally:
        store.close()


def test_cmd_last_daemon_down_returns_store_backed(
    hermetic_store: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import iai_mcp.cli as _cli_mod

    drained_text = "h2 last drained store turn text"
    _seed_store_with_drained_turn(hermetic_store, drained_text)

    monkeypatch.setattr(_cli_mod, "_send_jsonrpc_request", lambda *a, **k: None)

    from iai_mcp.iai_cli import cmd_last

    args = _make_args(n=10, session=None)
    rc = cmd_last(args)

    assert rc == 0, (
        f"H2 cmd_last with daemon down must return 0, got {rc}"
    )

    captured = capsys.readouterr()
    assert drained_text in captured.out, (
        f"H2: drained store turn not in cmd_last stdout;\n"
        f"stdout={captured.out!r}\n"
        "The live-layer fallback cannot see drained turns — must hit the store."
    )


def test_cmd_recall_daemon_down_returns_store_backed_degraded(
    hermetic_store: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import iai_mcp.cli as _cli_mod
    import iai_mcp.embed as _embed_mod

    drained_text = "h2 recall store backed degraded probe text"
    _seed_store_with_drained_turn(hermetic_store, drained_text)

    monkeypatch.setattr(_cli_mod, "_send_jsonrpc_request", lambda *a, **k: None)

    def _no_construct_funnel(_store):
        raise RuntimeError("hermetic: no embedder construct in this degrade test")

    monkeypatch.setattr(_embed_mod, "embedder_for_store", _no_construct_funnel)

    import iai_mcp.iai_cli as _iai_cli_mod

    def _no_bank_subprocess(*args, **kwargs):
        raise AssertionError(
            "H2 cmd_recall must use the direct store degraded path, "
            "not the bank-recall subprocess"
        )

    monkeypatch.setattr(_iai_cli_mod.subprocess, "run", _no_bank_subprocess)

    from iai_mcp.iai_cli import cmd_recall

    args = _make_args(cue="h2 recall store backed degraded", limit=5)
    rc = cmd_recall(args)

    assert rc == 0, (
        f"H2 cmd_recall with daemon down must return 0 (store-backed degraded), got {rc}"
    )

    captured = capsys.readouterr()
    assert drained_text in captured.out, (
        f"H2: drained store turn not in cmd_recall stdout;\n"
        f"stdout={captured.out!r}\n"
        "The bank-recall subprocess cannot see this turn — must be store-backed."
    )
