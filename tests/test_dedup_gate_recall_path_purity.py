"""RED/structural test proving the dedup gate never sits on the awake recall
critical path.

Two guards:
  1. Behavioural: monkeypatching the write-hot-path exact-authority helper
     (``MemoryStore._exact_near_dup_target``) to raise must leave
     ``memory_recall`` dispatch unaffected. The public confirm helper
     (``MemoryStore._exact_confirm_near_dup``) is also monkeypatched to
     raise as a coexisting assertion so a future refactor that re-routes
     recall through either symbol trips the guard.
  2. Structural: pipeline.py and core/__init__.py's recall dispatch carry
     ZERO references to the write-hot-path symbols, the write-time gate, or
     the capture-time threshold constant. Standing regression guard that
     the write-side authority never leaks onto the read path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import core
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

# The load-bearing write-hot-path symbol after the redundant-rescan removal:
# _exact_near_dup_target is the sole path from the gate to the exact
# authority. _exact_confirm_near_dup is a public helper kept as a coexisting
# armed sentinel — a future refactor that re-routes recall through either
# must trip this guard.
_EXACT_NEAR_DUP_TARGET_SYMBOL = "_exact_near_dup_target"
_EXACT_CONFIRM_SYMBOL = "_exact_confirm_near_dup"
_EXACT_SCAN_SYMBOL = "_exact_scan_full_corpus"


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _seed_one_record(store: MemoryStore, text: str = "reference content for probe") -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(rec)


# ---------------------------------------------------------------------------
# Behavioural guard: monkeypatch-to-raise leaves recall unaffected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recall_path_unaffected_by_exact_authority_raising(driver, tmp_path, monkeypatch):
    """memory_recall dispatch never touches the write-hot-path exact-authority
    helper, on either driver.

    Load-bearing target: ``_exact_near_dup_target`` — the sole write-hot-path
    call into the exact authority after the redundant-rescan removal.
    Coexisting armed sentinel: ``_exact_confirm_near_dup`` — a public helper
    kept armed so a future refactor that re-routes recall through either
    symbol trips this guard.
    """
    _select_driver(driver, monkeypatch)
    _monkeypatch_env(monkeypatch, tmp_path)

    if not hasattr(MemoryStore, _EXACT_NEAR_DUP_TARGET_SYMBOL):
        pytest.skip(
            f"MemoryStore.{_EXACT_NEAR_DUP_TARGET_SYMBOL} not yet added — "
            f"behavioural guard arms once the symbol exists"
        )

    def _boom_target(*args, **kwargs):
        raise AssertionError(
            f"{_EXACT_NEAR_DUP_TARGET_SYMBOL} called during memory_recall dispatch"
        )

    def _boom_confirm(*args, **kwargs):
        raise AssertionError(
            f"{_EXACT_CONFIRM_SYMBOL} called during memory_recall dispatch"
        )

    monkeypatch.setattr(MemoryStore, _EXACT_NEAR_DUP_TARGET_SYMBOL, _boom_target)
    if hasattr(MemoryStore, _EXACT_CONFIRM_SYMBOL):
        monkeypatch.setattr(MemoryStore, _EXACT_CONFIRM_SYMBOL, _boom_confirm)

    store_path = tmp_path / f"driver-{driver}-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_path)
    try:
        _seed_one_record(store, "reference content for dedup-gate recall no-op probe")

        resp = core.dispatch(store, "memory_recall", {
            "cue": "reference content",
            "session_id": f"sess-dedup-noop-{driver}",
            "cue_embedding": [0.1] * EMBED_DIM,
        })
        assert "error" not in resp, (
            f"[driver={driver}] recall dispatch errored: {resp.get('error')}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Structural guard: zero write-hot-path / write-time-gate references on the
# recall dispatch path. GREEN today; standing regression guard.
# ---------------------------------------------------------------------------

def test_recall_dispatch_files_carry_zero_dedup_gate_references():
    """pipeline.py and core/__init__.py's recall dispatch contain ZERO
    references to the write-hot-path exact-authority helpers, the
    write-time gate, or the capture-time threshold constant.

    GREEN today; this is the standing structural guard that the dedup gate —
    present or future — never leaks onto the awake recall critical path.
    """
    import iai_mcp.pipeline as pipeline_mod
    import iai_mcp.core as core_mod

    pipeline_src = Path(pipeline_mod.__file__).read_text(encoding="utf-8")
    core_src = Path(core_mod.__file__).read_text(encoding="utf-8")

    forbidden_symbols = [
        _EXACT_CONFIRM_SYMBOL,
        _EXACT_NEAR_DUP_TARGET_SYMBOL,
        _EXACT_SCAN_SYMBOL,
        "_pattern_separation_gate_with_hits",
        "DEDUP_COS_THRESHOLD",
    ]

    for symbol in forbidden_symbols:
        assert symbol not in pipeline_src, (
            f"pipeline.py must carry ZERO references to {symbol!r} — the "
            f"dedup gate is write-side only, never on the recall path"
        )
        assert symbol not in core_src, (
            f"core/__init__.py must carry ZERO references to {symbol!r} — "
            f"the dedup gate is write-side only, never on the recall path"
        )
