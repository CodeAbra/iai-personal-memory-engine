"""Tripwire: foresight stays off the awake recall path, structurally and
behaviorally — the same guard family working_tier already has. Foresight
owns exactly the machinery a future author would be tempted to reuse inside
memory_recall (exact-authority confirm, confidence floor, corrector probe);
wiring any of it in would couple recall to the working tier transitively
and put anticipation on the awake path."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import core
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

_SRC = Path(__file__).parent.parent / "src" / "iai_mcp"

_RECALL_DISPATCH_MODULES = [
    "retrieve.py",
    "pipeline.py",
    "semantic_recall.py",
    "core/__init__.py",
    "store/_store.py",
]


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def test_recall_dispatch_modules_never_import_foresight():
    pattern = re.compile(r"^\s*(from|import)\s+iai_mcp\.foresight\b|^\s*from\s+iai_mcp\s+import\s+.*\bforesight\b", re.M)
    offenders = []
    for rel in _RECALL_DISPATCH_MODULES:
        text = (_SRC / rel).read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(rel)
    assert not offenders, (
        f"recall-dispatch modules import foresight: {offenders} — "
        f"anticipation must never sit on the awake recall path"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recall_path_no_foresight_calls(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))

    from iai_mcp import foresight as fs

    def _boom(*args, **kwargs):
        raise AssertionError("foresight function called during memory_recall dispatch")

    monkeypatch.setattr(fs, "refresh_pack", _boom)
    monkeypatch.setattr(fs, "_exact_scores", _boom)
    monkeypatch.setattr(fs, "_blended_cue", _boom)
    monkeypatch.setattr(fs, "_correctors", _boom)
    monkeypatch.setattr(fs, "_current_correctors", _boom)

    store_path = tmp_path / f"driver-{driver}-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_path)
    try:
        now = datetime.now(timezone.utc)
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface="reference content for foresight recall no-op probe",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "seed", "role": "user"}],
            created_at=now,
            updated_at=now,
            tags=["role:user"],
            language="en",
            s5_trust_score=0.5,
            profile_modulation_gain={},
        )
        store.insert(rec)

        resp = core.dispatch(store, "memory_recall", {
            "cue": "reference content",
            "session_id": f"sess-foresight-noop-{driver}",
            "cue_embedding": [0.1] * EMBED_DIM,
        })
        assert "error" not in resp, (
            f"[driver={driver}] recall dispatch errored: {resp.get('error')}"
        )
    finally:
        store.close()
