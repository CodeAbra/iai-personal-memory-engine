"""The ambient in-session `memory_capture` RPC route enables the cosine
near-dup gate: paraphrased corrections of the same standing rule fold into
one reinforced, salience-raised record instead of accumulating as separate
rows. The never_merge/directive carve-out (capture.py's existing fold
branch) is preserved, and no other capture_turn caller's default behavior
changes.
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp import core as core_mod
from iai_mcp.capture import capture_turn
from iai_mcp.core import dispatch
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _seed_matching_neighbor(
    store: MemoryStore, text: str, *, never_merge: bool = False,
) -> MemoryRecord:
    """Insert a record whose embedding exactly matches what capture_turn
    will compute for `text` -- a deterministic stand-in for a paraphrase
    that lands at cos >= the dedup floor, without depending on
    embedder-specific paraphrase similarity."""
    from iai_mcp.embed import embedder_for_store

    embedding = list(embedder_for_store(store).embed(text))
    now = datetime.now(timezone.utc)
    neighbor = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="an earlier phrasing of the same standing correction",
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=never_merge,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["capture", "role:user"],
    )
    store.insert(neighbor)
    return neighbor


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_ambient_rpc_route_folds_paraphrase_into_reinforced_record(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)
    text = "the release gate is the sequential lilli pytest run, not bare pytest"
    neighbor = _seed_matching_neighbor(store, text)

    result = dispatch(
        store, "memory_capture",
        {
            "text": text, "session_id": "s-ambient-fold", "role": "user",
            "salience_level": "critical",
        },
    )

    assert result["status"] == "reinforced", result
    assert result["record_id"] == str(neighbor.id)
    refreshed = store.get(neighbor.id)
    assert refreshed is not None
    assert refreshed.salience_level == "critical", (
        "a folded paraphrase must raise the existing record's salience"
    )
    live = list(store.iter_records(where="tombstoned_at IS NULL"))
    assert len(live) == 1, (
        f"a paraphrased correction must fold into the ONE existing record, "
        f"not accumulate as a separate row: {live}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_ambient_rpc_route_never_folds_a_pinned_neighbor(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)
    text = "always keep the pinned formatting rule exactly as written"
    pinned = _seed_matching_neighbor(store, text, never_merge=True)

    result = dispatch(
        store, "memory_capture",
        {"text": text, "session_id": "s-ambient-pinned", "role": "user"},
    )

    assert result["status"] == "inserted", result
    assert result["record_id"] != str(pinned.id), (
        "a never_merge neighbour is a hard lock and must never absorb a fold"
    )
    live = list(store.iter_records(where="tombstoned_at IS NULL"))
    assert len(live) == 2, live


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_ambient_route_directive_never_folds_into_neighbor(driver, store, monkeypatch):
    """The memory_capture RPC is structurally incapable of minting a
    directive, so this exercises capture_turn directly with the same
    live_turn + near_dup_gate composition the ambient route now uses --
    proving the directive carve-out survives when both flags are set
    together."""
    _select_driver(driver, monkeypatch)
    text = "always answer in a formal tone from now on"
    neighbor = _seed_matching_neighbor(store, text)

    result = capture_turn(
        store=store, cue="c", text=text,
        directive=True, near_dup_gate=True, live_turn=True,
        session_id="s-ambient-directive", role="user",
    )

    assert result["status"] == "inserted", result
    assert result["record_id"] != str(neighbor.id)
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is True


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_non_ambient_capture_turn_default_unaffected(driver, store, monkeypatch):
    """A capture_turn caller that does not opt into near_dup_gate (replay,
    bulk ingest, transcript drain) must keep inserting its own row --
    capture_turn's near-dup gate is enabled only at the ambient RPC call
    site, never as a global default."""
    _select_driver(driver, monkeypatch)
    text = "the deploy checklist requires a green gate before cutting a release"
    neighbor = _seed_matching_neighbor(store, text)

    result = capture_turn(
        store=store, cue="c", text=text,
        session_id="s-non-ambient", role="user",
    )

    assert result["status"] == "inserted", result
    assert result["record_id"] != str(neighbor.id), (
        "capture_turn's default (near_dup_gate=False) must not fold a "
        "near-duplicate outside the ambient RPC route"
    )


def test_capture_turn_default_signature_unchanged():
    sig = inspect.signature(capture_turn)
    assert sig.parameters["near_dup_gate"].default is False, (
        "capture_turn's own default must stay False -- only the ambient "
        "RPC call site opts in explicitly"
    )


def test_near_dup_gate_true_enabled_only_at_the_ambient_rpc_call_site():
    tree = ast.parse(inspect.getsource(core_mod))
    near_dup_true_call_sites = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "near_dup_gate"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                near_dup_true_call_sites += 1
    assert near_dup_true_call_sites == 1, (
        "near_dup_gate=True must appear at exactly one keyword-argument call "
        "site (the ambient memory_capture RPC), never flipped as a global default"
    )
