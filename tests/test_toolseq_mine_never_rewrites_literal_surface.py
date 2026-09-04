"""Structural + behavioral proof that the Signal-B tool-sequence miner never
rewrites a record's verbatim surface. The module legitimately READS
literal_surface (it is the loader) -- what must never happen is a WRITE back
onto it. An AST walk targeting assignment nodes, not a text grep, so line
numbers and surrounding prose can drift freely."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.lilli.cycle.proc_mine import MIN_DISTINCT_SESSIONS, PAIR_COUNT_FLOOR
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import SCHEMA_VERSION_CURRENT, MemoryRecord

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLSEQ_MODULE = _REPO_ROOT / "src" / "iai_mcp" / "lilli" / "cycle" / "toolseq_mine.py"

_BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _literal_surface_write_targets(node: ast.AST) -> list[ast.Attribute]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute)
        and n.attr == "literal_surface"
        and isinstance(n.ctx, ast.Store)
    ]


def test_toolseq_mine_never_assigns_literal_surface():
    source = _TOOLSEQ_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_TOOLSEQ_MODULE))
    hits = _literal_surface_write_targets(tree)
    assert hits == [], (
        f"toolseq_mine.py must never WRITE literal_surface -- it is a "
        f"read-only trailer parser; found {len(hits)} assignment target(s) "
        f"at line(s) {[h.lineno for h in hits]}"
    )


_DECOY_SOURCE = """
def poison_a_turn(record):
    record.literal_surface = "corrupted by the miner"
"""


def test_guard_fires_on_decoy():
    tree = ast.parse(_DECOY_SOURCE, filename="<decoy>")
    hits = _literal_surface_write_targets(tree)
    assert hits != [], (
        "the guard must fire on a decoy that assigns onto "
        "record.literal_surface -- an empty hit list here means the scan "
        "has no teeth"
    )


# --- behavioral: byte-identity across a full step_proc_mine run ------------


def _fresh_store(tmp_path: Path) -> "tuple[MemoryStore, Path]":
    home = tmp_path / "operator-home"
    store_root = home / ".iai-mcp"
    store = MemoryStore(path=store_root)
    return store, home


def _uneven_splits(total: int, n: int) -> list[int]:
    base, rem = divmod(total, n)
    return [base + 1 if i < rem else base for i in range(n)]


def _insert_assistant_turn(
    store: MemoryStore, session_id: str, tools: "tuple[str, str]", created_at: datetime,
) -> UUID:
    text = (
        "the assistant finished a turn of ambient conversational work"
        f"\n[tools: {', '.join(tools)}]"
    )
    rec_id = uuid4()
    rec = MemoryRecord(
        id=rec_id,
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.0] * store._embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[
            {
                "ts": created_at.isoformat(),
                "cue": "test",
                "session_id": session_id,
                "role": "assistant",
            }
        ],
        created_at=created_at,
        updated_at=created_at,
        tags=["capture", "role:assistant"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    store.insert(rec)
    return rec_id


def _run_step(store: MemoryStore, tmp_path: Path, label: str) -> dict:
    lifecycle_path = tmp_path / f"lifecycle-{label}.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)
    done, payload = pipeline._step_proc_mine(interrupt_check=None)
    assert done is True
    return payload


def test_verbatim_bytes_unchanged_across_step_proc_mine(tmp_path, monkeypatch):
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    store, _home = _fresh_store(tmp_path)

    pair = ("Bash", "Agent")
    splits = _uneven_splits(PAIR_COUNT_FLOOR, MIN_DISTINCT_SESSIONS)
    ts = _BASE_TS
    record_ids = []
    for i, n in enumerate(splits):
        session_id = f"sess-{i}"
        for _j in range(n):
            record_ids.append(_insert_assistant_turn(store, session_id, pair, ts))
            ts = ts + timedelta(seconds=1)
    flush_record_buffer(store)

    before = {rid: store.get(rid).literal_surface for rid in record_ids}
    assert all(text.endswith("]") for text in before.values())

    _run_step(store, tmp_path, "verbatim-guard")

    after = {rid: store.get(rid).literal_surface for rid in record_ids}
    assert after == before
    store.close()
