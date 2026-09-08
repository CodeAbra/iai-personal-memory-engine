"""`remove directive: <id>` -- the symmetric transcript-drain marker.

`parse_directive_remove_marker` is pure-lexical (mirrors `is_directive_marker`).
Firing lives in `deferred_drain_worker.py`, the sole `directive_marker_allowed=True`
site: it retires through the SAME shared seam (`directive_ops.retire_directive`)
that the `iai directive remove` CLI uses, fires only on a genuinely-new (`status == "inserted"`)
`role == "user"` drained turn, and is idempotent for free because a re-drained
duplicate dedups before this branch runs again.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.deferred_drain_worker import _drain_files
from iai_mcp.directive_marker import parse_directive_remove_marker
from iai_mcp.migrate._blob_quarantine import _MARKER_GUARD_PREFIXES
from iai_mcp.store import MemoryStore


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


def _write_backlog_file(
    path: Path, session_id: str, events: list[dict],
) -> None:
    """Reproduce the verified live capture-file schema: header line 0 (four
    keys, no ts/source_uuid), event lines with the six production keys.
    Plain JSON, no encryption -- `_decode_spool_line` passes it through.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "cwd": "/tmp/work",
        "deferred_at": "2026-01-01T00:00:00+00:00",
        "session_id": session_id,
        "version": 1,
    }
    lines = [json.dumps(header)]
    for ev in events:
        lines.append(json.dumps(ev))
    path.write_text("\n".join(lines) + "\n")


def _mint(store: MemoryStore, text: str = "standing directive: reply in English") -> UUID:
    result = capture_turn(
        store=store, cue="c", text=text, directive=True, session_id="s0", role="user",
    )
    assert result["status"] == "inserted", result
    return UUID(result["record_id"])


def _is_live(store: MemoryStore, rid: UUID) -> bool:
    from iai_mcp.store import flush_record_buffer

    flush_record_buffer(store)
    return rid in {r.id for r in store.iter_records(where="directive = 1 AND tombstoned_at IS NULL")}


# --- parse_directive_remove_marker: pure-lexical unit cases -----------------


def test_parse_anchored_prefix_returns_token():
    assert parse_directive_remove_marker("remove directive: aabbccdd") == "aabbccdd"
    assert parse_directive_remove_marker("Remove Directive:   aabbccdd") == "aabbccdd"
    assert parse_directive_remove_marker("  remove directive:aabbccdd") == "aabbccdd"


def test_parse_rejects_mid_sentence_occurrence():
    assert parse_directive_remove_marker("please remove directive: aabbccdd") is None


def test_parse_rejects_empty_non_str_whitespace():
    assert parse_directive_remove_marker(None) is None
    assert parse_directive_remove_marker("") is None
    assert parse_directive_remove_marker("   ") is None
    assert parse_directive_remove_marker(123) is None  # type: ignore[arg-type]


def test_parse_rejects_prefix_with_no_token():
    assert parse_directive_remove_marker("remove directive:") is None
    assert parse_directive_remove_marker("remove directive:   ") is None


# --- drain path: genuine role=user turn retires the directive ---------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_drain_retires_directive_on_genuine_user_turn(driver, store, monkeypatch, tmp_path):
    _select_driver(driver, monkeypatch)
    rid = _mint(store)
    assert _is_live(store, rid)
    short_id = rid.hex[:8]

    backlog = tmp_path / "backlog" / "remove.jsonl"
    _write_backlog_file(
        backlog, "sess-remove",
        [{
            "cue": f"remove directive: {short_id}",
            "role": "user",
            "source_uuid": "sess-remove-0",
            "text": f"remove directive: {short_id}",
            "tier": "episodic",
            "ts": "2026-01-01T00:00:00+00:00",
        }],
    )

    tally = _drain_files(store, [str(backlog)])
    assert tally["inserted"] == 1
    assert not _is_live(store, rid)

    # The drained turn record itself is stored as a normal non-directive record.
    from iai_mcp.store import flush_record_buffer
    flush_record_buffer(store)
    turn_recs = [
        r for r in store.iter_records(where="tombstoned_at IS NULL")
        if short_id in r.literal_surface
    ]
    assert turn_recs, "the remove-marker turn itself must be captured"
    assert all(r.directive is False for r in turn_recs)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_drain_idempotent_second_drain_does_not_reinsert_or_reretire(
    driver, store, monkeypatch, tmp_path
):
    _select_driver(driver, monkeypatch)
    rid = _mint(store)
    short_id = rid.hex[:8]

    event = {
        "cue": f"remove directive: {short_id}",
        "role": "user",
        "source_uuid": "sess-remove-idem-0",
        "text": f"remove directive: {short_id}",
        "tier": "episodic",
        "ts": "2026-01-01T00:00:00+00:00",
    }

    backlog1 = tmp_path / "backlog" / "remove1.jsonl"
    _write_backlog_file(backlog1, "sess-remove-idem", [event])
    first = _drain_files(store, [str(backlog1)])
    assert first["inserted"] == 1
    assert not _is_live(store, rid)

    # _drain_files unlinks the file after draining -- re-create an identical
    # spool (same session_id/source_uuid/ts/text) to actually exercise the
    # dedup branch on the second pass, not a vacuous "nothing to drain" call.
    backlog2 = tmp_path / "backlog" / "remove2.jsonl"
    _write_backlog_file(backlog2, "sess-remove-idem", [event])
    second = _drain_files(store, [str(backlog2)])

    assert second["inserted"] == 0
    assert second["reinforced"] + second["skipped_existing"] == 1
    assert not _is_live(store, rid)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_assistant_role_never_retires(driver, store, monkeypatch, tmp_path):
    _select_driver(driver, monkeypatch)
    rid = _mint(store)
    short_id = rid.hex[:8]

    backlog = tmp_path / "backlog" / "assistant.jsonl"
    _write_backlog_file(
        backlog, "sess-assistant",
        [{
            "cue": f"remove directive: {short_id}",
            "role": "assistant",
            "source_uuid": "sess-assistant-0",
            "text": f"remove directive: {short_id}",
            "tier": "episodic",
            "ts": "2026-01-01T00:00:00+00:00",
        }],
    )

    tally = _drain_files(store, [str(backlog)])
    # The assistant turn was genuinely inserted (not skipped for an
    # unrelated reason such as dedup or too-short) -- proving the role
    # guard, not an accidental skip, is what kept the directive live.
    assert tally["inserted"] == 1
    assert _is_live(store, rid)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_rpc_style_capture_turn_call_never_retires(driver, store, monkeypatch):
    """capture_turn on its own (the memory_capture RPC shape,
    directive_marker_allowed=False) never fires REMOVE -- the logic lives
    entirely in deferred_drain_worker.py, never in capture_turn."""
    _select_driver(driver, monkeypatch)
    rid = _mint(store)
    short_id = rid.hex[:8]

    result = capture_turn(
        store=store, cue="c", text=f"remove directive: {short_id}",
        session_id="s1", role="user", directive_marker_allowed=False,
    )
    assert result["status"] == "inserted", result
    assert _is_live(store, rid)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_blob_guarded_text_never_retires(driver, store, monkeypatch, tmp_path):
    _select_driver(driver, monkeypatch)
    rid = _mint(store)
    short_id = rid.hex[:8]
    blob_prefix = _MARKER_GUARD_PREFIXES[0]
    text = f"{blob_prefix}remove directive: {short_id}"

    backlog = tmp_path / "backlog" / "blob.jsonl"
    _write_backlog_file(
        backlog, "sess-blob",
        [{
            "cue": text,
            "role": "user",
            "source_uuid": "sess-blob-0",
            "text": text,
            "tier": "episodic",
            "ts": "2026-01-01T00:00:00+00:00",
        }],
    )

    tally = _drain_files(store, [str(backlog)])
    assert tally["inserted"] == 1
    assert _is_live(store, rid)
