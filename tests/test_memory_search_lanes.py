"""Hybrid scoped search + the live watch loop — the context-mode replacement.

The lexical lane must find EXACT identifiers embeddings blur; the dispatch
tool merges both lanes under a hints-not-truth frame; the watch tick keeps
memory tracking a directory: changed files restudy, deleted files fade.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore, flush_record_buffer


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _seed(store, text, session="s"):
    result = capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=session, role="user",
    )
    flush_record_buffer(store)
    return result


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_lexical_lane_finds_exact_identifier(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    hit = _seed(store, "def _hippea_cascade_loop(store): return warm_set(store)")
    _seed(store, "Grocery list: oat milk, rye bread, and tomatoes.")

    results = store.lexical_search("_hippea_cascade_loop")
    assert results, f"[{driver}] exact identifier not found lexically"
    assert str(results[0][0].id) == hit["record_id"]

    # snake_case parts match too
    parts = store.lexical_search("cascade loop")
    assert any(str(r.id) == hit["record_id"] for r, _s in parts)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_lexical_index_refreshes_after_new_capture(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed(store, "First fact about the ANTIKYTHERA mechanism gearing.")
    assert store.lexical_search("antikythera")

    late = _seed(store, "Second fact mentions the ZODIAC dial of Antikythera.")
    results = store.lexical_search("zodiac")
    assert any(str(r.id) == late["record_id"] for r, _s in results), (
        f"[{driver}] index must rebuild when the corpus generation moves"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_memory_search_dispatch_merges_lanes_with_frame(driver, tmp_path, monkeypatch):
    from iai_mcp import core as _core

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed(store, "IAI_MCP_FORESIGHT_OFF disables the next-turn anticipation pack.")
    _seed(store, "Coral reefs host a quarter of all known marine species.")

    resp = _core.dispatch(store, "memory_search", {"query": "IAI_MCP_FORESIGHT_OFF"})
    assert resp["hits"], f"[{driver}] no hits for an exact env name"
    top = resp["hits"][0]
    assert "IAI_MCP_FORESIGHT_OFF" in top["surface"]
    assert top["lane"] in ("lexical", "both")
    assert "not ground truth" in resp["frame"]

    empty = _core.dispatch(store, "memory_search", {"query": "   "})
    assert empty["hits"] == []


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_watch_tick_studies_changes_and_fades_deletions(driver, tmp_path, monkeypatch):
    from iai_mcp.study import watch_scan_once

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / "store")
    proj = tmp_path / "proj"
    proj.mkdir()
    doc = proj / "notes.md"
    doc.write_text(
        "The hippocampus consolidates memories during sleep.", encoding="utf-8",
    )

    seen, ch = watch_scan_once(store, proj, {})
    assert ch["studied"] == 1 and "notes.md" in seen

    # untouched tree -> quiet tick
    seen, ch = watch_scan_once(store, proj, seen)
    assert ch["studied"] == 0 and ch["faded"] == 0

    # changed file -> restudy + supersede of the vanished chunk
    import os as _os

    doc.write_text("Completely different content about coral reefs now.", encoding="utf-8")
    _os.utime(doc, (doc.stat().st_atime, doc.stat().st_mtime + 5))
    seen, ch = watch_scan_once(store, proj, seen)
    assert ch["studied"] == 1
    assert ch["superseded_chunks"] >= 1, ch

    # deleted file -> its chunks fade (never deleted)
    trash = tmp_path / "trash"
    trash.mkdir()
    doc.rename(trash / "notes.md")
    seen, ch = watch_scan_once(store, proj, seen)
    assert ch["faded"] == 1
    assert ch["superseded_chunks"] >= 1
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT centrality FROM records WHERE tags_json LIKE ?"
            " AND tombstoned_at IS NULL",
            ('%"doc:notes.md"%',),
        ).fetchall()
    assert rows, "faded chunks must still exist"
    assert all(float(r[0] or 0) == 0.0 for r in rows)

def test_bm25_ranks_rare_and_concise_over_common_and_verbose():
    """Pure index-level check: a rare identifier outweighs a ubiquitous
    token, and a short precise doc outranks a long one repeating the term."""
    from iai_mcp.store._lexical_index import LexicalIndex

    idx = LexicalIndex()
    rows = [
        ("precise", "hippea_cascade warms the recall pool"),
        ("verbose",
         "cascade cascade cascade cascade cascade cascade cascade cascade "
         + "filler " * 200),
    ] + [(f"noise{i}", f"the store keeps records number {i} in the store") for i in range(20)]
    idx.build(rows, generation=1)

    hits = idx.query("hippea_cascade", k=5)
    assert hits and hits[0][0] == "precise"

    ranked = idx.query("cascade", k=5)
    ids = [rid for rid, _s in ranked]
    assert ids.index("precise") < ids.index("verbose"), (
        f"BM25 must prefer the concise doc over term-stuffing: {ranked}"
    )

    common = idx.query("store", k=25)
    rare = idx.query("hippea_cascade", k=25)
    assert rare[0][1] > common[0][1], "rare-token idf must dominate"


def test_cli_search_is_daemon_free(tmp_path, monkeypatch, capsys):
    """`iai search --json` is the wrapper's daemon-down fallback lane."""
    import argparse
    import json as _json

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "root"))
    store = MemoryStore(path=tmp_path / "root")
    _seed(store, "IAI_MCP_FORESIGHT_OFF disables the anticipation pack.")
    store.close()

    from iai_mcp.iai_cli import cmd_search

    rc = cmd_search(argparse.Namespace(
        query="IAI_MCP_FORESIGHT_OFF", limit=5, json=True,
    ))
    assert rc == 0
    resp = _json.loads(capsys.readouterr().out.strip())
    assert resp["hits"] and "IAI_MCP_FORESIGHT_OFF" in resp["hits"][0]["surface"]


def test_desktop_crate_spawns_dashboard_daemon_free(tmp_path):
    """The native desktop app's contract with the CLI: it spawns
    `iai brain --no-open --port N` and resolves the CLI through the
    install-time cache. Both halves must exist on the Python side."""
    crate = Path(__file__).resolve().parents[1] / "desktop/src-tauri/src/main.rs"
    src = crate.read_text(encoding="utf-8")
    assert '"brain"' in src and '"--no-open"' in src and '"--port"' in src
    assert ".iai-mcp/.cli-path" in src
    # the flag the app relies on must stay real
    from iai_mcp.iai_cli import _build_parser

    args = _build_parser().parse_args(["brain", "--no-open", "--port", "4477"])
    assert args.no_open is True and args.port == 4477
