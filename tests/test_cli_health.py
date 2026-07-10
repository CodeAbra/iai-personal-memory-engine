from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4



def test_cli_health_no_events(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_health

    args = argparse.Namespace()
    exit_code = cmd_health(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "no events" in out.lower()


def test_cli_health_reports_last_event(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_health
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    write_event(
        store,
        kind="llm_health",
        data={"status": "ok"},
        severity="info",
    )
    args = argparse.Namespace()
    exit_code = cmd_health(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "llm_health" in out
    assert "info" in out


def test_cli_migrate_dry_run(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_migrate
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_LEGACY, EMBED_DIM

    store = MemoryStore()
    for i in range(3):
        r = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"Legacy v1 record number {i} with words to detect.",
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
            provenance=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=[],
            language="en",
            schema_version=SCHEMA_VERSION_LEGACY,
        )
        r.language = ""
        store.insert(r)

    args = argparse.Namespace(from_=1, to=2, dry_run=True, verbose=False)
    exit_code = cmd_migrate(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "would migrate" in out.lower()

    for r in store.all_records():
        if not r.pinned or r.id == uuid4():
            continue
    v1_count = sum(1 for r in store.all_records() if r.schema_version == 1)
    assert v1_count >= 3


def test_cli_entrypoint_exists():
    from iai_mcp.cli import main

    assert callable(main)
