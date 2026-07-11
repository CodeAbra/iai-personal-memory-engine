from __future__ import annotations

import argparse
from datetime import datetime, timezone
from uuid import uuid4



def _seed_pinned_l1(store, n=3):
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    now = datetime.now(timezone.utc)
    for i in range(n):
        rec = MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=f"Pinned fact {i}: high-detail context.",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.5,
            detail_level=5,
            pinned=True,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=True,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
        )
        store.insert(rec)


def test_stdout_contains_four_segments_in_fixed_order(tmp_path, monkeypatch, capsys):
    from iai_mcp import cli as cli_mod, profile as profile_mod
    from iai_mcp.core import _seed_l0_identity, dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    _seed_pinned_l1(store, 3)

    state = profile_mod.default_state()
    state["wake_depth"] = "standard"
    monkeypatch.setattr("iai_mcp.core._profile_state", state, raising=False)

    def _stub(method, params, **_kw):
        result = dispatch(store, method, params)
        if not result.get("l2"):
            result["l2"] = ["[community deadbeef] W:0/example community line"]
        if not result.get("rich_club"):
            result["rich_club"] = "W:0/n: rich-club hub line"
        return {"jsonrpc": "2.0", "id": 1, "result": result}

    monkeypatch.setattr(cli_mod, "_send_jsonrpc_request", _stub)

    rc = cli_mod.cmd_session_start(argparse.Namespace(session_id="abc12345"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "## Identity" in out, out
    assert "## Critical facts" in out, out
    assert "## Topic communities" in out, out
    assert "## Key memories" in out, out
    i0 = out.index("## Identity")
    i1 = out.index("## Critical facts")
    i2 = out.index("## Topic communities")
    i3 = out.index("## Key memories")
    assert i0 < i1 < i2 < i3, (i0, i1, i2, i3, out)
    for header in ("## Identity", "## Critical facts", "## Topic communities", "## Key memories"):
        h_idx = out.index(header)
        tail = out[h_idx + len(header):]
        assert tail.startswith("\n"), header
        body_end = tail.find("\n## ")
        body = tail[1:] if body_end == -1 else tail[1:body_end]
        assert body.strip() != "", f"empty body under {header}: {body!r}"
