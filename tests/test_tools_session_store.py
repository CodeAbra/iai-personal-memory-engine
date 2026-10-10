"""Query-view filters (substring/category/time), structural per-session
privacy, and the brain-import-boundary contract for the query path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_BRAIN_MODULES = (
    "iai_mcp.capture",
    "iai_mcp.hippo",
    "iai_mcp.lilli",
    "iai_mcp_native.embed",
    "iai_mcp_native",
)


def _write_raw_events(sessions_dir: Path, session_id: str, rows: list[dict]) -> None:
    """Writes pre-built event rows directly to the JSONL file, bypassing
    capture's wall-clock timestamp so tests can control `ts` precisely.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{session_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _event(seq: int, session_id: str, ts: str, category: str, data: str) -> dict:
    return {
        "seq": seq,
        "session_id": session_id,
        "ts": ts,
        "category": category,
        "type": "Edit",
        "data": data,
        "priority": 2,
        "source_hook": "PostToolUse",
        "data_hash": "00000000",
    }


def test_filters_return_expected_rows_in_timeline_order(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path))

    from iai_mcp.tools import session

    rows = [
        _event(0, "sess-filters", "2026-01-01T00:00:00Z", "file", "src/Alpha.py"),
        _event(1, "sess-filters", "2026-01-02T00:00:00Z", "error", "TypeError: boom"),
        _event(2, "sess-filters", "2026-01-03T00:00:00Z", "file", "src/beta.py"),
    ]
    _write_raw_events(tmp_path / "sessions", "sess-filters", rows)

    all_events = session.query_events("sess-filters")
    assert len(all_events) == 3
    # Timeline (append) order.
    assert [e["data"] for e in all_events] == [
        "src/Alpha.py",
        "TypeError: boom",
        "src/beta.py",
    ]

    # Case-insensitive substring match.
    by_substring = session.query_events("sess-filters", substring="ALPHA")
    assert len(by_substring) == 1
    assert by_substring[0]["data"] == "src/Alpha.py"

    # Exact category match, timeline order preserved across two hits.
    by_category = session.query_events("sess-filters", category="file")
    assert [e["data"] for e in by_category] == ["src/Alpha.py", "src/beta.py"]

    # Time range: bound to just the middle event's timestamp.
    by_time = session.query_events(
        "sess-filters",
        since="2026-01-02T00:00:00Z",
        until="2026-01-02T23:59:59Z",
    )
    assert len(by_time) == 1
    assert by_time[0]["data"] == "TypeError: boom"


def test_cross_session_query_never_returns_the_other_sessions_events(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path))

    from iai_mcp.tools import session

    session.capture_event(
        {
            "tool_name": "Edit",
            "file_path": "a-only.py",
            "session_id": "sess-a",
            "source_hook": "PostToolUse",
        }
    )
    session.capture_event(
        {
            "tool_name": "Edit",
            "file_path": "b-only.py",
            "session_id": "sess-b",
            "source_hook": "PostToolUse",
        }
    )

    a_events = session.query_events("sess-a")
    assert len(a_events) == 1
    assert all(e["data"] != "b-only.py" for e in a_events)

    b_events = session.query_events("sess-b")
    assert len(b_events) == 1
    assert all(e["data"] != "a-only.py" for e in b_events)

    # An unfiltered query for A still cannot surface B's events, and vice versa.
    a_broad = session.query_events("sess-a", substring="")
    b_broad = session.query_events("sess-b", substring="")
    assert not any(e["data"] == "b-only.py" for e in a_broad)
    assert not any(e["data"] == "a-only.py" for e in b_broad)


def test_traversal_shaped_session_id_never_escapes_the_sessions_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path))

    from iai_mcp.tools import session

    # A valid id always resolves under `{root}/sessions/`.
    session.capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/valid.py",
            "session_id": "sess-valid",
            "source_hook": "PostToolUse",
        }
    )
    valid_path = tmp_path / "sessions" / "sess-valid.jsonl"
    assert valid_path.exists()
    assert valid_path.parent == tmp_path / "sessions"

    # A traversal-shaped id is rejected outright: no file is created
    # anywhere, inside or outside the sessions directory.
    session.capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/evil.py",
            "session_id": "../../etc/passwd",
            "source_hook": "PostToolUse",
        }
    )
    assert list((tmp_path / "sessions").iterdir()) == [valid_path]
    assert session.query_events("../../etc/passwd") == []


def test_query_path_imports_no_brain_module(tmp_path):
    script = (
        "import sys\n"
        "import os\n"
        f"os.environ['IAI_TOOLS_ROOT'] = {str(tmp_path)!r}\n"
        "import iai_mcp.tools.session as session\n"
        "session.query_events('sess-store-independence')\n"
        f"hit = [m for m in {_BRAIN_MODULES!r} if m in sys.modules]\n"
        "print(','.join(hit))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"brain module(s) loaded on the query path: {result.stdout.strip()!r} / {result.stderr}"
    )
