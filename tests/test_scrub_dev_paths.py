"""Guard: developer home paths must not reach shipped source.

`scripts/scrub_dev_paths.py` walks the shipped roots and rejects any
`/Users/<name>/` or `/home/<name>/` segment. An absolute path from whoever
built the release leaks that account name and breaks on every other machine.
The rule is generic — no account name is special-cased.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# scripts/ lives at repo root, not on the src/ pythonpath. tests/conftest.py
# prepends the repo root to sys.path so this import resolves.
from scripts.scrub_dev_paths import (
    find_home_paths,
    main,
    scan_file,
)


# ---------------------------------------------------------------------------
# find_home_paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "line",
    [
        'CACHE = "/Users/alice/.iai-mcp"',
        "log_dir = Path('/home/bob/logs')",
        'DEFAULT = "/Users/ci-runner/build/out"',
        "path = '/home/deploy.user/state'",
    ],
)
def test_home_paths_flagged(line):
    assert find_home_paths(line) != [], f"should flag a home path: {line!r}"


@pytest.mark.parametrize(
    "line",
    [
        'store = Path.home() / ".iai-mcp"',
        'DOCS = "https://example.com/Users/guide"',
        "# the /Users API is a macOS convention",
        'rel = "src/iai_mcp/store.py"',
        "",
    ],
)
def test_clean_lines_not_flagged(line):
    assert find_home_paths(line) == [], f"false positive on: {line!r}"


def test_account_segment_required():
    """A bare mention without an account segment is not a dev path."""
    assert find_home_paths("mounted at /Users") == []
    assert find_home_paths("see /home for details") == []


def test_all_matches_returned():
    line = '"/Users/alice/a" and "/home/bob/b"'
    assert len(find_home_paths(line)) == 2


# ---------------------------------------------------------------------------
# scan_file
# ---------------------------------------------------------------------------

def test_scan_file_reports_line_numbers(tmp_path: Path):
    target = tmp_path / "mod.py"
    target.write_text(
        "clean = 1\n"
        'leaked = "/Users/alice/.config"\n'
        "also_clean = 2\n",
        encoding="utf-8",
    )
    hits = scan_file(target)
    assert len(hits) == 1
    line_no, match, _line = hits[0]
    assert line_no == 2
    assert match == "/Users/alice/"


def test_scan_file_clean_returns_empty(tmp_path: Path):
    target = tmp_path / "clean.py"
    target.write_text("value = Path.home()\n", encoding="utf-8")
    assert scan_file(target) == []


def test_scan_file_skips_unreadable(tmp_path: Path):
    target = tmp_path / "blob.py"
    target.write_bytes(b"\xff\xfe\x00binary")
    assert scan_file(target) == []


def test_scan_file_missing_path_is_silent(tmp_path: Path):
    assert scan_file(tmp_path / "absent.py") == []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _shipped(tmp_path: Path, name: str, body: str) -> None:
    root = tmp_path / "src" / "iai_mcp"
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(body, encoding="utf-8")


def test_main_exits_zero_on_clean_tree(tmp_path: Path, monkeypatch):
    _shipped(tmp_path, "ok.py", "value = Path.home()\n")
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0


def test_main_exits_nonzero_on_violation(tmp_path: Path, monkeypatch):
    _shipped(tmp_path, "bad.py", 'p = "/Users/alice/.iai-mcp"\n')
    monkeypatch.chdir(tmp_path)
    assert main([]) == 1


def test_main_reports_violation_to_stderr(tmp_path: Path, monkeypatch, capsys):
    _shipped(tmp_path, "bad.py", 'p = "/home/bob/state"\n')
    monkeypatch.chdir(tmp_path)
    main([])
    err = capsys.readouterr().err
    assert "bad.py" in err
    assert "/home/bob/" in err


def test_main_ignores_paths_outside_shipped_roots(tmp_path: Path, monkeypatch):
    """Fixtures and docs may hold sample paths; only shipped source is gated."""
    fixtures = tmp_path / "tests"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "sample.py").write_text('p = "/Users/alice/x"\n', encoding="utf-8")
    _shipped(tmp_path, "ok.py", "value = 1\n")
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0


def test_main_ignores_unscanned_suffixes(tmp_path: Path, monkeypatch):
    root = tmp_path / "src" / "iai_mcp"
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.md").write_text("see /Users/alice/notes\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0
