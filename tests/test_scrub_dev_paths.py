"""Home-path linter: production source must not carry a developer home path.

Covers the generic rule (`/Users/<name>/`, `/home/<name>/`), the scan-root
allowlist (tests/, bench/, docs live outside SCAN_ROOTS), and both modes'
exit contract.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scrub_dev_paths import find_home_paths, main, scan_file  # noqa: E402


def test_macos_home_path_is_flagged() -> None:
    assert find_home_paths('path = "/Users/somedev/.iai-mcp"') != []


def test_linux_home_path_is_flagged() -> None:
    assert find_home_paths('path = "/home/somedev/.config"') != []


def test_bare_users_api_mention_passes() -> None:
    assert find_home_paths("the /Users API returns accounts") == []


def test_no_trailing_slash_passes() -> None:
    assert find_home_paths('name = "/Users/somedev"') == []


def test_scan_file_reports_line_numbers(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text('ok = 1\nhome = "/Users/somedev/x"\n', encoding="utf-8")
    hits = scan_file(f)
    assert len(hits) == 1
    assert hits[0][0] == 2


def test_binary_file_skips_silently(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_bytes(b"\xff\xfe\x00bad")
    assert scan_file(f) == []


def _repo(tmp_path: Path, rel: str, content: str) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "iai_mcp").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return root


def test_default_mode_flags_src(tmp_path: Path, monkeypatch) -> None:
    root = _repo(tmp_path, "src/iai_mcp/mod.py", 'p = "/Users/somedev/x"\n')
    monkeypatch.chdir(root)
    assert main([]) == 1


def test_default_mode_ignores_tests_dir(tmp_path: Path, monkeypatch) -> None:
    root = _repo(tmp_path, "tests/test_x.py", 'p = "/Users/somedev/x"\n')
    monkeypatch.chdir(root)
    assert main([]) == 0


def test_default_mode_clean_exits_zero(tmp_path: Path, monkeypatch) -> None:
    root = _repo(tmp_path, "src/iai_mcp/mod.py", "ok = 1\n")
    monkeypatch.chdir(root)
    assert main([]) == 0


def test_check_staged_flags_staged_src(tmp_path: Path, monkeypatch) -> None:
    root = _repo(tmp_path, "src/iai_mcp/mod.py", 'p = "/home/somedev/x"\n')
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    monkeypatch.chdir(root)
    assert main(["--check-staged"]) == 1
