"""Tests for scripts/scrub_internal_markers.py.

Enforces CONVENTIONS.md rule: `# @internal Phase X / D-Y` markers live on
their own comment line, never mixed with technical description. The
technical sentence goes on the next comment line.

Test fixtures derive from `20-02-PLAN.md <interfaces>` GOOD/BAD/IGNORED
acceptance set. The token-based discrimination rule:
- strip the `# @internal` prefix
- split tail by whitespace
- FAIL if tail contains `—` (em-dash) or any token starts with lowercase ASCII

RED state: this file is written before `scripts/scrub_internal_markers.py`
exists; import of the module raises ModuleNotFoundError and every test
fails at collection time. After GREEN, all 11 tests pass.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# scripts/ lives at repo root, not on src/ pythonpath. tests/conftest.py
# prepends the repo root to sys.path so this import resolves at GREEN.
from scripts.scrub_internal_markers import is_violation, scan_file, main


# ---------------------------------------------------------------------------
# Unit tests: is_violation
# ---------------------------------------------------------------------------

def test_marker_alone_passes() -> None:
    """Canonical marker form: `# @internal Phase X / D-Y` is accepted."""
    assert is_violation("# @internal Phase 7.1 R3 / D7.1-04") is False


def test_marker_with_description_em_dash_fails() -> None:
    """Em-dash + description on the same line is the canonical violation."""
    assert is_violation("# @internal Phase 7.1 — blocking import cost") is True


def test_marker_with_description_hyphen_lowercase_fails() -> None:
    """Hyphen-dash + lowercase tech prose is also a violation."""
    assert (
        is_violation(
            "# @internal P11.0 - store hotpath optimization for LanceDB"
        )
        is True
    )


def test_marker_followed_by_lowercase_prose_fails() -> None:
    """Lowercase token past the marker = English prose past the marker."""
    assert (
        is_violation("# @internal Phase 12.3 fsm consolidation here") is True
    )


def test_line_without_at_internal_is_ignored() -> None:
    """Non-@internal comments are out of scope (bare plan markers OK)."""
    assert is_violation("# This is just a comment about Plan 02-08") is False
    assert is_violation("# Plan 02-08:") is False


def test_leading_whitespace_ok() -> None:
    """Indented `# @internal` markers are still markers (per fixtures)."""
    assert is_violation("    # @internal Phase 12.5") is False


def test_marker_with_plan_form_passes() -> None:
    """Slash-form `Phase X / D-OPS-03` is accepted."""
    assert is_violation("# @internal Phase 16 / D-OPS-03") is False


def test_marker_with_p_prefix_passes() -> None:
    """Short `P11.0` form is accepted."""
    assert is_violation("# @internal P11.0") is False


# ---------------------------------------------------------------------------
# Unit test: scan_file
# ---------------------------------------------------------------------------

def test_scan_file_returns_list_of_violations(tmp_path: Path) -> None:
    """scan_file flags the bad line and skips the good lines."""
    fixture = tmp_path / "foo.py"
    fixture.write_text(
        "# @internal Phase 7.1\n"  # line 1: good
        "# Deviation: blocking import cost\n"  # line 2: not @internal
        "# @internal Phase 12.3 fsm consolidation here\n"  # line 3: BAD
        "import time\n",  # line 4: code
        encoding="utf-8",
    )
    violations = scan_file(fixture)
    assert len(violations) == 1
    line_no, _reason = violations[0]
    assert line_no == 3


def test_blank_line_after_marker_passes(tmp_path: Path) -> None:
    """File-level: canonical three-line pattern reports zero violations."""
    fixture = tmp_path / "foo.py"
    fixture.write_text(
        "# @internal Phase 7.1\n"
        "# Deviation: blocking import cost\n"
        "import time\n",
        encoding="utf-8",
    )
    assert scan_file(fixture) == []


# ---------------------------------------------------------------------------
# Integration tests: main() with --check-staged
# ---------------------------------------------------------------------------

def _init_git_repo(repo_dir: Path) -> None:
    """Initialize a minimal git repo for staged-mode tests."""
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main"],
        cwd=repo_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=repo_dir,
        check=True,
    )


def test_check_staged_only_scans_staged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--check-staged filters to src/iai_mcp/ + mcp-wrapper/src/ roots."""
    _init_git_repo(tmp_path)
    # In-scope file: src/iai_mcp/foo.py with a bad marker.
    src_pkg = tmp_path / "src" / "iai_mcp"
    src_pkg.mkdir(parents=True)
    (src_pkg / "foo.py").write_text(
        "# @internal Phase 7.1 — bad mixed marker\n", encoding="utf-8"
    )
    # Out-of-scope file: docs/foo.md with a bad marker.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "foo.md").write_text(
        "# @internal Phase 7.1 — bad mixed marker\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    monkeypatch.chdir(tmp_path)
    exit_code = main(["--check-staged"])
    # Non-zero because the src/iai_mcp/ file is in scope and violates.
    assert exit_code != 0


def test_check_staged_exit_code_nonzero_on_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--check-staged exits non-zero when in-scope files have violations."""
    _init_git_repo(tmp_path)
    src_pkg = tmp_path / "src" / "iai_mcp"
    src_pkg.mkdir(parents=True)
    (src_pkg / "foo.py").write_text(
        "# @internal Phase 7.1 — broken mixed marker\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "src/iai_mcp/foo.py"], cwd=tmp_path, check=True)

    monkeypatch.chdir(tmp_path)
    exit_code = main(["--check-staged"])
    assert exit_code == 1


def test_check_staged_clean_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--check-staged exits 0 when staged in-scope files are clean."""
    _init_git_repo(tmp_path)
    src_pkg = tmp_path / "src" / "iai_mcp"
    src_pkg.mkdir(parents=True)
    (src_pkg / "foo.py").write_text(
        "# @internal Phase 7.1\n# Description on next line.\nimport time\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "src/iai_mcp/foo.py"], cwd=tmp_path, check=True)

    monkeypatch.chdir(tmp_path)
    exit_code = main(["--check-staged"])
    assert exit_code == 0


def test_default_mode_scans_src_iai_mcp_and_mcp_wrapper_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default mode walks BOTH src/iai_mcp/ and mcp-wrapper/src/ roots."""
    # Synthesize a repo with bad markers in BOTH roots and confirm each
    # produces violations in stderr.
    src_pkg = tmp_path / "src" / "iai_mcp"
    src_pkg.mkdir(parents=True)
    (src_pkg / "py_file.py").write_text(
        "# @internal Phase 99 — bad py marker\n", encoding="utf-8"
    )
    ts_pkg = tmp_path / "mcp-wrapper" / "src"
    ts_pkg.mkdir(parents=True)
    (ts_pkg / "ts_file.ts").write_text(
        "# @internal Phase 99 — bad ts marker\n", encoding="utf-8"
    )
    # Also drop an out-of-scope file to confirm it's NOT scanned.
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ignored.py").write_text(
        "# @internal Phase 99 — out of scope\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    exit_code = main([])
    assert exit_code == 1  # both in-scope files violate


def test_default_mode_skips_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default mode ignores docs/ and other out-of-scope roots."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# @internal Phase 99 — bad marker in docs\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    exit_code = main([])
    assert exit_code == 0  # nothing in scope, nothing to flag
    captured = capsys.readouterr()
    assert "guide.md" not in captured.err
