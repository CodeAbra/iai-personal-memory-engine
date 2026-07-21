from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

def _planning_is_tracked(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "ls-files", ".planning"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return bool(result.stdout.strip())

def test_license_exists_and_is_mit() -> None:
    license_path = _REPO_ROOT / "LICENSE"
    assert license_path.exists(), f"LICENSE not found at {license_path}"

    content = license_path.read_text(encoding="utf-8")
    assert "MIT License" in content, "LICENSE missing 'MIT License' header"
    assert "Copyright (c) 2026" in content, "LICENSE missing canonical copyright year"
    assert "WITHOUT WARRANTY" in content, "LICENSE missing warranty disclaimer"
    assert (
        license_path.stat().st_size > 1024
    ), f"LICENSE too small ({license_path.stat().st_size} bytes); expected > 1 KiB"

    assert "GPL" not in content, "LICENSE contains 'GPL' — license collision"
    assert "Apache-2.0" not in content, "LICENSE contains 'Apache-2.0' — license collision"
    assert "Proprietary" not in content, "LICENSE contains 'Proprietary' — license collision"

def test_notice_exists_and_lists_runtime_deps() -> None:
    notice_path = _REPO_ROOT / "NOTICE.md"
    assert notice_path.exists(), f"NOTICE.md not found at {notice_path}"

    content = notice_path.read_text(encoding="utf-8")
    assert (
        notice_path.stat().st_size > 2048
    ), f"NOTICE.md too small ({notice_path.stat().st_size} bytes); expected > 2 KiB"

    python_runtime_deps = (
        "lancedb",
        "pyarrow",
        "numpy",
        "pydantic",
        "torch-hd",
        "structlog",
        "networkx",
        "numba",
        "anthropic",
        "tiktoken",
        "cryptography",
        "keyring",
        "cachetools",
        "psutil",
    )
    for dep in python_runtime_deps:
        assert dep in content, f"NOTICE.md missing Python runtime dep: {dep!r}"

    npm_runtime_deps = ("@modelcontextprotocol/sdk", "zod")
    for dep in npm_runtime_deps:
        assert dep in content, f"NOTICE.md missing npm runtime dep: {dep!r}"

@pytest.mark.skipif(
    _planning_is_tracked(_REPO_ROOT),
    reason="private repo tracks .planning/ by design; the gitignore convention is a public-release check",
)
def test_planning_dir_is_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v", ".planning/STATE.md"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"git check-ignore --no-index returned {result.returncode}; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert ".gitignore:" in result.stdout, (
        f"git check-ignore stdout does not cite .gitignore: {result.stdout!r}"
    )
    assert ".planning/" in result.stdout, (
        f"git check-ignore stdout missing .planning/ pattern: {result.stdout!r}"
    )

def test_scrub_dev_paths_exits_clean_on_full_tree() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/scrub_dev_paths.py"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"scrub_dev_paths.py exited {result.returncode}.\n"
        f"stderr (violations):\n{result.stderr}"
    )

def test_scrub_internal_markers_exits_clean_on_full_tree() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/scrub_internal_markers.py"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"scrub_internal_markers.py exited {result.returncode}.\n"
        f"stderr (violations):\n{result.stderr}"
    )
