"""Guards the storage-driver default's single source of truth.

Every native-engine-only test gate and every path resolving "what driver
applies to a fresh store" must derive from one imported constant. A gate
that reads the raw environment variable with a hardcoded string fallback
desynchronises from the real resolved default the moment that default
changes, silently reintroducing a stale skip predicate with no failing
test to catch it. This module is that failing test.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER, _resolve_effective_driver

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
_SELF_PATH = Path(__file__).resolve()

# Matches `os.environ.get("LILLI_STORAGE_DRIVER", <quoted literal>)` in either
# quote style. Keys on the quoted literal in the fallback position, not on the
# two-argument call shape itself -- the compliant form passes the imported
# constant (an identifier, never a quote) as that argument.
_HARDCODED_FALLBACK_PATTERN = re.compile(
    r"""os\.environ\.get\(\s*["']LILLI_STORAGE_DRIVER["']\s*,\s*["']"""
)


def _scan_dir(root: Path) -> list[str]:
    offenders: list[str] = []
    if not root.exists():
        return offenders
    for path in sorted(root.rglob("*.py")):
        if path.resolve() == _SELF_PATH:
            continue
        if "egg-info" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _HARDCODED_FALLBACK_PATTERN.search(line):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    return offenders


def _offending_lines() -> list[str]:
    # Every production call site resolving "what driver applies to a fresh
    # store" AND every test-tree gate predicate must derive from the same
    # imported constant -- so both trees are in scope, not just tests/.
    return _scan_dir(_TESTS_DIR) + _scan_dir(_SRC_DIR)


def test_no_hardcoded_driver_fallback_reintroduced() -> None:
    offenders = _offending_lines()
    assert not offenders, (
        "hardcoded storage-driver fallback reintroduced -- pass "
        "DEFAULT_STORAGE_DRIVER (imported from iai_mcp.hippo._db) as the "
        "os.environ.get(...) fallback instead of a literal string:\n"
        + "\n".join(offenders)
    )


def test_default_constant_resolves_consistently_for_a_fresh_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert DEFAULT_STORAGE_DRIVER == "lilli"

    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    fresh_path = str(tmp_path / "not-yet-created" / "brain.sqlite3")
    assert not os.path.exists(fresh_path)
    assert _resolve_effective_driver(fresh_path) == DEFAULT_STORAGE_DRIVER
