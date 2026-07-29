"""File NAMES are a shipped surface too. The content guard let 22
phase-coded test filenames survive every release; downstream the public
tree grew coded/clean duplicate pairs where a sync updates one copy and
the other goes silently stale. A name is forbidden when it carries a
planning code — a phase/wave/task/milestone number, a requirement id, or
governance framing — rather than what the module tests.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_FORBIDDEN_NAME_PATTERNS = (
    re.compile(r"(?:^|_)(?:phase|wave|task|plan|milestone)_?\d", re.IGNORECASE),
    re.compile(r"(?:^|_)mem_?\d{2}", re.IGNORECASE),
    re.compile(r"constitutional", re.IGNORECASE),
    re.compile(r"(?:^|_)[dr]\d{2}(?:_|\.)", re.IGNORECASE),
)

_SCANNED_ROOTS = ("src", "tests", "bench", "scripts", "mcp-wrapper/src")

# Code surfaces only: archived result records under bench/results/ keep
# their historical names by design.
_SOURCE_SUFFIXES = (".py", ".ts", ".js", ".sh", ".rs")


def _tracked_files() -> "list[str]":
    out = subprocess.run(
        ["git", "ls-files", *_SCANNED_ROOTS],
        cwd=_REPO, capture_output=True, text=True, check=True,
    )
    return [
        l for l in out.stdout.splitlines()
        if l.strip()
        and l.endswith(_SOURCE_SUFFIXES)
        and not l.startswith("bench/results/")
    ]


def test_no_planning_codes_in_tracked_file_names() -> None:
    violations = []
    for rel in _tracked_files():
        name = Path(rel).name
        for pat in _FORBIDDEN_NAME_PATTERNS:
            if pat.search(name):
                violations.append(f"{rel}  (pattern: {pat.pattern})")
                break
    assert not violations, (
        "planning codes in file names — name the module after what it "
        "does, not the plan that produced it:\n" + "\n".join(violations)
    )
