"""Shipped text carries no development-process bookkeeping.

Comments, docstrings, string literals and names all travel into releases, so
none of them may carry planning identifiers, internal document references, or
collaboration bookkeeping — they are meaningless to anyone outside this repo.

This is a ratchet, not a sweep. ``KNOWN`` records the files that still carry
markers, with an exact count each. A new marker anywhere fails immediately; a
cleaned-up file fails too, which is the signal to lower its number. The list may
only shrink.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

MARKERS = (
    # No word-boundary anchors on the left/right of phase/wave/constitutional:
    # planning codes hide inside identifiers (`_os_phaseNN`,
    # `..._constitutional_value`), where `\b` never fires.
    re.compile(r"phase[ _-]?\d", re.I),
    re.compile(r"\bplan \d{2}-\d{2}\b", re.I),
    re.compile(r"wave[ _-]?\d", re.I),
    re.compile(r"\bD-\d{2}\b"),
    re.compile(r"constitutional", re.I),
    re.compile(r"\.planning/"),
    re.compile(r"\b(?:CONTEXT|ROADMAP|STATE)\.md\b"),
    re.compile(r"Co-Authored-By", re.I),
    re.compile(r"#\s*@internal\b"),
)

SCANNED_SUFFIXES = {
    ".py", ".pyi", ".sh", ".ts", ".rs", ".toml", ".json", ".yml", ".yaml", ".md",
}

# These files carry the markers because they are the detectors, plus the two
# documents whose JOB is to hold the process record: the CHANGELOG (the
# sanctioned home for the "why") and the public-code scrub rulebook.
DETECTORS = {
    "tests/test_no_dev_process_markers.py",
    "tests/test_scrub_internal_markers.py",
    "tests/test_release_hardening.py",
    "scripts/scrub_internal_markers.py",
    "scripts/lint_personal.py",
    "scripts/scrub_dev_paths.py",
    "tests/test_no_planning_codes_in_filenames.py",
    "CHANGELOG.md",
    "CTO-PUBLIC-CODE-RULES.md",
    # names the forbidden commit trailer in order to forbid it
    "CONTRIBUTING.md",
}

# Working-notes trees that never ship and never port: measurement archives and
# the planning records themselves. Everything else that is tracked is scanned.
EXCLUDED_PREFIXES = (".planning/", "bench/results/")

# Process records whose integrity depends on NOT rewriting them: the
# pre-registered bench protocol (amendments are append-only by design), an
# archived third-party opinion, the orchestrator's working notes, and the
# project instruction file whose job is to state these very rules.
RECORDS = {
    "BENCH_PROTOCOL_lme500.md",
    "bench/OPINION_FROM_CURSOR_2026-05-02.md",
    "CLAUDE.md",
    "FABLE_PROGRESS.md",
    "FABLE_FINDINGS.json",
}

# Outstanding cleanup, counted exactly. Lower a number when you clean a file;
# never raise one, and never add a line.
KNOWN: dict[str, int] = {
    "tests/fixtures/sigma_baseline.json": 1,
}


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [
        line
        for line in out.stdout.splitlines()
        if Path(line).suffix in SCANNED_SUFFIXES
        and line not in DETECTORS
        and line not in RECORDS
        and not line.startswith(EXCLUDED_PREFIXES)
    ]


def _hits(path: str) -> list[str]:
    try:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    found = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rx in MARKERS:
            if rx.search(line):
                found.append(f"{path}:{lineno}: {line.strip()[:100]}")
                break
    return found


@pytest.fixture(scope="module")
def scan() -> dict[str, list[str]]:
    return {p: h for p in _tracked_files() if (h := _hits(p))}


def test_no_new_dev_process_markers(scan: dict[str, list[str]]) -> None:
    unexpected = []
    for path, hits in sorted(scan.items()):
        allowed = KNOWN.get(path, 0)
        if len(hits) > allowed:
            unexpected.extend(hits[allowed:] if allowed else hits)
    assert not unexpected, (
        "development-process markers in shipped text — rewrite them functionally "
        "or drop them (planning codes, internal doc names and collaboration "
        "bookkeeping mean nothing to a reader outside this repo):\n  "
        + "\n  ".join(unexpected)
    )


def test_known_list_only_shrinks(scan: dict[str, list[str]]) -> None:
    stale = {
        path: (allowed, len(scan.get(path, [])))
        for path, allowed in KNOWN.items()
        if len(scan.get(path, [])) < allowed
    }
    assert not stale, (
        "these files are cleaner than KNOWN claims — lower the numbers so the "
        "ratchet keeps holding (path: recorded -> actual): "
        + ", ".join(f"{p}: {a} -> {n}" for p, (a, n) in sorted(stale.items()))
    )
