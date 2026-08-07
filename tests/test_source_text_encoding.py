"""Every tracked text surface stays UTF-8 and free of debug scaffolding."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".sh", ".ts", ".yml", ".yaml", ".cfg"}

# A UTF-8 file decoded as cp1251 and re-encoded turns every dash and curly
# quote into one of these. They are never legitimate in our sources.
MOJIBAKE = ("вЂ", "Ð²Ð", "â€", "Ã¢â‚¬")

# PowerShell 5.1 needs the marker to read a script as UTF-8; nothing else does.
BOM_ALLOWED_SUFFIXES = {".ps1"}

DEBUG_MARKERS = ('print("DEBUG', "print(f\"DEBUG", 'print("TRACEBACK', "traceback.print_exc(")

# Only the shipped library surface. A bench harness prints stack traces on
# purpose: the diagnostics are its output, not scaffolding left behind.
DEBUG_ROOTS = ("src/", "mcp-wrapper/src/")

DEBUG_EXEMPT = {
    # names the markers in order to forbid them
    "tests/test_source_text_encoding.py",
}


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    return [
        REPO / rel
        for rel in out
        if rel and (Path(rel).suffix in TEXT_SUFFIXES or Path(rel).suffix == ".ps1")
    ]


TRACKED = _tracked_text_files()


def test_the_scan_sees_a_real_tree() -> None:
    assert len(TRACKED) > 100


@pytest.mark.parametrize("path", TRACKED, ids=lambda p: str(p.relative_to(REPO)))
def test_no_mojibake(path: Path) -> None:
    text = path.read_bytes().decode("utf-8", errors="replace")
    for marker in MOJIBAKE:
        assert marker not in text, (
            f"{path.relative_to(REPO)} carries {marker!r} — the file was written "
            "by an editor that read it as cp1251. Re-save it as UTF-8."
        )


@pytest.mark.parametrize("path", TRACKED, ids=lambda p: str(p.relative_to(REPO)))
def test_no_unexpected_byte_order_mark(path: Path) -> None:
    if path.suffix in BOM_ALLOWED_SUFFIXES:
        return
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf"), (
        f"{path.relative_to(REPO)} starts with a byte-order mark"
    )


@pytest.mark.parametrize(
    "path",
    [
        p
        for p in TRACKED
        if p.suffix == ".py"
        and str(p.relative_to(REPO)).startswith(DEBUG_ROOTS)
    ],
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_no_debug_scaffolding(path: Path) -> None:
    rel = str(path.relative_to(REPO))
    if rel in DEBUG_EXEMPT:
        return
    text = path.read_text(encoding="utf-8")
    for marker in DEBUG_MARKERS:
        assert marker not in text, f"{rel} still carries {marker!r}"
