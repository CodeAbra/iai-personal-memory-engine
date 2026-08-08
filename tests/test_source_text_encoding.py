"""Every tracked text surface stays UTF-8, free of debug scaffolding, and
every text-mode file open under src/ declares its encoding — the platform
default is cp1252 on Windows and would silently mangle any non-ASCII
state-file content."""
from __future__ import annotations

import re
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

# The detector names every marker in order to forbid it — exempt from all scans.
SELF = "tests/test_source_text_encoding.py"


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split("\0")
    return [
        REPO / rel
        for rel in out
        if rel
        and rel != SELF
        and (Path(rel).suffix in TEXT_SUFFIXES or Path(rel).suffix == ".ps1")
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
    text = path.read_text(encoding="utf-8")
    for marker in DEBUG_MARKERS:
        assert marker not in text, f"{rel} still carries {marker!r}"


# --- text-mode opens must declare their encoding -------------------------

#: Call shapes that read or write TEXT through the platform-default codec
#: when no encoding is given. Object methods that merely share the name
#: (zipfile/tarfile/webbrowser/engine .open) are excluded by shape.
_FDOPEN_RX = re.compile(r'os\.fdopen\([^)]*"[rwa]"')
_PATH_OPEN_RX = re.compile(r'\.open\(\s*(?:"[rwa]"\s*)?\)')
_READ_TEXT_RX = re.compile(r"\.read_text\(")
_WRITE_TEXT_RX = re.compile(r"\.write_text\(")
_BARE_OPEN_RX = re.compile(r'(?<![\w.])open\((?![^)]*"[rw]b")')


def _joined_call(lines: list[str], idx: int) -> str:
    # A call may close on a later line; judge up to four joined lines so a
    # trailing encoding= argument is seen.
    return " ".join(x.strip() for x in lines[idx:idx + 4])


@pytest.mark.parametrize(
    "path",
    [
        p
        for p in TRACKED
        if p.suffix == ".py" and str(p.relative_to(REPO)).startswith("src/iai_mcp")
    ],
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_text_opens_declare_encoding(path: Path) -> None:
    rel = str(path.relative_to(REPO))
    lines = path.read_text(encoding="utf-8").splitlines()
    offenders = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        hit = (
            _FDOPEN_RX.search(line)
            or _READ_TEXT_RX.search(line)
            or _WRITE_TEXT_RX.search(line)
            or _PATH_OPEN_RX.search(line)
            or _BARE_OPEN_RX.search(line)
        )
        if not hit:
            continue
        window = _joined_call(lines, i)
        if "encoding=" in window:
            continue
        if "read_bytes" in line or "write_bytes" in line or "os.open(" in line:
            continue
        # importlib.metadata's Distribution.read_text(filename) takes no
        # encoding parameter (it decodes utf-8 itself); forcing one raises
        # TypeError.
        if "distribution(" in line:
            continue
        offenders.append(f"{rel}:{i + 1}: {stripped[:100]}")
    assert not offenders, (
        "text-mode open without encoding= (platform default is cp1252 on "
        "Windows):\n" + "\n".join(offenders)
    )
