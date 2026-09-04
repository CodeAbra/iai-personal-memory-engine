"""Lint: reject developer home paths in production source.

Production source under `src/iai_mcp/` and `mcp-wrapper/src/` must not
contain a literal user home path — `/Users/<name>/` (macOS) or
`/home/<name>/` (Linux). A hardcoded home path leaks a maintainer or
contributor identity into the shipped surface and always indicates a
config value that should come from `Path.home()` or the environment.

Out of scope by construction (not under SCAN_ROOTS): tests/, docs/,
bench/, .planning/, mcp-wrapper/test/, repo-root files. Fixtures and
harnesses may legitimately carry example paths.

Modes:
    default         Walk SCAN_ROOTS; report all hits.
    --check-staged  Scan only files in `git diff --cached --name-only`
                    (pre-commit hook mode).

Output: `path:line: <pattern>: <line_content>` to stderr.
Exit: 0 clean, 1 on any violation. Stdlib only.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCAN_ROOTS: tuple[str, ...] = ("src/iai_mcp", "mcp-wrapper/src")
SCAN_SUFFIXES: tuple[str, ...] = (".py", ".ts")

# A home-path segment: /Users/<name>/ or /home/<name>/ where <name> is a
# plausible account name (no separators, no whitespace, no quotes). The
# trailing slash keeps bare protocol mentions ("the /Users API") clean.
_HOME_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")


def find_home_paths(line: str) -> list[str]:
    """Return home-path matches in `line` (empty = clean)."""
    return _HOME_PATH_RE.findall(line)


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Walk `path` line-by-line; return [(line_no, match, line), ...].

    File-read errors (binary file, broken UTF-8) skip the file silently.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    out: list[tuple[int, str, str]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        for match in find_home_paths(line):
            out.append((idx, match, line))
    return out


def _iter_default_files(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for root_rel in SCAN_ROOTS:
        root = repo_root / root_rel
        if not root.is_dir():
            continue
        for suffix in SCAN_SUFFIXES:
            out.extend(root.rglob(f"*{suffix}"))
    return out


def _iter_staged_files(repo_root: Path) -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    out: list[Path] = []
    for rel in proc.stdout.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        if not rel.endswith(SCAN_SUFFIXES):
            continue
        if not any(rel.startswith(root + "/") for root in SCAN_ROOTS):
            continue
        path = repo_root / rel
        if path.is_file():
            out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint developer home paths in production source."
    )
    parser.add_argument(
        "--check-staged",
        action="store_true",
        help="scan only staged files (pre-commit hook mode)",
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    files = (
        _iter_staged_files(repo_root)
        if args.check_staged
        else _iter_default_files(repo_root)
    )

    violations = 0
    for path in files:
        rel = path.relative_to(repo_root) if path.is_absolute() else path
        for line_no, pattern, line in scan_file(path):
            print(f"{rel}:{line_no}: {pattern}: {line}", file=sys.stderr)
            violations += 1

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
