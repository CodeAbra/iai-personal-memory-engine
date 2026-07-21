"""Lint: enforce CONVENTIONS.md `# @internal` marker rule.

Rule:
    `# @internal Phase X / D-Y` lives on its own comment line; the technical
    sentence goes on the next comment line so the public-repo scrub can drop
    the tag line without losing the rationale.

Discrimination contract (token-based, verified against the 8-line fixture set
in `20-02-PLAN.md <interfaces>`):
    1. Strip the optional leading whitespace and `# @internal` prefix.
    2. If nothing remains -> OK (bare marker).
    3. If the tail contains an em-dash `—` -> VIOLATION.
    4. Split the tail by whitespace; if ANY token starts with a lowercase
       ASCII letter -> VIOLATION (English prose past the marker).
    5. Otherwise -> OK (capitalized Phase/Plan/Decision IDs are accepted).

Modes:
    default       Walk `src/iai_mcp/` and `mcp-wrapper/src/`; report all hits.
                  Used for manual audit. NOT used by the pre-commit hook
                  because CONVENTIONS.md mandates "no mass rewrite" of
                  pre-existing historical lines.
    --check-staged
                  Walk only files in `git diff --cached --name-only` that
                  fall under the same two roots. This is the forward-promise
                  mode used by `.git/hooks/pre-commit`.

Output: each violation prints `path:line: <reason>` to stderr.
Exit: 0 on clean, 1 on any violation.

Stdlib only — no third-party dependencies.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Files we scan: Python in src/iai_mcp/, TypeScript in mcp-wrapper/src/.
SCAN_ROOTS: tuple[str, ...] = ("src/iai_mcp", "mcp-wrapper/src")
SCAN_SUFFIXES: tuple[str, ...] = (".py", ".ts")

# Marker prefix: optional leading whitespace, `#`, optional space, `@internal`.
_MARKER_RE = re.compile(r"^\s*#\s*@internal\b(?P<tail>.*)$")


def is_violation(line: str) -> bool:
    """Return True if `line` mixes `# @internal` with technical description.

    See module docstring for the discrimination contract. Lines without the
    `@internal` marker return False (out of scope).
    """
    match = _MARKER_RE.match(line)
    if match is None:
        return False  # not an @internal line; not our concern.
    tail = match.group("tail").rstrip("\n")
    # Strip leading whitespace inside the tail (the space right after
    # `@internal`).
    stripped = tail.strip()
    if not stripped:
        return False  # bare `# @internal` is fine.
    # Em-dash + description is the canonical bad form.
    if "—" in stripped:
        return True
    # Any token starting with lowercase ASCII = English prose past marker.
    for token in stripped.split():
        if token and token[0].islower() and token[0].isascii():
            return True
    return False


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Walk `path` line-by-line; return [(line_no, reason), ...].

    `line_no` is 1-indexed. `reason` describes the violation for stderr.
    File-read errors return an empty list (the file is skipped silently).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    out: list[tuple[int, str]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if is_violation(line):
            out.append((idx, "@internal marker mixed with description"))
    return out


def _iter_default_files(repo_root: Path) -> list[Path]:
    """Walk SCAN_ROOTS recursively; return all .py and .ts files."""
    out: list[Path] = []
    for root_rel in SCAN_ROOTS:
        root = repo_root / root_rel
        if not root.is_dir():
            continue
        for suffix in SCAN_SUFFIXES:
            out.extend(root.rglob(f"*{suffix}"))
    return out


def _iter_staged_files(repo_root: Path) -> list[Path]:
    """Run `git diff --cached --name-only`; filter to SCAN_ROOTS + suffixes."""
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
    """Entry point. Returns 0 on clean, 1 on any violation."""
    parser = argparse.ArgumentParser(
        description="Lint `# @internal` markers per CONVENTIONS.md."
    )
    parser.add_argument(
        "--check-staged",
        action="store_true",
        help=(
            "scan only files in `git diff --cached --name-only` "
            "(used by the pre-commit hook)"
        ),
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
        for line_no, reason in scan_file(path):
            print(f"{rel}:{line_no}: {reason}", file=sys.stderr)
            violations += 1

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
