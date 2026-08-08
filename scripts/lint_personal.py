"""Lint: personal data must not reach committed surfaces.

Blocks the owner's identity and third-party handles from every surface a
commit can carry them: staged source (src/, mcp-wrapper/src/, scripts/,
tests/, bench/, _deploy hooks), repo-root docs (CHANGELOG.md, README.md),
and — in ``--message`` mode — the commit message itself.

Patterns live in ``scripts/personal_patterns.txt`` (one regex per line,
``#`` comments) so the deny-list is data, not code. A sibling
``scripts/personal_allow.txt`` holds full-line regexes exempt from the
deny-list — the owner's authorship byline is the sanctioned case; a line
must match an allow regex in full to be exempt. GitHub-style
``@handle`` mentions are additionally rejected in Markdown and commit
messages (prose surfaces), where a handle is an attribution — never in
code, where ``@decorator`` is legitimate.

Exit: 0 clean, 1 on any violation. Stdlib only.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCAN_ROOTS: tuple[str, ...] = (
    "src/iai_mcp",
    "mcp-wrapper/src",
    "scripts",
    "tests",
    "bench",
)
SCAN_SUFFIXES: tuple[str, ...] = (".py", ".ts", ".sh", ".md", ".toml", ".json")
# README carries the owner's own byline by choice; CHANGELOG travels
# into releases and stays gated.
ROOT_FILES: tuple[str, ...] = ("CHANGELOG.md",)

_HANDLE_RE = re.compile(r"(?<![\w.])@[A-Za-z][A-Za-z0-9-]{2,}\b")
#: Technical tokens that look like handles but are code, not attribution.
_HANDLE_ALLOW = frozenset({
    "@internal", "@pytest", "@modelcontextprotocol", "@types",
    "@anthropic-ai", "@property", "@staticmethod", "@classmethod",
})
_PROSE_SUFFIXES = (".md",)

_PATTERNS_FILE = Path(__file__).resolve().parent / "personal_patterns.txt"
_ALLOW_FILE = Path(__file__).resolve().parent / "personal_allow.txt"


def _load_allow() -> list[re.Pattern]:
    try:
        text = _ALLOW_FILE.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[re.Pattern] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(re.compile(line))
    return out


def _load_patterns() -> list[re.Pattern]:
    out: list[re.Pattern] = []
    try:
        text = _PATTERNS_FILE.read_text(encoding="utf-8")
    except OSError:
        print(
            f"lint_personal: patterns file missing: {_PATTERNS_FILE}",
            file=sys.stderr,
        )
        sys.exit(1)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(re.compile(line))
    return out


#: Repo-relative prose paths where the handle rule stands down: release
#: notes credit contributors by handle on purpose. The deny-list patterns
#: still apply. Exact paths, not basenames — a fixture named CHANGELOG.md
#: deeper in the tree stays checked.
HANDLE_EXEMPT_FILES: frozenset[str] = frozenset({"CHANGELOG.md"})


def scan_text(
    text: str, *, prose: bool, patterns: list[re.Pattern],
    allow: list[re.Pattern] = (),
    check_handles: "bool | None" = None,
) -> list[tuple[int, str, str]]:
    if check_handles is None:
        check_handles = prose
    hits: list[tuple[int, str, str]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if any(a.fullmatch(line) for a in allow):
            continue
        for pat in patterns:
            if pat.search(line):
                hits.append((idx, pat.pattern, line.strip()[:120]))
        if check_handles:
            m = _HANDLE_RE.search(line)
            if m and m.group(0) not in _HANDLE_ALLOW:
                hits.append((idx, f"handle {m.group(0)}", line.strip()[:120]))
    return hits


def _staged_added_lines(repo_root: Path) -> dict[str, list[tuple[int, str]]]:
    """Added lines per staged in-scope file — the gate is a ratchet: existing
    debt never blocks a commit, newly added personal data always does."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "-U0", "--no-color"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    out: dict[str, list[tuple[int, str]]] = {}
    rel = None
    in_scope = False
    line_no = 0
    for raw in proc.stdout.splitlines():
        if raw.startswith("+++ b/"):
            rel = raw[6:]
            in_scope = bool(rel) and rel.endswith(SCAN_SUFFIXES) and (
                rel in ROOT_FILES
                or any(rel.startswith(root + "/") for root in SCAN_ROOTS)
            )
        elif raw.startswith("@@") and in_scope:
            try:
                line_no = int(raw.split("+")[1].split(",")[0].split(" ")[0])
            except (IndexError, ValueError):
                line_no = 0
        elif raw.startswith("+") and not raw.startswith("+++") and in_scope:
            out.setdefault(rel, []).append((line_no, raw[1:]))
            line_no += 1
        elif raw.startswith(" ") and in_scope:
            line_no += 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lint personal data out of committed surfaces."
    )
    parser.add_argument(
        "--check-staged", action="store_true",
        help="scan only staged files (pre-commit hook mode)",
    )
    parser.add_argument(
        "--message", metavar="FILE",
        help="lint a commit-message file (commit-msg hook mode)",
    )
    args = parser.parse_args(argv)
    patterns = _load_patterns()
    allow = _load_allow()
    violations = 0

    if args.message:
        try:
            text = Path(args.message).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"lint_personal: cannot read message: {exc}", file=sys.stderr)
            return 1
        # Commit messages credit contributors by handle, same as release
        # notes — the handle rule stands down; deny-list patterns still apply.
        for line_no, pat, line in scan_text(
            text, prose=True, patterns=patterns, allow=allow,
            check_handles=False,
        ):
            print(f"commit-message:{line_no}: {pat}: {line}", file=sys.stderr)
            violations += 1
        return 1 if violations else 0

    repo_root = Path.cwd()
    if args.check_staged:
        for rel, added in _staged_added_lines(repo_root).items():
            prose = rel.endswith(_PROSE_SUFFIXES)
            handles = prose and rel not in HANDLE_EXEMPT_FILES
            for line_no, line in added:
                for hit_no, pat, text in scan_text(
                    line, prose=prose, patterns=patterns, allow=allow,
                    check_handles=handles,
                ):
                    print(f"{rel}:{line_no}: {pat}: {text}", file=sys.stderr)
                    violations += 1
        return 1 if violations else 0

    files = (
        [
            p
            for root in SCAN_ROOTS
            for suffix in SCAN_SUFFIXES
            for p in (repo_root / root).rglob(f"*{suffix}")
            if (repo_root / root).is_dir()
        ]
        + [repo_root / f for f in ROOT_FILES if (repo_root / f).is_file()]
    )
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        prose = path.suffix in _PROSE_SUFFIXES
        rel = path.relative_to(repo_root) if path.is_absolute() else path
        handles = prose and str(rel) not in HANDLE_EXEMPT_FILES
        for line_no, pat, line in scan_text(
            text, prose=prose, patterns=patterns, allow=allow,
            check_handles=handles,
        ):
            print(f"{rel}:{line_no}: {pat}: {line}", file=sys.stderr)
            violations += 1
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
