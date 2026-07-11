#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

AGENTS_DIR = Path(".claude/agents")
_TOOLS_TOKEN = re.compile(r"(?<![A-Za-z_])(Grep|Glob)(?![A-Za-z_])")
_CALL_FORM = re.compile(r"(?<![A-Za-z_])(Grep|Glob)\s*\(")
_DIRECTIVE_MARKERS = (
    "Do NOT use Grep or Glob",
    "DO NOT use Grep or Glob",
    "DO NOT use Grep/Glob",
    "do not use Grep or Glob",
    "do NOT use Grep or Glob",
)


def _frontmatter_tools(text: str) -> str | None:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return None
        if lines[i].startswith("tools:"):
            return lines[i]
    return None


def _strip_directive_lines(text: str) -> str:
    return "\n".join(
        line for line in text.split("\n")
        if not any(marker in line for marker in _DIRECTIVE_MARKERS)
    )


def _violations_in(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    findings: list[str] = []

    tools_line = _frontmatter_tools(text)
    if tools_line and _TOOLS_TOKEN.search(tools_line):
        findings.append(
            f"{path}: frontmatter `tools:` line still lists Grep/Glob -> {tools_line.strip()!r}"
        )

    body_for_check = _strip_directive_lines(text)
    for match in _CALL_FORM.finditer(body_for_check):
        line_no = body_for_check.count("\n", 0, match.start()) + 1
        findings.append(
            f"{path}:{line_no}: inline `{match.group(1)}(...)` call form forbidden — "
            f"use ctx_search / ctx_execute_file instead"
        )
    return findings


def main(argv: list[str]) -> int:
    if not AGENTS_DIR.is_dir():
        print(f"lint_agent_prompts: {AGENTS_DIR} not present; skipping", file=sys.stderr)
        return 0

    files = sorted(AGENTS_DIR.glob("*.md"))
    if not files:
        print(f"lint_agent_prompts: no *.md files under {AGENTS_DIR}", file=sys.stderr)
        return 0

    all_findings: list[str] = []
    for p in files:
        all_findings.extend(_violations_in(p))

    if all_findings:
        print("lint_agent_prompts: violations found:", file=sys.stderr)
        for f in all_findings:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"lint_agent_prompts: {len(files)} agent prompt(s) clean", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
