"""Path/interpreter quoting for hook command strings.

Used by cli/_capture.py and cli/_antigravity_hooks.py to quote the
interpreter and script-path tokens interpolated into a hook command string.
Apply each function exactly once per value -- neither is idempotent, so a
second pass nests quotes and breaks the result.
"""
from __future__ import annotations

import shlex
import shutil
from typing import Callable, Optional


def quote_posix(value: str) -> str:
    return shlex.quote(value)


def quote_powershell(value: str) -> str:
    # PowerShell single-quoted strings treat everything literally except a
    # doubled single quote, which is how a literal ' is embedded. No stdlib
    # PowerShell-quoting primitive exists -- subprocess.list2cmdline targets
    # cmd.exe's argv convention, not PowerShell's own parser.
    return "'" + value.replace("'", "''") + "'"


def resolve_bash(which: Optional[Callable[[str], Optional[str]]] = None) -> str:
    finder = which or shutil.which
    return finder("bash") or "bash"
