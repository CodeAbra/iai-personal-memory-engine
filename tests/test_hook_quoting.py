from __future__ import annotations

import re
from pathlib import Path

from iai_mcp.cli import _hook_quoting


def test_quote_posix_wraps_a_space_bearing_path():
    value = "C:\\Program Files\\Git\\bin\\bash.exe"
    assert _hook_quoting.quote_posix(value) == "'" + value + "'"


def test_quote_posix_is_identity_on_a_space_free_path():
    value = "/usr/local/bin/bash"
    assert _hook_quoting.quote_posix(value) == value


def test_quote_powershell_wraps_a_space_bearing_path():
    value = "C:\\Program Files\\Git\\bin\\bash.exe"
    assert _hook_quoting.quote_powershell(value) == "'" + value + "'"


def test_quote_powershell_doubles_an_embedded_single_quote():
    value = "C:\\Users\\O'Brien\\bash.exe"
    assert _hook_quoting.quote_powershell(value) == "'C:\\Users\\O''Brien\\bash.exe'"


def test_resolve_bash_returns_the_finder_result_verbatim():
    win_path = "C:\\Program Files\\Git\\bin\\bash.exe"
    result = _hook_quoting.resolve_bash(which=lambda name: win_path)
    assert result == win_path


def test_resolve_bash_falls_back_to_the_bare_token_when_the_finder_misses():
    result = _hook_quoting.resolve_bash(which=lambda name: None)
    assert result == "bash"


def test_quoting_functions_are_not_idempotent():
    # A caller must apply the quoting exactly once -- a second pass on an
    # already-quoted value nests quotes and breaks the result.
    value = "a value with a space"
    once = _hook_quoting.quote_posix(value)
    twice = _hook_quoting.quote_posix(once)
    assert twice != once


def test_module_source_carries_no_dev_process_markers():
    source = Path(_hook_quoting.__file__).read_text(encoding="utf-8")
    assert not re.search(r"#\d+\b", source)
    assert not re.search(r"\bWERG-\d+\b", source)
    assert not re.search(r"\bPhase\s+\d+\b", source, re.IGNORECASE)
    assert not re.search(r"\b\d{3}-\d{2}\b", source)
