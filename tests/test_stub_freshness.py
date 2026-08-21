"""Guard against drift across the manually-synced native-extension `.pyi` stubs.

Stub generation is a 3-stage manual pipeline: `cargo run --bin stub_gen -p
iai_mcp_native` writes into a gitignored staging dir, a dev hand-flattens
into the tracked `rust/iai_mcp_native/iai_mcp_native/*.pyi` copies, then
hand-mirrors into `src/*.pyi`. Nothing else enforces the three stay equal or
current against the Rust source -- a stale-but-syntactically-valid stub would
ship silently.

Always-on checks are cheap (file reads only, no cargo). The regen-diff check
is opt-in (`IAI_MCP_STUB_GATE`) since it invokes `cargo run`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_RUST_STUB_DIR = _REPO_ROOT / "rust" / "iai_mcp_native" / "iai_mcp_native"
_SRC_STUB_DIR = _REPO_ROOT / "src"
_STAGED_STUB_DIR = _REPO_ROOT / "rust" / "iai_mcp_native" / "stubs" / "iai_mcp_native"

# (logical name, tracked rust copy, tracked src mirror, staged regen output)
_STUB_TRIPLES = [
    (
        "__init__",
        _RUST_STUB_DIR / "__init__.pyi",
        _SRC_STUB_DIR / "iai_mcp_native.pyi",
        _STAGED_STUB_DIR / "__init__.pyi",
    ),
    (
        "embed",
        _RUST_STUB_DIR / "embed.pyi",
        _SRC_STUB_DIR / "embed.pyi",
        _STAGED_STUB_DIR / "embed" / "__init__.pyi",
    ),
    (
        "graph",
        _RUST_STUB_DIR / "graph.pyi",
        _SRC_STUB_DIR / "graph.pyi",
        _STAGED_STUB_DIR / "graph" / "__init__.pyi",
    ),
]

_EXPECTED_SYMBOLS = {
    "__init__": ["embed", "graph"],
    "embed": ["Embedder"],
    "graph": ["answer"],
}


@pytest.mark.parametrize(
    "name,rust_copy,src_copy,_staged", _STUB_TRIPLES, ids=[t[0] for t in _STUB_TRIPLES]
)
def test_tracked_stub_copies_are_nonempty(name, rust_copy, src_copy, _staged):
    for copy in (rust_copy, src_copy):
        assert copy.exists(), f"{name}: tracked stub missing at {copy}"
        assert copy.stat().st_size > 0, f"{name}: tracked stub is empty at {copy}"


@pytest.mark.parametrize(
    "name,rust_copy,src_copy,_staged", _STUB_TRIPLES, ids=[t[0] for t in _STUB_TRIPLES]
)
def test_tracked_stub_copies_carry_expected_symbols(name, rust_copy, src_copy, _staged):
    for copy in (rust_copy, src_copy):
        text = copy.read_text()
        for symbol in _EXPECTED_SYMBOLS[name]:
            assert symbol in text, (
                f"{name}: expected symbol {symbol!r} missing from {copy}"
            )


@pytest.mark.parametrize(
    "name,rust_copy,src_copy,_staged", _STUB_TRIPLES, ids=[t[0] for t in _STUB_TRIPLES]
)
def test_rust_and_src_stub_copies_match(name, rust_copy, src_copy, _staged):
    """The rust/ and src/ tracked mirrors must be byte-identical.

    Catches the "flattened into rust/ but forgot src/" half of the drift
    failure scenario without needing a fresh cargo regen.
    """
    assert rust_copy.read_text() == src_copy.read_text(), (
        f"{name}: rust/ and src/ tracked stub copies differ -- "
        f"re-flatten the same staged regen output into both."
    )


# ---------------------------------------------------------------------------
# Regen-diff (opt-in): catches "forgot to regenerate at all" against the
# live Rust source. Heavy relative to the two checks above (invokes cargo).
# ---------------------------------------------------------------------------
_STUB_GATE_ENABLED = bool(os.environ.get("IAI_MCP_STUB_GATE"))


@pytest.mark.skipif(
    not _STUB_GATE_ENABLED,
    reason="stub regen gate is opt-in: set IAI_MCP_STUB_GATE (invokes cargo run)",
)
def test_tracked_stubs_match_fresh_regen():
    if shutil.which("cargo") is None:
        pytest.skip("cargo not available in this environment")

    result = subprocess.run(
        ["cargo", "run", "--bin", "stub_gen", "-p", "iai_mcp_native"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"stub_gen failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    stale = []
    for name, rust_copy, src_copy, staged in _STUB_TRIPLES:
        assert staged.exists(), f"{name}: stub_gen did not produce {staged}"
        fresh = staged.read_text()
        for copy in (rust_copy, src_copy):
            if copy.read_text() != fresh:
                stale.append(str(copy))

    assert not stale, (
        "tracked stub(s) are stale against a fresh regen -- re-flatten "
        f"from {_STAGED_STUB_DIR}: {stale}"
    )
