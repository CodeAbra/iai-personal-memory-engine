"""Guard: the Rust SQL engine carries no cryptographic surface, and COUNT(*) is fast.

The engine reads and writes opaque blobs and is deliberately blind to their
meaning — all sealing happens above the storage seam in the host. This scans
every ``crates/lilliengine/src/`` source for any cryptographic identifier and
fails on a single hit, so a sealing symbol can never silently land in the engine
(control by absence). It also locks the COUNT(*) leaf-cell fast path against
regression through the Rust engine connection.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


def _repo_root() -> Path:
    # tests/ sits directly under the repo root.
    return Path(__file__).resolve().parent.parent


# Cryptographic tokens that must never appear in the engine source. The bare word
# "key" is intentionally excluded — it is a legitimate engine term (record keys,
# conflict keys, key ordering); only cryptographic compounds and primitives are
# forbidden. Matched as whole identifiers so e.g. "decrypt" is caught but an
# unrelated substring is not over-matched.
_FORBIDDEN = [
    "aes",
    "gcm",
    "cipher",
    "encrypt",
    "decrypt",
    "crypto",
    "nonce",
    "openssl",
    "aead",
    "key_material",
    "keymaterial",
    "aes_key",
    # Common Rust crypto crate names that would signal a sealing dependency.
    "ring",
    "chacha",
    "poly1305",
]


def _word_pattern() -> re.Pattern:
    # Word-boundary match so 'ring' does not fire on 'string'/'ordering' and
    # 'aead' does not fire inside a longer identifier.
    alternation = "|".join(re.escape(tok) for tok in _FORBIDDEN)
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def test_engine_source_has_no_crypto_symbol() -> None:
    """No crypto/key-material identifier appears anywhere under lilliengine/src/."""
    src = _repo_root() / "crates" / "lilliengine" / "src"
    assert src.is_dir(), f"expected the engine source dir at {src}"

    files = sorted(src.rglob("*.rs"))
    assert files, "expected at least one .rs source under crates/lilliengine/src/"

    pattern = _word_pattern()
    hits: list[str] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            m = pattern.search(line)
            if m:
                hits.append(f"{file}:{lineno}: forbidden crypto token {m.group(1)!r} -> {line.strip()}")

    assert not hits, (
        "the SQL engine must carry no cryptographic surface (control by absence); found:\n"
        + "\n".join(hits)
    )


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)
def test_count_star_fast_path_through_engine(tmp_path) -> None:
    """COUNT(*) through the Rust engine returns the exact row total on a large table.

    The leaf-cell COUNT(*) fast path is locked against regression: the bare count
    must equal the inserted row total. Routed through ``engine.Connection`` under
    the Rust-class guard so a stale wheel skips loudly.
    """
    from iai_mcp_native import engine

    conn = engine.Connection.open(str(tmp_path / "count.db"), 384)
    assert isinstance(conn, engine.Connection)
    conn.execute("CREATE TABLE t (id TEXT NOT NULL, n INTEGER)")
    total = 4000
    conn.execute("BEGIN")
    for i in range(total):
        conn.execute("INSERT INTO t (id, n) VALUES (?, ?)", (f"id-{i}", i))
    conn.execute("COMMIT")

    got = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert got == total
    assert isinstance(got, int)
