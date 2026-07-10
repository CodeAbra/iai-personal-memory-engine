"""Structural guard for the AES boundary.

The storage engine (Rust crates + the legacy pure-Python engine) must only
ever see ciphertext BLOBs: no crypto capability may exist below the
MemoryStore encryption seam. This test makes the boundary a mechanism instead
of a convention — it scans the engine surfaces for crypto imports and crypto
crate dependencies and fails on any hit.

Detector functions are pure so the guard itself is testable against planted
synthetic violations (the tree stays clean).
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_ENGINE_CRATES = (
    "crates/lillibrain",
    "crates/lilliengine",
    "crates/lilli-hd",
    "crates/lilli-parity",
)

_ENGINE_PY_SURFACES = (
    "src/iai_mcp/lillibrain",
    "src/iai_mcp/runtime_graph_cache_worker.py",
)

_FORBIDDEN_CRATE_DEPS = re.compile(
    r"^\s*(aes|aes-gcm|aead|cipher|chacha20(poly1305)?|openssl|rsa|ring)\s*=",
    re.MULTILINE,
)

_FORBIDDEN_RUST_USE = re.compile(
    r"^\s*use\s+(aes|aes_gcm|aead|cipher|chacha20|openssl|rsa|ring)\b",
    re.MULTILINE,
)

_FORBIDDEN_PY_IMPORT = re.compile(
    r"^\s*(from\s+iai_mcp\.crypto\s+import|import\s+iai_mcp\.crypto"
    r"|from\s+iai_mcp\s+import\s+.*\bcrypto\b"
    r"|from\s+cryptography|import\s+cryptography)",
    re.MULTILINE,
)


def cargo_violations(text: str, origin: str) -> list[str]:
    return [f"{origin}: crate dep {m.group(1)!r}" for m in _FORBIDDEN_CRATE_DEPS.finditer(text)]


def rust_violations(text: str, origin: str) -> list[str]:
    return [f"{origin}: use {m.group(1)!r}" for m in _FORBIDDEN_RUST_USE.finditer(text)]


def python_violations(text: str, origin: str) -> list[str]:
    return [f"{origin}: {m.group(0).strip()!r}" for m in _FORBIDDEN_PY_IMPORT.finditer(text)]


def _scan_tree() -> list[str]:
    found: list[str] = []
    for crate in _ENGINE_CRATES:
        root = _REPO / crate
        manifest = root / "Cargo.toml"
        if manifest.is_file():
            found += cargo_violations(manifest.read_text(encoding="utf-8"), str(manifest))
        for rs in sorted(root.rglob("*.rs")):
            found += rust_violations(rs.read_text(encoding="utf-8"), str(rs))
    for surface in _ENGINE_PY_SURFACES:
        path = _REPO / surface
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for py in files:
            if py.is_file():
                found += python_violations(py.read_text(encoding="utf-8"), str(py))
    return found


def test_engine_surfaces_carry_no_crypto_capability():
    violations = _scan_tree()
    assert not violations, (
        "AES boundary breach — crypto capability below the MemoryStore "
        f"encryption seam: {violations}"
    )


def test_guard_detects_planted_cargo_dep():
    text = '[dependencies]\naes-gcm = "0.10"\nserde = "1"\n'
    assert cargo_violations(text, "synthetic"), "planted crate dep not detected"


def test_guard_detects_planted_rust_use():
    text = "use aes_gcm::Aes256Gcm;\nfn main() {}\n"
    assert rust_violations(text, "synthetic"), "planted use not detected"


def test_guard_detects_planted_python_import():
    for line in (
        "from iai_mcp.crypto import decrypt_field",
        "import iai_mcp.crypto",
        "from cryptography.hazmat.primitives.ciphers.aead import AESGCM",
    ):
        assert python_violations(line + "\n", "synthetic"), f"missed: {line}"


def test_guard_ignores_benign_mentions():
    benign_rust = "// the engine only ever sees ciphertext BLOBs\nfn f() {}\n"
    assert not rust_violations(benign_rust, "synthetic")
    benign_py = "x = 'ciphertext stays opaque here'\n"
    assert not python_violations(benign_py, "synthetic")
