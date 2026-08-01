"""Verification of lilli.core primitives: deterministic seeding, hypervector generation, the SHA256-locked projection matrix, and tier-agnostic similarity functions.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from iai_mcp.lilli.core.seed import hv_from_seed, seed_from_str
from iai_mcp.lilli.core.projection import P, P_SHA256_HASH, project
from iai_mcp.lilli.core.similarity import cosine_packed, hamming, jaccard

# seed.py tests


def test_seed_from_str_deterministic() -> None:
    """Same (prefix, value) always returns the same seed."""
    s1 = seed_from_str("a", "b")
    s2 = seed_from_str("a", "b")
    assert s1 == s2


def test_seed_from_str_cross_process_stable() -> None:
    """seed_from_str result is identical across two independent subprocesses."""
    cmd = [
        sys.executable,
        "-c",
        "from iai_mcp.lilli.core.seed import seed_from_str; print(seed_from_str('iai', 'mcp'))",
    ]
    r1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    r2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert r1.stdout.strip() == r2.stdout.strip()
    assert r1.stdout.strip() != ""


def test_hv_from_seed_length_4096() -> None:
    """D=4096 produces exactly 512 bytes (ceil(4096/8) = 512)."""
    assert len(hv_from_seed(0, 4096)) == 512


def test_hv_from_seed_length_10000() -> None:
    """D=10000 produces exactly 1250 bytes (ceil(10000/8) = 1250)."""
    assert len(hv_from_seed(0, 10000)) == 1250


def test_hv_from_seed_length_2048() -> None:
    """D=2048 produces exactly 256 bytes (ceil(2048/8) = 256)."""
    assert len(hv_from_seed(0, 2048)) == 256


def test_hv_from_seed_deterministic() -> None:
    """Same (seed, D) always produces identical packed bytes."""
    hv1 = hv_from_seed(42, 4096)
    hv2 = hv_from_seed(42, 4096)
    assert hv1 == hv2


def test_hv_from_seed_rejects_zero_D() -> None:
    """D=0 raises ValueError."""
    with pytest.raises(ValueError):
        hv_from_seed(0, 0)


# projection.py tests


def test_projection_matrix_is_locked() -> None:
    """P has the expected shape and dtype."""
    assert P.shape == (384, 10000)
    assert P.dtype.name == "float32"


def test_projection_matrix_cross_process_stable() -> None:
    """The P matrix SHA256 hash is identical across two independent subprocesses."""
    cmd = [
        sys.executable,
        "-c",
        (
            "from iai_mcp.lilli.core.projection import P; "
            "import hashlib; "
            "print(hashlib.sha256(P.tobytes()).hexdigest())"
        ),
    ]
    r1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    r2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    h1 = r1.stdout.strip()
    h2 = r2.stdout.strip()
    assert h1 == h2, f"P hash mismatch across processes: {h1} vs {h2}"
    assert len(h1) == 64, f"Expected 64-hex digest, got len={len(h1)}"


def test_projection_matrix_full_sha256_locked() -> None:
    """The full P.tobytes() SHA256 digest matches the locked constant.
    """
    actual = hashlib.sha256(P.tobytes()).hexdigest()
    assert P_SHA256_HASH != "BOOTSTRAP_PENDING", (
        "P_SHA256_HASH is still BOOTSTRAP_PENDING -- bootstrap procedure was not completed"
    )
    assert len(P_SHA256_HASH) == 64, (
        f"P_SHA256_HASH must be a 64-hex-char string, got len={len(P_SHA256_HASH)}"
    )
    assert actual == P_SHA256_HASH, (
        f"P matrix has drifted: "
        f"expected {P_SHA256_HASH}, got {actual}"
    )


def test_projection_dot_product() -> None:
    """project(zeros(384)) returns shape (10000,) of zeros."""
    emb = np.zeros(384, dtype=np.float32)
    result = project(emb)
    assert result.shape == (10000,)
    assert result.dtype.name == "float32"
    np.testing.assert_array_equal(result, np.zeros(10000, dtype=np.float32))


def test_projection_rejects_wrong_shape() -> None:
    """project() raises ValueError on wrong embedding shape."""
    with pytest.raises(ValueError):
        project(np.zeros(383, dtype=np.float32))


# similarity.py tests


def test_similarity_hamming_identical() -> None:
    """Identical buffers have Hamming distance 0.0."""
    assert hamming(b"\x00" * 125, b"\x00" * 125) == 0.0


def test_similarity_hamming_opposite() -> None:
    """All-ones vs all-zeros buffers have Hamming distance 1.0."""
    assert hamming(b"\xff" * 125, b"\x00" * 125) == 1.0


def test_similarity_cosine_packed_identical() -> None:
    """Identical 1250-byte buffers have cosine similarity 1.0 (within 1e-6)."""
    buf = np.random.default_rng(7).integers(0, 256, size=1250, dtype=np.uint8).tobytes()
    result = cosine_packed(buf, buf)
    assert abs(result - 1.0) < 1e-6, f"Expected ~1.0, got {result}"


def test_similarity_jaccard_basic() -> None:
    """Jaccard of {1,2,3} and {2,3,4} is 0.5."""
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5


def test_similarity_jaccard_empty() -> None:
    """Jaccard of two empty sets is 0.0."""
    assert jaccard(set(), set()) == 0.0


# Public-repo-clean token guard


def test_lilli_package_no_forbidden_tokens() -> None:
    """No file under src/iai_mcp/lilli/ contains forbidden internal markers.

    Two-stage guard:
    1. Plain literal tokens the generic regex cannot express.
    2. Generic plan/phase/decision-code regex with word boundaries so that
       runtime registry keys like AUTIST-01 and MCP-12 do NOT trip it.
    """
    import re

    # Specific tokens (literals the regex cannot express)
    _PLAIN_TOKENS = [
        "LILLIHD-",
        "D-TEM",
        "CONN-",
        "MEM-0",
    ]
    # Generic plan/phase/decision-code patterns
    _PATTERNS = re.compile(
        r"\bPlan\s+\d+\b|\bPhase\s+\d+\b|\bD-\d+\b"
    )
    # __file__ is now tests/lilli/test_core.py; project root is three levels up
    lilli_root = (
        pathlib.Path(__file__).parent.parent.parent / "src" / "iai_mcp" / "lilli"
    )
    assert lilli_root.is_dir(), f"lilli/ package directory not found: {lilli_root}"
    violations: list[str] = []
    for py_file in lilli_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for token in _PLAIN_TOKENS:
            if token in text:
                violations.append(f"{py_file}: plain token {token!r}")
        if _PATTERNS.search(text):
            for lineno, line in enumerate(text.splitlines(), start=1):
                if _PATTERNS.search(line):
                    violations.append(f"{py_file}:{lineno}: {line.rstrip()!r}")
    assert not violations, (
        "Forbidden tokens found in lilli/ source (public-repo-clean violation):\n"
        + "\n".join(violations)
    )
