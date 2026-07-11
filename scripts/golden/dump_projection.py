"""Dump the frozen projection matrix P[384, 10000] as a raw little-endian golden.

Run under the project virtualenv:

    source .venv/bin/activate
    python scripts/golden/dump_projection.py

P is regenerated deterministically at import; this dumper re-asserts its locked
sha256 before writing so a drift fails loudly instead of overwriting the golden.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

# Make both the source tree and the local script dir importable before any
# project import, so the dumper runs from a checkout without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "golden"))

from iai_mcp.lilli.core.projection import P, P_SHA256_HASH  # noqa: E402
from _common import write_golden  # noqa: E402

SOURCE = "src/iai_mcp/lilli/core/projection.py"
GENERATED_BY = "scripts/golden/dump_projection.py"
OUT_STEM = _REPO_ROOT / "fixtures" / "golden" / "hdc" / "projection"

EXPECTED_LEN = 384 * 10000 * 4  # 15,360,000


def main() -> None:
    assert P.shape == (384, 10000), f"P shape mismatch: {P.shape}"
    assert P.dtype == np.float32, f"P dtype mismatch: {P.dtype}"
    assert P.flags["C_CONTIGUOUS"], "P must be C-contiguous"

    # Force little-endian, C-contiguous f32 bytes.
    buf = np.ascontiguousarray(P).astype("<f4", copy=False).tobytes()

    assert len(buf) == EXPECTED_LEN, f"byte length mismatch: {len(buf)} != {EXPECTED_LEN}"
    digest = hashlib.sha256(buf).hexdigest()
    assert digest == P_SHA256_HASH, (
        f"projection golden sha256 drifted: {digest} != {P_SHA256_HASH}"
    )

    written = write_golden(
        path_stem=OUT_STEM,
        arr_bytes=buf,
        dtype="f32",
        shape=[384, 10000],
        byte_order="le",
        subsystem="hdc",
        source=SOURCE,
        generated_by=GENERATED_BY,
    )
    assert written == P_SHA256_HASH, "written digest disagrees with locked hash"
    print(f"wrote {OUT_STEM}.bin ({len(buf)} bytes) sha256={written}")


if __name__ == "__main__":
    main()
