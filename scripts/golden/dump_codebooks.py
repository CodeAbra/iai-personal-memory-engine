"""Dump the frozen HDC role + filler codebooks as raw little-endian goldens.

Run under the project virtualenv:

    source .venv/bin/activate
    python scripts/golden/dump_codebooks.py

NumPy's PCG64 + ``integers``/``choice`` draw orders are not reproducible
bit-for-bit outside NumPy, so every RNG-derived codebook vector is frozen here
from the Python reference and asserted against by the Rust kernels. Each artifact
is re-asserted (shape/length) against the live reference before writing so a drift
fails loudly instead of silently overwriting a golden. Re-running reproduces
byte-identical files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make both the source tree and the local script dir importable before any
# project import, so the dumper runs from a checkout without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "golden"))

from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa  # noqa: E402
from _common import write_golden  # noqa: E402

GENERATED_BY = "scripts/golden/dump_codebooks.py"
OUT_DIR = _REPO_ROOT / "fixtures" / "golden" / "hdc"
SAMPLES_PATH = OUT_DIR / "sample_strings.json"

BSC_D_SMALL = 4096
BSC_D_LARGE = 10000
BSC_BYTES_SMALL = BSC_D_SMALL // 8  # 512
BSC_BYTES_LARGE = BSC_D_LARGE // 8  # 1250
FHRR_BYTES = fhrr.LILLI_FHRR_DIM  # 10000
SPARSE_PACKED_BYTES = sparse_vsa.SPARSE_K * 2  # 40

ROLES = bsc.BSC_ROLE_VOCABULARY


def _load_samples() -> list[str]:
    return json.loads(SAMPLES_PATH.read_text())


def _stack(rows: list[bytes], elem_len: int) -> bytes:
    """Concatenate fixed-length byte rows after asserting each row's length."""
    for i, row in enumerate(rows):
        assert len(row) == elem_len, f"row {i} length {len(row)} != {elem_len}"
    return b"".join(rows)


def main() -> None:
    samples = _load_samples()
    assert len(samples) >= 500, f"expected >=500 sample strings, got {len(samples)}"
    assert len(ROLES) == 18, f"expected 18 role names, got {len(ROLES)}"

    # --- BSC role codebooks, frozen per dimension (never a flat concatenated blob).
    bsc_4096 = _stack(
        [bsc.role_hv(r, D=BSC_D_SMALL) for r in ROLES], BSC_BYTES_SMALL
    )
    write_golden(
        path_stem=OUT_DIR / "bsc_roles_4096",
        arr_bytes=bsc_4096,
        dtype="u8",
        shape=[len(ROLES), BSC_BYTES_SMALL],
        byte_order="le",
        subsystem="hdc",
        source="src/iai_mcp/lilli/tiers/bsc.py role_hv(role, D=4096)",
        generated_by=GENERATED_BY,
    )

    bsc_10000 = _stack(
        [bsc.role_hv(r, D=BSC_D_LARGE) for r in ROLES], BSC_BYTES_LARGE
    )
    write_golden(
        path_stem=OUT_DIR / "bsc_roles_10000",
        arr_bytes=bsc_10000,
        dtype="u8",
        shape=[len(ROLES), BSC_BYTES_LARGE],
        byte_order="le",
        subsystem="hdc",
        source="src/iai_mcp/lilli/tiers/bsc.py role_hv(role, D=10000)",
        generated_by=GENERATED_BY,
    )

    # --- BSC filler sample (open-vocab parity gate) at D=4096.
    bsc_fillers = _stack(
        [bsc.filler_hv(s, D=BSC_D_SMALL) for s in samples], BSC_BYTES_SMALL
    )
    write_golden(
        path_stem=OUT_DIR / "bsc_fillers_sample",
        arr_bytes=bsc_fillers,
        dtype="u8",
        shape=[len(samples), BSC_BYTES_SMALL],
        byte_order="le",
        subsystem="hdc",
        source="src/iai_mcp/lilli/tiers/bsc.py filler_hv(s, D=4096) over sample_strings.json",
        generated_by=GENERATED_BY,
    )

    # --- FHRR role + filler codebooks (D=10000, uint8 phases).
    fhrr_roles = _stack([fhrr.role_hv(r) for r in ROLES], FHRR_BYTES)
    write_golden(
        path_stem=OUT_DIR / "fhrr_roles",
        arr_bytes=fhrr_roles,
        dtype="u8",
        shape=[len(ROLES), FHRR_BYTES],
        byte_order="le",
        subsystem="hdc",
        source="src/iai_mcp/lilli/tiers/fhrr.py role_hv(role)",
        generated_by=GENERATED_BY,
    )

    fhrr_fillers = _stack([fhrr.filler_hv(s) for s in samples], FHRR_BYTES)
    write_golden(
        path_stem=OUT_DIR / "fhrr_fillers_sample",
        arr_bytes=fhrr_fillers,
        dtype="u8",
        shape=[len(samples), FHRR_BYTES],
        byte_order="le",
        subsystem="hdc",
        source="src/iai_mcp/lilli/tiers/fhrr.py filler_hv(s) over sample_strings.json",
        generated_by=GENERATED_BY,
    )

    # --- Sparse role + filler codebooks, packed to 40 bytes via pack_indices.
    sparse_roles = _stack(
        [sparse_vsa.pack_indices(sparse_vsa.role_hv(r)) for r in ROLES],
        SPARSE_PACKED_BYTES,
    )
    write_golden(
        path_stem=OUT_DIR / "sparse_roles",
        arr_bytes=sparse_roles,
        dtype="u8",
        shape=[len(ROLES), SPARSE_PACKED_BYTES],
        byte_order="le",
        subsystem="hdc",
        source="src/iai_mcp/lilli/tiers/sparse_vsa.py pack_indices(role_hv(role))",
        generated_by=GENERATED_BY,
    )

    sparse_fillers = _stack(
        [sparse_vsa.pack_indices(sparse_vsa.filler_hv(s)) for s in samples],
        SPARSE_PACKED_BYTES,
    )
    write_golden(
        path_stem=OUT_DIR / "sparse_fillers_sample",
        arr_bytes=sparse_fillers,
        dtype="u8",
        shape=[len(samples), SPARSE_PACKED_BYTES],
        byte_order="le",
        subsystem="hdc",
        source="src/iai_mcp/lilli/tiers/sparse_vsa.py pack_indices(filler_hv(s)) over sample_strings.json",
        generated_by=GENERATED_BY,
    )

    print(f"dumped 7 codebook golden pairs into {OUT_DIR}")


if __name__ == "__main__":
    main()
