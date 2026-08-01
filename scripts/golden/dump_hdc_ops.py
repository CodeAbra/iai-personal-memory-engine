"""Dump the 100-embedding projection golden + per-tier op-result tables.

Run under the project virtualenv:

    source .venv/bin/activate
    python scripts/golden/dump_hdc_ops.py

Two classes of golden are frozen here from the Python reference:

  * The projection-apply gate: 100 fixed f32 input embeddings (proj_100_in,
    shape [100,384]) and their bit-packed projection outputs (proj_100, shape
    [100,1250], 10000 bits MSB-first). The exact f32 inputs are fed to the Rust
    ``project`` and the exact 1250-byte outputs are asserted bit-identical.

  * Per-tier op tables: byte-valued results (bind/unbind/bundle/permute) in
    ``<tier>_ops`` and their float similarity siblings (hamming/cosine,
    cosine-mean, jaccard) in ``<tier>_ops_sims`` as little-endian f64.

The op input sequence + result order is documented below and echoed into each
manifest ``source`` so the Rust differential replays the identical sequence.
Each result is re-asserted (length/shape) against the live reference before
write_golden. Re-running reproduces byte-identical files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make both the source tree and the local script dir importable before any
# project import, so the dumper runs from a checkout without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "golden"))

from iai_mcp.lilli.core import similarity as sim  # noqa: E402
from iai_mcp.lilli.crossmodal.embed_to_hv import from_embedding  # noqa: E402
from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa  # noqa: E402
from _common import write_golden  # noqa: E402

GENERATED_BY = "scripts/golden/dump_hdc_ops.py"
OUT_DIR = _REPO_ROOT / "fixtures" / "golden" / "hdc"
SAMPLES_PATH = OUT_DIR / "sample_strings.json"

EMBED_DIM = 384
HV_BYTES = 10000 // 8  # 1250 — packbits of 10000 projected bits
N_PROJ = 100
PROJ_DUMP_SEED = 0  # fixed dump-time RNG seed for the 100 input embeddings (dump-time only, not the locked projection)

BSC_D = 4096
ROLES = bsc.BSC_ROLE_VOCABULARY


def _f64_bytes(values: list[float]) -> bytes:
    arr = np.asarray(values, dtype="<f8")
    return np.ascontiguousarray(arr).tobytes()


def _load_samples() -> list[str]:
    return json.loads(SAMPLES_PATH.read_text())


# --------------------------------------------------------------------------- #
# Projection-apply gate: proj_100_in (f32 [100,384]) + proj_100 (u8 [100,1250])
# --------------------------------------------------------------------------- #
def dump_projection_pair() -> None:
    rng = np.random.default_rng(PROJ_DUMP_SEED)
    embs = rng.standard_normal((N_PROJ, EMBED_DIM)).astype(np.float32)
    assert embs.shape == (N_PROJ, EMBED_DIM) and embs.dtype == np.float32

    in_bytes = np.ascontiguousarray(embs).astype("<f4", copy=False).tobytes()
    assert len(in_bytes) == N_PROJ * EMBED_DIM * 4
    write_golden(
        path_stem=OUT_DIR / "proj_100_in",
        arr_bytes=in_bytes,
        dtype="f32",
        shape=[N_PROJ, EMBED_DIM],
        byte_order="le",
        subsystem="hdc",
        source=(
            "default_rng(0).standard_normal((100,384)).astype(float32) — "
            "the exact inputs fed to project()"
        ),
        generated_by=GENERATED_BY,
    )

    out_rows: list[bytes] = []
    for i in range(N_PROJ):
        hv = from_embedding(embs[i].tolist())
        assert len(hv) == HV_BYTES, f"proj row {i} length {len(hv)} != {HV_BYTES}"
        out_rows.append(hv)
    out_bytes = b"".join(out_rows)
    write_golden(
        path_stem=OUT_DIR / "proj_100",
        arr_bytes=out_bytes,
        dtype="u8",
        shape=[N_PROJ, HV_BYTES],
        byte_order="le",
        subsystem="hdc",
        source=(
            "src/iai_mcp/lilli/crossmodal/embed_to_hv.py from_embedding(proj_100_in[i]) — "
            "packbits MSB-first of (emb@P >= 0), 1250 bytes per row"
        ),
        generated_by=GENERATED_BY,
    )


# --------------------------------------------------------------------------- #
# BSC op table (D=4096)
# --------------------------------------------------------------------------- #
def dump_bsc_ops(samples: list[str]) -> None:
    # Fixed input set: the 18 roles + the first 20 sample fillers.
    fillers = [bsc.filler_hv(s, D=BSC_D) for s in samples[:20]]
    role_names = list(ROLES)

    byte_rows: list[bytes] = []
    sims: list[float] = []

    # 1) bind(role_hv(role_i), filler_i) for the first 18 (role, filler) pairs.
    bound_list: list[bytes] = []
    for i, rn in enumerate(role_names):
        f = fillers[i % len(fillers)]
        b = bsc.bind(bsc.role_hv(rn, D=BSC_D), f)
        assert len(b) == BSC_D // 8
        byte_rows.append(b)
        bound_list.append(b)

    # 2) bundle of (role, filler) pairs at 3, 7, 10 pairs (all within cap=10).
    for n in (3, 7, 10):
        pairs = [(role_names[i], fillers[i % len(fillers)]) for i in range(n)]
        bundled = bsc.bundle(pairs, D=BSC_D)
        assert len(bundled) == BSC_D // 8
        byte_rows.append(bundled)

    # 3) permute(bound_list[0], shift) for a few shifts (bit-level roll).
    for shift in (1, 7, 33, BSC_D - 1):
        p = bsc.permute(bound_list[0], shift)
        assert len(p) == BSC_D // 8
        byte_rows.append(p)

    # Float siblings: hamming + cosine_packed over consecutive bound pairs.
    for i in range(len(bound_list) - 1):
        sims.append(sim.hamming(bound_list[i], bound_list[i + 1]))
        sims.append(sim.cosine_packed(bound_list[i], bound_list[i + 1]))

    blob = b"".join(byte_rows)
    write_golden(
        path_stem=OUT_DIR / "bsc_ops",
        arr_bytes=blob,
        dtype="u8",
        shape=[len(byte_rows), BSC_D // 8],
        byte_order="le",
        subsystem="hdc",
        source=(
            "src/iai_mcp/lilli/tiers/bsc.py D=4096 op table: "
            "rows[0:18]=bind(role_i, filler_{i%20}); "
            "rows[18:21]=bundle(first n pairs) for n in (3,7,10); "
            "rows[21:25]=permute(rows[0], shift) for shift in (1,7,33,4095)"
        ),
        generated_by=GENERATED_BY,
    )
    write_golden(
        path_stem=OUT_DIR / "bsc_ops_sims",
        arr_bytes=_f64_bytes(sims),
        dtype="f64",
        shape=[len(sims)],
        byte_order="le",
        subsystem="hdc",
        source=(
            "src/iai_mcp/lilli/core/similarity.py over consecutive bsc_ops bind rows: "
            "interleaved [hamming, cosine_packed] for i in range(17)"
        ),
        generated_by=GENERATED_BY,
    )


# --------------------------------------------------------------------------- #
# BSC bundle at D=10000 (the structure-HV dimension). 10000 bits = 1250 bytes =
# 156 u64 words + a 2-byte tail, so this is the ONLY parity case that exercises
# the scalar tail-byte vote of the SWAR bundle (the D=4096 table is word-aligned,
# tail empty). Each bundled input is bind(role_i, role_{i+1}) — distinct packed
# HVs that both sides reproduce from the frozen role golden with no RNG-drawn
# filler, so the Rust differential replays the identical inputs.
# --------------------------------------------------------------------------- #
BSC_STRUCTURE_D = 10000


def dump_bsc_bundle_10000() -> None:
    role_names = list(ROLES)
    n_roles = len(role_names)
    # pairs[i] = (role_i, filler=role_hv(role_{i+1})) -> bind = role_i XOR role_{i+1}.
    pairs_all = [
        (role_names[i], bsc.role_hv(role_names[(i + 1) % n_roles], D=BSC_STRUCTURE_D))
        for i in range(n_roles)
    ]

    byte_rows: list[bytes] = []
    for n in (3, 7, 10):
        bundled = bsc.bundle(pairs_all[:n], D=BSC_STRUCTURE_D)
        assert len(bundled) == BSC_STRUCTURE_D // 8
        byte_rows.append(bundled)

    blob = b"".join(byte_rows)
    write_golden(
        path_stem=OUT_DIR / "bsc_bundle_10000",
        arr_bytes=blob,
        dtype="u8",
        shape=[len(byte_rows), BSC_STRUCTURE_D // 8],
        byte_order="le",
        subsystem="hdc",
        source=(
            "src/iai_mcp/lilli/tiers/bsc.py D=10000 bundle table (1250 bytes/row, "
            "2-byte tail): bound_i = bind(role_i, role_{i+1}); "
            "rows[0:3]=bundle(bound[:n]) for n in (3,7,10). "
            "Exercises the scalar tail-byte vote of the SWAR bundle."
        ),
        generated_by=GENERATED_BY,
    )


# --------------------------------------------------------------------------- #
# FHRR op table (D=10000)
# --------------------------------------------------------------------------- #
def dump_fhrr_ops(samples: list[str]) -> None:
    role_names = list(ROLES)
    fillers = [fhrr.filler_hv(s) for s in samples[:10]]
    roles = [fhrr.role_hv(rn) for rn in role_names[:10]]

    byte_rows: list[bytes] = []
    sims: list[float] = []

    # 1) bind(role_i, filler_i) for 10 pairs.
    bound: list[bytes] = []
    for i in range(10):
        b = fhrr.bind(roles[i], fillers[i])
        assert len(b) == fhrr.LILLI_FHRR_DIM
        byte_rows.append(b)
        bound.append(b)

    # 2) unbind(bound_i, role_i) round-trips back toward filler_i.
    for i in range(10):
        u = fhrr.unbind(bound[i], roles[i])
        assert len(u) == fhrr.LILLI_FHRR_DIM
        byte_rows.append(u)

    # 3) bundle: single-HV passthrough case + multi-HV cases (2, 3, 5 HVs) that
    #    exercise the atan2 circular-mean quantization across the 256 phase bins.
    single = fhrr.bundle([bound[0]])
    assert single == bound[0], "single-HV bundle must passthrough unchanged"
    byte_rows.append(single)
    for n in (2, 3, 5):
        bl = fhrr.bundle(bound[:n])
        assert len(bl) == fhrr.LILLI_FHRR_DIM
        byte_rows.append(bl)

    # 4) permute (byte-roll) for a few shifts.
    for shift in (1, 13, fhrr.LILLI_FHRR_DIM - 1):
        p = fhrr.permute(bound[0], shift)
        assert len(p) == fhrr.LILLI_FHRR_DIM
        byte_rows.append(p)

    # Float siblings: similarity over consecutive bound pairs.
    for i in range(len(bound) - 1):
        sims.append(fhrr.similarity(bound[i], bound[i + 1]))

    blob = b"".join(byte_rows)
    write_golden(
        path_stem=OUT_DIR / "fhrr_ops",
        arr_bytes=blob,
        dtype="u8",
        shape=[len(byte_rows), fhrr.LILLI_FHRR_DIM],
        byte_order="le",
        subsystem="hdc",
        source=(
            "src/iai_mcp/lilli/tiers/fhrr.py D=10000 op table: "
            "rows[0:10]=bind(role_i, filler_i); rows[10:20]=unbind(bound_i, role_i); "
            "row[20]=bundle([bound_0]) (passthrough); rows[21:24]=bundle(bound[:n]) for n in (2,3,5); "
            "rows[24:27]=permute(bound_0, shift) for shift in (1,13,9999)"
        ),
        generated_by=GENERATED_BY,
    )
    write_golden(
        path_stem=OUT_DIR / "fhrr_ops_sims",
        arr_bytes=_f64_bytes(sims),
        dtype="f64",
        shape=[len(sims)],
        byte_order="le",
        subsystem="hdc",
        source=(
            "src/iai_mcp/lilli/tiers/fhrr.py similarity over consecutive fhrr_ops bind rows "
            "for i in range(9)"
        ),
        generated_by=GENERATED_BY,
    )


# --------------------------------------------------------------------------- #
# Sparse op table (packed 40-byte vectors). Inputs stay at or above SPARSE_K so
# the over-cap truncation path runs and the underflow _pad_to_k path NEVER does.
# --------------------------------------------------------------------------- #
def dump_sparse_ops(samples: list[str]) -> None:
    role_names = list(ROLES)
    a = sparse_vsa.role_hv(role_names[0])   # 20 indices
    b = sparse_vsa.role_hv(role_names[1])   # 20 indices
    c = sparse_vsa.role_hv(role_names[2])   # 20 indices
    fillers = [sparse_vsa.filler_hv(s) for s in samples[:6]]

    byte_rows: list[bytes] = []
    sims: list[float] = []

    def _pack(idx: list[int]) -> bytes:
        packed = sparse_vsa.pack_indices(idx)
        assert len(packed) == sparse_vsa.SPARSE_K * 2
        return packed

    # 1) bind: two distinct 20-index sets -> union >= 21 -> truncates to K (no pad).
    bind_ab = sparse_vsa.bind(a, b)
    assert len(bind_ab) == sparse_vsa.SPARSE_K, "bind must stay at K (no pad path)"
    byte_rows.append(_pack(bind_ab))
    bind_pairs = [(a, b), (a, c), (b, c)]
    for x, y in bind_pairs:
        r = sparse_vsa.bind(x, y)
        assert len(r) == sparse_vsa.SPARSE_K
        byte_rows.append(_pack(r))

    # 2) unbind: bound = union of three roles (60 distinct), key = one role.
    #    remainder (>= 40) truncates to K -> no _pad_to_k.
    union3 = sorted(set(a) | set(b) | set(c))
    for key in (a, b, c):
        u = sparse_vsa.unbind(union3, key)
        assert len(u) == sparse_vsa.SPARSE_K, "unbind must stay at K (no pad path)"
        byte_rows.append(_pack(u))

    # 3) bundle of several HVs (frequency vote, deterministic).
    for n in (2, 4, 6):
        bl = sparse_vsa.bundle([a, b, c, *fillers][:n])
        assert len(bl) <= sparse_vsa.SPARSE_K
        byte_rows.append(_pack(bl))

    # 4) permute (index shift modulo D).
    for shift in (1, 17, sparse_vsa.LILLI_SPARSE_DIM - 1):
        p = sparse_vsa.permute(a, shift)
        assert len(p) == sparse_vsa.SPARSE_K
        byte_rows.append(_pack(p))

    # Float siblings: jaccard similarity over a fixed set of pairs.
    for x, y in ((a, b), (a, c), (b, c), (a, a)):
        sims.append(sparse_vsa.similarity(x, y))

    blob = b"".join(byte_rows)
    write_golden(
        path_stem=OUT_DIR / "sparse_ops",
        arr_bytes=blob,
        dtype="u8",
        shape=[len(byte_rows), sparse_vsa.SPARSE_K * 2],
        byte_order="le",
        subsystem="hdc",
        source=(
            "src/iai_mcp/lilli/tiers/sparse_vsa.py packed op table "
            "(a,b,c = role_hv of roles[0:3]; fillers = filler_hv of samples[0:6]): "
            "rows[0:4]=bind[(a,b),(a,b),(a,c),(b,c)]; "
            "rows[4:7]=unbind(union(a,b,c), key) for key in (a,b,c); "
            "rows[7:10]=bundle([a,b,c,fillers][:n]) for n in (2,4,6); "
            "rows[10:13]=permute(a, shift) for shift in (1,17,2047). "
            "All inputs stay at/above SPARSE_K — the _pad_to_k path is never taken."
        ),
        generated_by=GENERATED_BY,
    )
    write_golden(
        path_stem=OUT_DIR / "sparse_ops_sims",
        arr_bytes=_f64_bytes(sims),
        dtype="f64",
        shape=[len(sims)],
        byte_order="le",
        subsystem="hdc",
        source=(
            "src/iai_mcp/lilli/tiers/sparse_vsa.py jaccard similarity for pairs "
            "[(a,b),(a,c),(b,c),(a,a)]"
        ),
        generated_by=GENERATED_BY,
    )


def main() -> None:
    samples = _load_samples()
    dump_projection_pair()
    dump_bsc_ops(samples)
    dump_bsc_bundle_10000()
    dump_fhrr_ops(samples)
    dump_sparse_ops(samples)
    print(f"dumped proj pair + 3 op tables + bsc D=10000 bundle (+ f64 sims siblings) into {OUT_DIR}")


if __name__ == "__main__":
    main()
