"""Python-vs-Rust byte-equality differential for the hypervector kernels.

Each op is run under the pure-Python reference and under the native backend on
the same inputs; the outputs must be byte-identical (similarity floats within a
tiny tolerance, since they are used for ranking, not persisted). The sparse
pad-path underflow case and the over-cap emit-then-raise contract are covered so
the native delegation cannot silently drop either property.

The whole module skips cleanly when the native ``hd`` module is not built, so a
Python-only checkout never hard-fails here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if the native hd sub-module is unavailable.
hd = pytest.importorskip("iai_mcp_native.hd", reason="native hd module not built")

from iai_mcp.lilli.crossmodal import embed_to_hv
from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "golden" / "hdc"

_BACKEND_ENV = "IAI_MCP_HD_BACKEND"


def _under(monkeypatch: pytest.MonkeyPatch, backend: str, fn, *args, **kwargs):
    monkeypatch.setenv(_BACKEND_ENV, backend)
    try:
        return fn(*args, **kwargs)
    finally:
        monkeypatch.delenv(_BACKEND_ENV, raising=False)


def _both(monkeypatch: pytest.MonkeyPatch, fn, *args, **kwargs):
    py = _under(monkeypatch, "python", fn, *args, **kwargs)
    ru = _under(monkeypatch, "rust", fn, *args, **kwargs)
    return py, ru


def _sample_strings() -> list[str]:
    return json.loads((_FIXTURES / "sample_strings.json").read_text())


# --- BSC -------------------------------------------------------------------


def test_bsc_bind_byte_equal(monkeypatch) -> None:
    for s in _sample_strings()[:20]:
        a = bsc.filler_hv(s)
        b = bsc.filler_hv(s + "-pair")
        py, ru = _both(monkeypatch, bsc.bind, a, b)
        assert py == ru, f"bsc.bind mismatch on {s!r}"


def test_bsc_bundle_byte_equal(monkeypatch) -> None:
    pairs = [("WHEN", bsc.filler_hv("x")), ("WHERE", bsc.filler_hv("y")),
             ("ROLE", bsc.filler_hv("z"))]
    py, ru = _both(monkeypatch, bsc.bundle, pairs)
    assert py == ru


def test_bsc_bundle_d10000_tail_byte_equal(monkeypatch) -> None:
    # 10000 bits = 1250 bytes = 156 u64 words + a 2-byte tail. Bundling at
    # D=10000 drives the scalar tail-byte vote of the SWAR kernel that the
    # word-aligned D=4096 default never reaches.
    pairs = [
        ("WHEN", bsc.filler_hv("x", D=10000)),
        ("WHERE", bsc.filler_hv("y", D=10000)),
        ("ROLE", bsc.filler_hv("z", D=10000)),
    ]
    py, ru = _both(monkeypatch, bsc.bundle, pairs, D=10000)
    assert py == ru


def test_bsc_permute_byte_equal(monkeypatch) -> None:
    hv = bsc.role_hv("WHEN")
    for shift in (0, 1, -1, 7, 4095, 4096):
        py, ru = _both(monkeypatch, bsc.permute, hv, shift)
        assert py == ru, f"bsc.permute mismatch at shift={shift}"


def test_bsc_similarity_close(monkeypatch) -> None:
    a = bsc.filler_hv("alpha")
    b = bsc.filler_hv("beta")
    py, ru = _both(monkeypatch, bsc.similarity, a, b)
    # BSC similarity is (D - 2*ham)/D, an integer-exact rational on both sides.
    assert py == ru
    py_same, ru_same = _both(monkeypatch, bsc.similarity, a, a)
    assert py_same == ru_same == 1.0


# --- FHRR ------------------------------------------------------------------


def test_fhrr_bind_unbind_byte_equal(monkeypatch) -> None:
    a = fhrr.role_hv("WHEN")
    b = fhrr.filler_hv("today")
    py_bind, ru_bind = _both(monkeypatch, fhrr.bind, a, b)
    assert py_bind == ru_bind
    py_unbind, ru_unbind = _both(monkeypatch, fhrr.unbind, py_bind, a)
    assert py_unbind == ru_unbind


def test_fhrr_bundle_byte_equal(monkeypatch) -> None:
    hvs = [fhrr.filler_hv(s) for s in _sample_strings()[:5]]
    py, ru = _both(monkeypatch, fhrr.bundle, hvs)
    assert py == ru


def test_fhrr_permute_byte_equal(monkeypatch) -> None:
    hv = fhrr.role_hv("WHERE")
    for shift in (0, 1, -3, 9999, 10000):
        py, ru = _both(monkeypatch, fhrr.permute, hv, shift)
        assert py == ru, f"fhrr.permute mismatch at shift={shift}"


def test_fhrr_similarity_close(monkeypatch) -> None:
    a = fhrr.role_hv("ACTOR")
    b = fhrr.filler_hv("alice")
    py, ru = _both(monkeypatch, fhrr.similarity, a, b)
    assert abs(py - ru) < 1e-9


# --- sparse VSA ------------------------------------------------------------


def test_sparse_bind_bundle_byte_equal(monkeypatch) -> None:
    a = sparse_vsa.role_hv("a")
    b = sparse_vsa.role_hv("b")
    py_bind, ru_bind = _both(monkeypatch, sparse_vsa.bind, a, b)
    assert py_bind == ru_bind
    py_bundle, ru_bundle = _both(monkeypatch, sparse_vsa.bundle, [a, b])
    assert py_bundle == ru_bundle


def test_sparse_permute_similarity_equal(monkeypatch) -> None:
    a = sparse_vsa.role_hv("WHEN")
    py_perm, ru_perm = _both(monkeypatch, sparse_vsa.permute, a, 5)
    assert py_perm == ru_perm
    py_sim, ru_sim = _both(monkeypatch, sparse_vsa.similarity, a, sparse_vsa.role_hv("b"))
    assert py_sim == ru_sim


def test_sparse_pad_path_underflow_byte_equal(monkeypatch) -> None:
    # Union of these two tiny lists has only 3 elements (< SPARSE_K=20), so the
    # deterministic backfill (the pad path) must run and agree byte-for-byte.
    a, b = [1, 2], [2, 3]
    py_bind, ru_bind = _both(monkeypatch, sparse_vsa.bind, a, b)
    assert len(py_bind) == sparse_vsa.SPARSE_K
    assert py_bind == ru_bind, "sparse pad path diverges python vs rust"

    # Unbind underflow: a single-element remainder also triggers the pad.
    py_unbind, ru_unbind = _both(monkeypatch, sparse_vsa.unbind, [5, 6, 7], [6, 7])
    assert len(py_unbind) == sparse_vsa.SPARSE_K
    assert py_unbind == ru_unbind, "sparse unbind pad path diverges python vs rust"


# --- projection ------------------------------------------------------------


def test_projection_production_path_backend_invariant(monkeypatch) -> None:
    # Production projection always runs on numpy (the hybrid exception), so the
    # output of from_embedding must be identical under both backend flags.
    raw = (_FIXTURES / "proj_100_in.bin").read_bytes()
    embs = np.frombuffer(raw, dtype="<f4").reshape(100, 384)
    for i in range(100):
        emb = embs[i].tolist()
        py, ru = _both(monkeypatch, embed_to_hv.from_embedding, emb)
        assert py == ru, f"projection production path diverged on embedding {i}"
        assert len(ru) == 1250


def test_native_projection_kernel_byte_equal_numpy() -> None:
    # The native projection kernel is parity-only (not on the production path),
    # but it must still reproduce the numpy reference byte-for-byte across all
    # 100 frozen embeddings — that byte-identity is what makes it a valid frozen
    # reference even though production never routes through it.
    raw = (_FIXTURES / "proj_100_in.bin").read_bytes()
    embs = np.frombuffer(raw, dtype="<f4").reshape(100, 384)
    for i in range(100):
        emb = embs[i]
        reference = embed_to_hv.from_embedding(emb.tolist())
        native = bytes(hd.project(np.ascontiguousarray(emb, dtype=np.float32)))
        assert native == reference, f"native projection kernel diverged on {i}"
        assert len(native) == 1250


# --- emit-then-raise under the rust backend --------------------------------


def test_over_cap_emit_then_raise_under_rust(tmp_path, monkeypatch) -> None:
    from iai_mcp.events import query_events
    from iai_mcp.lilli.errors import BundleCapacityError
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_KEYRING_BYPASS", "true")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-hd-differential-pp")
    monkeypatch.setenv(_BACKEND_ENV, "rust")
    store = MemoryStore(path=tmp_path / "store")
    try:
        roles = [
            "WHEN", "WHERE", "ROLE", "PROJECT", "COMMUNITY_ID",
            "TEMPORAL_POSITION", "ACTOR", "OBJECT", "INTENT", "MODALITY", "LANG",
        ]
        pairs = [(r, bsc.filler_hv(f"v{i}")) for i, r in enumerate(roles)]
        with pytest.raises(BundleCapacityError):
            bsc.bundle(pairs, store=store)
        events = query_events(store, kind="role_saturation_warning")
        assert len(events) >= 1, (
            "emit-then-raise contract violated under the rust backend: the "
            "saturation event must land before the raise"
        )
    finally:
        try:
            store.close()
        except Exception:
            pass


def test_pad_deterministic_across_pythonhashseed() -> None:
    # The pad path must not depend on per-process hash randomisation: the same
    # underflow bind must produce identical output across hash seeds.
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from iai_mcp.lilli.tiers import sparse_vsa as s; "
        "print(s.bind([1, 2], [2, 3]))"
    )
    outs = []
    for seed in ("0", "1", "2"):
        env = {
            "PYTHONHASHSEED": seed,
            _BACKEND_ENV: "python",
            "PATH": __import__("os").environ.get("PATH", ""),
        }
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, check=True,
        )
        outs.append(res.stdout.strip())
    assert len(set(outs)) == 1, f"pad still hash-seed-salted: {outs}"
