"""Hermetic marginal phys_footprint harness for the columnar rank-feature
index.

Builds a wholly synthetic, production-shaped corpus (never real store
content) as flat Python columns, samples macOS ``phys_footprint``
immediately before and after constructing the columnar ``RankIndex`` once,
and reports the marginal delta -- the same before/after-one-build
differential the naive layout was measured with. The corpus is shaped on
four cost-driving axes (record count, average surface length, token
vocabulary/posting count, symmetrized adjacency count) because the index's
arena, postings CSR and adjacency CSR are driven by text and graph shape,
not record count alone -- a record-count-only corpus would make the
assertion below vacuous.

The measurement never touches a live store, graph, or daemon: the columns
are handed straight to the crate's constructor, isolating the struct's own
allocation cost from every Python-side pipeline cost around it. Each
measurement runs in a fresh subprocess (no accumulated allocator state
across repeats) and the run is repeated so the reported number is a
stable median, not a single noisy sample.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

_RESULT_SENTINEL = "RANK_INDEX_MARGINAL_FOOTPRINT_RESULT_JSON "
_RUSAGE_INFO_V2 = 2
_BYTES_PER_MIB = 1024.0 ** 2

_TARGET_RECORDS = 14_489
_TARGET_CHARS_PER_SURFACE = 559
_TARGET_VOCAB = 21_192
_TARGET_POSTINGS = 387_583
_TARGET_UNDIRECTED_EDGES = 30_031
_TARGET_ADJACENCY_ENTRIES = 60_062
_DIM = 384
_GATE_MIB = 45.0
_GATE_BYTES = int(_GATE_MIB * _BYTES_PER_MIB)
_NUM_RUNS = 3

_TOKEN_WIDTH = 8  # "tkn" + 5 digits -- fixed width, exact char-length math
_OCC_PER_DOC = max(1, round((_TARGET_CHARS_PER_SURFACE + 1) / (_TOKEN_WIDTH + 1)))


def _phys_footprint_bytes(pid: "int | None" = None) -> "int | None":
    """macOS charged-memory reader, vendored from
    ``tests/rss_rca/phys_footprint_harness.py`` so the worker subprocess
    stays self-contained."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        class _RUsageInfoV2(ctypes.Structure):
            _fields_ = [
                ("ri_uuid", ctypes.c_uint8 * 16),
                ("ri_user_time", ctypes.c_uint64),
                ("ri_system_time", ctypes.c_uint64),
                ("ri_pkg_idle_wkups", ctypes.c_uint64),
                ("ri_interrupt_wkups", ctypes.c_uint64),
                ("ri_pageins", ctypes.c_uint64),
                ("ri_wired_size", ctypes.c_uint64),
                ("ri_resident_size", ctypes.c_uint64),
                ("ri_phys_footprint", ctypes.c_uint64),
                ("ri_proc_start_abstime", ctypes.c_uint64),
                ("ri_proc_exit_abstime", ctypes.c_uint64),
                ("ri_child_user_time", ctypes.c_uint64),
                ("ri_child_system_time", ctypes.c_uint64),
                ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
                ("ri_child_interrupt_wkups", ctypes.c_uint64),
                ("ri_child_pageins", ctypes.c_uint64),
                ("ri_child_elapsed_abstime", ctypes.c_uint64),
                ("ri_diskio_bytesread", ctypes.c_uint64),
                ("ri_diskio_byteswritten", ctypes.c_uint64),
            ]

        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        info = _RUsageInfoV2()
        target = os.getpid() if pid is None else int(pid)
        rc = libsystem.proc_pid_rusage(target, _RUSAGE_INFO_V2, ctypes.byref(info))
        if rc != 0:
            return None
        footprint = int(info.ri_phys_footprint)
        return footprint if footprint > 0 else None
    except Exception:  # noqa: BLE001 -- unreadable footprint must never crash
        return None


def _sample_footprint_stable(n_samples: int = 3, pause: float = 0.05) -> int:
    """Median of a few back-to-back readings, to damp single-sample noise
    without repeating the build itself (a repeated build would understate
    the true marginal cost once the allocator starts reusing freed pages)."""
    vals: list[int] = []
    for _ in range(n_samples):
        gc.collect()
        v = _phys_footprint_bytes()
        if v is None:
            raise RuntimeError("phys_footprint unavailable on this host")
        vals.append(v)
        time.sleep(pause)
    return int(statistics.median(vals))


def _token(i: int) -> str:
    return f"tkn{i:05d}"


def _build_pairs(n: int, target_undirected: int) -> "list[tuple[int, int]]":
    """Ring plus chord layers over ``n`` ids -- a cheap, deterministic way
    to hit an exact undirected pair count without a real graph structure
    (the footprint cost here is a function of adjacency ENTRY count, not
    topology)."""
    pairs: list[tuple[int, int]] = []
    for off in (1, 2):
        for i in range(n):
            pairs.append((i + 1, ((i + off) % n) + 1))
    remaining = target_undirected - len(pairs)
    for i in range(max(0, remaining)):
        pairs.append((i + 1, ((i + 3) % n) + 1))
    return pairs[:target_undirected]


def _build_columns(n: int, seed: int) -> "tuple[dict, dict]":
    """Builds the exact positional columns ``RankIndex.new()`` takes, plus
    the achieved corpus-shape stats for the evidence record."""
    rnd = random.Random(seed)

    # Unique-tokens-per-doc split: distributes the total posting-pair target
    # evenly over the record count (remainder to the first N docs), then
    # shuffled so no doc position is systematically larger.
    base, extra = divmod(_TARGET_POSTINGS, n)
    unique_counts = [base + 1] * extra + [base] * (n - extra)
    rnd.shuffle(unique_counts)

    surfaces: list[str] = []
    used_vocab_ids: set[int] = set()
    total_chars = 0
    for u in unique_counts:
        u = min(u, _TARGET_VOCAB)
        picks = rnd.sample(range(_TARGET_VOCAB), u)
        used_vocab_ids.update(picks)
        occ = max(u, _OCC_PER_DOC)
        seq = [picks[i % u] for i in range(occ)]
        rnd.shuffle(seq)
        text = " ".join(_token(i) for i in seq)
        surfaces.append(text)
        total_chars += len(text)

    ids = list(range(1, n + 1))

    import numpy as np

    vectors = np.random.default_rng(seed).standard_normal((n, _DIM)).astype("float32")

    pairs = _build_pairs(n, _TARGET_UNDIRECTED_EDGES)
    edge_map: "dict[int, list[tuple[int, float, str]]]" = {}
    for a, b in pairs:
        edge_map.setdefault(a, []).append((b, 0.5, "hebbian"))
        edge_map.setdefault(b, []).append((a, 0.5, "hebbian"))
    edges = list(edge_map.items())
    adjacency_entries = sum(len(v) for v in edge_map.values())

    columns = dict(
        dim=_DIM,
        generation=1,
        ids=ids,
        vectors=vectors,
        edges=edges,
        surfaces=surfaces,
        aaak_index=[""] * n,
        created_at=[""] * n,
        stability=[0.5] * n,
        tier=["episodic"] * n,
        tags=[[] for _ in range(n)],
        salience_level=[0] * n,
        centrality=[0.0] * n,
    )
    achieved = dict(
        records=n,
        avg_surface_chars=total_chars / n,
        vocab_target=_TARGET_VOCAB,
        vocab_realized=len(used_vocab_ids),
        postings_target=_TARGET_POSTINGS,
        postings_achieved=sum(unique_counts),
        undirected_edge_pairs_target=_TARGET_UNDIRECTED_EDGES,
        undirected_edge_pairs_achieved=len(pairs),
        adjacency_entries_target=_TARGET_ADJACENCY_ENTRIES,
        adjacency_entries_achieved=adjacency_entries,
    )
    return columns, achieved


def _worker_main(seed: int) -> dict:
    columns, achieved = _build_columns(_TARGET_RECORDS, seed)

    before = _sample_footprint_stable()

    from iai_mcp_native import rank as _rank_native

    index = _rank_native.RankIndex(
        columns["dim"],
        columns["generation"],
        columns["ids"],
        columns["vectors"],
        columns["edges"],
        columns["surfaces"],
        columns["aaak_index"],
        columns["created_at"],
        columns["stability"],
        columns["tier"],
        columns["tags"],
        columns["salience_level"],
        columns["centrality"],
    )
    after = _sample_footprint_stable()
    built_len = len(index)  # keeps the struct referenced through the sample

    marginal = after - before
    return {
        "ok": True,
        "seed": seed,
        "before_bytes": before,
        "after_bytes": after,
        "marginal_bytes": marginal,
        "built_len": built_len,
        "achieved": achieved,
    }


def _run_worker_subprocess(seed: int) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--seed", str(seed)]
    proc = subprocess.run(
        cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=600,
    )
    for line in (proc.stdout or "").splitlines():
        if line.startswith(_RESULT_SENTINEL):
            return json.loads(line[len(_RESULT_SENTINEL):])
    raise RuntimeError(
        "marginal footprint worker produced no result line; "
        f"returncode={proc.returncode} stderr_tail={(proc.stderr or '')[-4000:]}"
    )


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="phys_footprint is macOS-only")
@pytest.mark.timeout(900)
def test_columnar_rank_index_marginal_phys_footprint() -> None:
    results = []
    for i in range(_NUM_RUNS):
        r = _run_worker_subprocess(seed=1000 + i)
        assert r.get("ok"), f"worker run {i} failed: {r}"
        results.append(r)

    marginals = [r["marginal_bytes"] for r in results]
    marginal_median = int(statistics.median(marginals))

    print(
        "\ncolumnar RankIndex marginal phys_footprint over "
        f"{_NUM_RUNS} independent hermetic builds:"
    )
    for r in results:
        a = r["achieved"]
        print(
            f"  seed={r['seed']} marginal={r['marginal_bytes']} bytes "
            f"({r['marginal_bytes'] / _BYTES_PER_MIB:.2f} MiB) "
            f"records={a['records']} avg_surface_chars={a['avg_surface_chars']:.1f} "
            f"vocab_realized={a['vocab_realized']}/{a['vocab_target']} "
            f"postings={a['postings_achieved']}/{a['postings_target']} "
            f"adjacency={a['adjacency_entries_achieved']}/{a['adjacency_entries_target']}"
        )
    print(
        f"  median={marginal_median} bytes ({marginal_median / _BYTES_PER_MIB:.2f} MiB) "
        f"min={min(marginals)} max={max(marginals)} gate={_GATE_BYTES} bytes ({_GATE_MIB} MiB)"
    )

    for r in results:
        a = r["achieved"]
        assert a["records"] == _TARGET_RECORDS
        assert a["vocab_realized"] >= 0.9 * _TARGET_VOCAB, (
            f"realized vocabulary {a['vocab_realized']} undershoots the "
            f"{_TARGET_VOCAB} target -- corpus is not production-shaped"
        )
        assert a["adjacency_entries_achieved"] == _TARGET_ADJACENCY_ENTRIES

    assert marginal_median <= _GATE_BYTES, (
        f"marginal phys_footprint gate MISS: median {marginal_median} bytes "
        f"({marginal_median / _BYTES_PER_MIB:.2f} MiB) exceeds the "
        f"{_GATE_MIB} MiB gate over a production-shaped corpus "
        f"(runs={marginals})"
    )


def _cli_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if not args.worker:
        return 0
    try:
        result = _worker_main(args.seed)
    except Exception as exc:  # noqa: BLE001 -- report failure as a result line
        result = {"ok": False, "error": repr(exc)}
    sys.stdout.write(_RESULT_SENTINEL + json.dumps(result, default=str) + "\n")
    sys.stdout.flush()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(_cli_main())
