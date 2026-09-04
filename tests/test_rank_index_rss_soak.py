"""Hermetic subprocess RSS soak over the Python-resident rank caches.

Builds the synthetic production-shaped corpus (``tests/_synthetic_cue_corpus``)
in a throwaway subprocess, warms the recall path, then samples macOS
``phys_footprint`` across repeated warm-recall cycles until the reading
plateaus (mirrors ``test_generational_cache_rss.py``'s plateau-detection
shape). Never touches the live daemon.

Three of the five original caches are retired (memo-drop):
``graph._collected_pool``, ``graph._normalized_pool``,
``graph._degree_map_cache`` never populate on the graph anymore. Two are
kept (documented live readers -- see the rank-cache-retirement guard test):
``graph._records_view_cache`` and ``LexicalIndex._postings``.

The worker is a plain function driven by a ``--worker`` CLI flag on this same
file, matching ``tests/rss_rca/phys_footprint_harness.py``'s parent/worker
split. Reused unmodified for the post-retirement measurement -- only the
``--label``/``--scale-records`` values passed at invocation change; the soak
logic stays the same regardless of which caches are actually resident on the
tree it runs against.

The committed pytest test runs the 64-record base corpus only -- enough to
prove the harness methodology and confirm cache residency, NOT enough to
resolve the two retired pool matrices' own delta (at 64 records that delta is
~200 KB, smaller than one cycle's own sampling noise here). A
production-representative measurement needs ``--scale-records N`` at real
scale, matching the accepted arithmetic-lower-bound RSS baseline this
retirement was scoped against; this replicates the base corpus's records
(fresh ids, reused embeddings -- no extra embedder cost) rather than
re-embedding N real sentences.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_RESULT_SENTINEL = "RANK_INDEX_RSS_SOAK_RESULT_JSON "
_PLATEAU_WINDOW = 5
_PLATEAU_TOLERANCE_BYTES = 3 * 1024 * 1024
_MAX_CYCLES = 30
_RUSAGE_INFO_V2 = 2


def _phys_footprint_bytes(pid: "int | None" = None) -> "int | None":
    """macOS charged-memory reader, vendored from
    ``tests/rss_rca/phys_footprint_harness.py`` so the worker subprocess
    stays self-contained (no pytest conftest active outside a pytest run)."""
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


def _scale_records(records: list, target_n: int, seed: int = 0) -> list:
    """Replicate the base corpus up to ``target_n`` records for a
    production-representative RSS measurement, reusing each source record's
    real embedding (no extra embedder cost) with a fresh id and a small
    created_at jitter."""
    import dataclasses
    import random
    from datetime import timedelta
    from uuid import uuid4

    if target_n <= len(records):
        return records
    rng = random.Random(seed)
    out = list(records)
    i = 0
    while len(out) < target_n:
        base = records[i % len(records)]
        jitter = timedelta(hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
        stamp = base.created_at + jitter
        out.append(dataclasses.replace(base, id=uuid4(), created_at=stamp, updated_at=stamp))
        i += 1
    return out


def _worker_main(label: str, scale_records: int = 0) -> dict:
    """In-process measurement, run inside the isolated worker subprocess."""
    tmp_root = Path(tempfile.mkdtemp(prefix="rank-index-rss-soak-"))
    fake_home = tmp_root / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(fake_home)
    os.environ["IAI_MCP_STORE"] = str(tmp_root / "store")
    os.environ["IAI_DAEMON_SOCKET_PATH"] = str(tmp_root / "daemon.sock")
    os.environ["IAI_MCP_RECALL_SAMPLE_RATE"] = "1.0"
    os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = "iai-mcp-rss-soak-passphrase"
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import keyring as _keyring

    fake_keyring: dict = {}
    _keyring.get_password = lambda s, u: fake_keyring.get((s, u))
    _keyring.set_password = lambda s, u, p: fake_keyring.__setitem__((s, u), p)
    _keyring.delete_password = lambda s, u: fake_keyring.pop((s, u), None)

    from tests._synthetic_cue_corpus import (
        build_corpus_records,
        build_cue_set,
        flatten_cues,
        insert_corpus,
    )

    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    embedder = Embedder()
    records = build_corpus_records(seed=0, embedder=embedder)
    if scale_records > 0:
        records = _scale_records(records, scale_records)
    store = MemoryStore(path=Path(os.environ["IAI_MCP_STORE"]))
    insert_corpus(store, records)
    graph, assignment, rich_club = build_runtime_graph(store)

    cues = flatten_cues(build_cue_set(seed=0))

    # Warm the resident lexical postings the way the real scoped-search /
    # nightly warm-up surface does -- the recall path itself only ever reads
    # a warm index (`lexical_query_warm`), never builds one.
    store.lexical_search(cues[0].text, k=10)

    def _run_cycle() -> None:
        for cue in cues:
            recall_for_response(
                store=store,
                graph=graph,
                assignment=assignment,
                rich_club=rich_club,
                embedder=embedder,
                cue=cue.text,
                session_id="rank-index-rss-soak",
                budget_tokens=1500,
                mode=cue.mode,
            )

    samples: "list[int]" = []
    for _ in range(_MAX_CYCLES):
        _run_cycle()
        gc.collect()
        v = _phys_footprint_bytes()
        if v is None:
            return {"ok": False, "error": "phys_footprint unavailable on this host"}
        samples.append(v)
        if len(samples) >= _PLATEAU_WINDOW:
            window = samples[-_PLATEAU_WINDOW:]
            if max(window) - min(window) <= _PLATEAU_TOLERANCE_BYTES:
                break

    from iai_mcp.store._lexical_index import LexicalIndex

    lex_idx = getattr(store, "_lexical_idx", None)
    caches_resident = {
        "collected_pool": getattr(graph, "_collected_pool", None) is not None,
        "normalized_pool": getattr(graph, "_normalized_pool", None) is not None,
        "records_view_cache": bool(getattr(graph, "_records_view_cache", None)),
        "degree_map_cache": getattr(graph, "_degree_map_cache", None) is not None,
        "lexical_postings": (
            isinstance(lex_idx, LexicalIndex)
            and lex_idx.generation is not None
            and bool(list(lex_idx.iter_token_postings()))
        ),
    }

    return {
        "ok": True,
        "label": label,
        "plateau_phys_footprint_bytes": samples[-1],
        "cycles_run": len(samples),
        "samples": samples,
        "record_count": len(records),
        "cue_count": len(cues),
        "caches_resident": caches_resident,
    }


def _run_soak_subprocess(label: str, scale_records: int = 0) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--label", label, "--scale-records", str(scale_records),
    ]
    proc = subprocess.run(
        cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=1200,
    )
    for line in (proc.stdout or "").splitlines():
        if line.startswith(_RESULT_SENTINEL):
            return json.loads(line[len(_RESULT_SENTINEL) :])
    raise RuntimeError(
        "rss soak worker produced no result line; "
        f"returncode={proc.returncode} stderr_tail={(proc.stderr or '')[-2000:]}"
    )


# Three of five caches are retired (memo-drop): collected_pool,
# normalized_pool, degree_map_cache. records_view_cache and lexical_postings
# are kept (documented live readers -- see the rank-cache-retirement guard
# test).
_RETIRED_CACHE_KEYS = ("collected_pool", "normalized_pool", "degree_map_cache")
_KEPT_CACHE_KEYS = ("records_view_cache", "lexical_postings")


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="phys_footprint is macOS-only")
@pytest.mark.timeout(1200)
def test_rank_cache_steady_state_rss_plateau() -> None:
    # Base corpus only (see module docstring): proves the harness and cache
    # residency, not the pool-matrix delta -- that needs --scale-records at
    # production scale, run manually against the RSS baseline this
    # retirement was scoped against.
    result = _run_soak_subprocess("after")
    assert result.get("ok"), f"rss soak worker failed: {result}"

    plateau = result["plateau_phys_footprint_bytes"]
    assert isinstance(plateau, int) and plateau > 0, (
        f"soak reported a non-positive plateau: {plateau!r}"
    )
    assert result["record_count"] > 0
    assert result["cue_count"] > 0

    caches_resident = result["caches_resident"]
    for name in _KEPT_CACHE_KEYS:
        assert caches_resident.get(name), (
            f"{name} is a KEPT cache but was not resident at soak plateau"
        )
    for name in _RETIRED_CACHE_KEYS:
        assert not caches_resident.get(name), (
            f"{name} is retired but was resident at soak plateau -- this "
            "run cannot stand as a post-retirement RSS measurement"
        )


def _cli_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--label", default="before")
    parser.add_argument(
        "--scale-records", type=int, default=0,
        help="replicate the base corpus up to N records before the soak "
             "(0 = base corpus as-is); use for a production-representative "
             "measurement the base 64-record corpus cannot resolve",
    )
    args = parser.parse_args()
    if not args.worker:
        return 0
    try:
        result = _worker_main(args.label, scale_records=args.scale_records)
    except Exception as exc:  # noqa: BLE001 -- report failure as a result line
        result = {"ok": False, "error": repr(exc)}
    sys.stdout.write(_RESULT_SENTINEL + json.dumps(result, default=str) + "\n")
    sys.stdout.flush()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(_cli_main())
