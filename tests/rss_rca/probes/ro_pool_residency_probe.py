"""Two-process probe: aggregate RSS of a pooled-RO-borrow workload across the pager-bound commit boundary.

Runs a capped ``--worker`` subprocess against a synthetic corpus, pins the
RO pool to one slot, borrows it repeatedly, and samples (parent + children)
RSS in a hermetic HOME/store sandbox, following the same pattern as
``tests/rss_rca/phys_footprint_harness.py``. No env value expresses
"unbounded", so reading the effect of the bound (``_OwnedEngineRawConn.__init__``)
requires running this same script twice across the commit boundary and
diffing ``residency_growth_bytes`` -- no single run yields a verdict on its
own.

Self-check (arm-independent workload validity, not a before/after verdict)
----------------------------------------------------------------------------
``--self-check`` asserts only what is true on EITHER arm: the borrow loop's
query-key coverage of the id space clears a floor (``key_coverage_fraction``),
and the mid-loop refresh engaged the engine's real re-arm path
(``slot_refresh_count >= 1``, ``slot_reopen_count`` stays at 0). A
``--self-check`` failure means the FIXTURE is broken (too few distinct keys,
or a fresh-slot-per-borrow bug) -- not that the connection is unbounded.

Invocation::

    python tests/rss_rca/probes/ro_pool_residency_probe.py --n 3000 --self-check
    python tests/rss_rca/probes/ro_pool_residency_probe.py --n 3000 8000 15000

This is a measurement tool. It changes no production source and never starts
the live daemon.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.rss_rca.phys_footprint_harness import (  # noqa: E402
    _VOLATILE_PREFIXES,
    _build_synthetic_edges,
    _kill_tree,
    _read_record_ids,
    sample_aggregate,
)

_RESULT_SENTINEL = "RO_POOL_RESIDENCY_RESULT_JSON "

# Additional volatile prefixes beyond phys_footprint_harness's own list: any
# concurrently running iai-mcp session churns directive/session-continuity/
# working-tier cache files that the harness's list predates. Defined locally
# rather than editing the shared harness module (out of this lever's scope).
_EXTRA_VOLATILE_MARKERS: tuple[str, ...] = (
    ".working-tier.",
    ".directives.cached.md",
    ".deferred-provenance.jsonl",
    ".memory-bank/",
    ".next-turn-pack.",
    ".session-continuity.cached.md",
    "hippo/.max-created-at",
    # iai-tools' own code-search index (segments/manifests under tools/) is a
    # background compaction/merge process unrelated to the Hippo memory store
    # this probe measures; it churns independently of anything this probe does.
    "tools/",
    ".daemon-watchdog.log",
)


def _real_store_signature() -> dict[str, tuple[int, int]] | None:
    """Fingerprint the real ``~/.iai-mcp/`` for the hermeticity check.

    Wraps the harness's own signature scan but additionally excludes markers
    a concurrently running iai-mcp session churns (see module docstring
    above) so this probe's hermeticity check does not false-positive on that
    background activity.
    """
    real = Path.home() / ".iai-mcp"
    if not real.exists():
        return None
    sig: dict[str, tuple[int, int]] = {}
    for p in real.rglob("*"):
        try:
            rel = str(p.relative_to(real))
        except ValueError:
            continue
        if any(
            rel.startswith(v) or rel == v.rstrip("/") for v in _VOLATILE_PREFIXES
        ):
            continue
        if any(marker in rel for marker in _EXTRA_VOLATILE_MARKERS):
            continue
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        sig[rel] = (st.st_size, st.st_mtime_ns)
    return sig


_BYTES_PER_GIB = 1024.0 ** 3
_BYTES_PER_MIB = 1024.0 ** 2

# Minimum fraction of the corpus's distinct ids the borrow loop must touch as
# query keys to plausibly clear the bound's ~1024-page budget (8 MiB / the
# engine's real 8 KiB PAGE_SIZE). Derived from the loop's own WHERE...IN(...)
# id set rather than cells_visited_count(), which is a result-size signal on
# this probe's sparse edge topology, not a page-touch signal.
_MIN_KEY_COVERAGE_FRACTION = 0.95


# ---------------------------------------------------------------------------
# Worker -- builds the store, then runs the pinned-slot borrow loop.
# ---------------------------------------------------------------------------


def _borrow_loop(
    pool: Any,
    store: Any,
    ids: list[Any],
    *,
    m_borrows: int,
    sample_every: int,
    record: Any,
) -> dict[str, Any]:
    """Run ``m_borrows`` borrows of the single pinned slot, sampling RSS.

    Triggers exactly one real mid-loop refresh: a second connection (the
    still-open writer) commits a write, then the pool's generation is bumped
    so the slot's NEXT borrow re-arms via the engine's ``refresh()`` in place.
    """
    from iai_mcp.store._buffers import flush_edge_buffer

    n_ids = len(ids)
    # window capped at 1000: engine SELECT ... IN (...) rejects a longer list.
    window = min(1000, max(16, n_ids // 9)) if n_ids else 16
    # A lap of lap_len borrows tiles [0, n_ids) contiguously once, guaranteeing
    # key_coverage_fraction == 1.0 after the first lap -- required by the
    # self-check floor below.
    lap_len = max(1, -(-n_ids // window)) if n_ids else 1
    lap_offset_step = max(1, (n_ids // 7) | 1)

    rss_curve: list[dict[str, Any]] = []
    touched_ids: set[str] = set()
    refreshed_mid = False
    mid = m_borrows // 2

    for i in range(m_borrows):
        if n_ids:
            lap, pos = divmod(i, lap_len)
            offset = (lap * lap_offset_step) % n_ids
            start = (pos * window + offset) % n_ids
            end = start + window
            batch_ids = ids[start:end]
            if end > n_ids:
                batch_ids = batch_ids + ids[: end - n_ids]
        else:
            batch_ids = []
        params = [str(x) for x in batch_ids] or [
            "00000000-0000-0000-0000-000000000000"
        ]
        touched_ids.update(params)
        placeholders = ",".join("?" * len(params))
        sql = (
            f"SELECT src, dst FROM edges WHERE src IN ({placeholders}) "
            f"OR dst IN ({placeholders})"
        )
        with pool.borrow() as conn:
            conn.execute(sql, (*params, *params)).fetchall()

        if i == mid and n_ids >= 2:
            store.boost_edges(
                [(ids[0], ids[1])], delta=0.05, edge_type="hebbian"
            )
            flush_edge_buffer(store)
            pool.mark_stale()
            refreshed_mid = True

        if sample_every and i % sample_every == 0:
            gc.collect()
            rss_curve.append(record(f"borrow-{i}"))

    gc.collect()
    settled = record("after-borrow-loop-settled")

    key_coverage_fraction = (len(touched_ids) / n_ids) if n_ids else 0.0

    return {
        "window": window,
        "key_coverage_fraction": key_coverage_fraction,
        "distinct_ids_touched": len(touched_ids),
        "refreshed_mid_loop": refreshed_mid,
        "slot_refresh_count": pool.slot_refresh_count,
        "slot_reopen_count": pool.slot_reopen_count,
        "settled_sample": settled,
        "rss_curve": rss_curve,
    }


def _worker_run(
    *,
    n: int,
    seed: int,
    edge_factor: float,
    m_borrows: int,
    sample_every: int,
    out_jsonl: Path | None,
) -> dict[str, Any]:
    """In-process measurement. Returns the result dict (also written to jsonl)."""
    real_sig_before = _real_store_signature()

    tmp_root = Path(tempfile.mkdtemp(prefix="iai-ro-residency-"))
    prev_env = {
        k: os.environ.get(k)
        for k in (
            "HOME", "IAI_MCP_STORE", "IAI_MCP_USER_MODEL_PATH",
            "IAI_MCP_CRYPTO_PASSPHRASE", "PYTHON_KEYRING_BACKEND",
            "IAI_MCP_RO_POOL_SIZE",
        )
    }
    os.environ["HOME"] = str(tmp_root)
    os.environ["IAI_MCP_STORE"] = str(tmp_root / ".iai-mcp" / "hippo")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(
        tmp_root / ".iai-mcp" / "user_model.json"
    )
    if not prev_env["IAI_MCP_CRYPTO_PASSPHRASE"]:
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
            "iai-mcp-ro-residency-probe-deterministic-2026"
        )
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    # Pin the RO pool to exactly one slot for the whole worker lifetime, so
    # residency accumulates on a single slot rather than spreading across
    # several -- a fresh slot per borrow would reset it and hide the effect.
    os.environ["IAI_MCP_RO_POOL_SIZE"] = "1"
    (tmp_root / ".iai-mcp").mkdir(parents=True, exist_ok=True)

    out_fh = out_jsonl.open("w", encoding="utf-8") if out_jsonl is not None else None

    def record(label: str) -> dict[str, Any]:
        s = sample_aggregate(label)
        if out_fh is not None:
            out_fh.write(json.dumps(s) + "\n")
            out_fh.flush()
        return s

    store = None
    pool = None
    try:
        from iai_mcp.hippo._ro_pool import RoConnPool
        from iai_mcp.store import MemoryStore
        from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

        record("boot")

        store_path = Path(os.environ["IAI_MCP_STORE"])
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(path=store_path)
        eff_dim = int(store.embed_dim)

        from tests.rss_rca.corpus import build_corpus

        inserted = build_corpus(
            store, n=n, seed=seed, dim=eff_dim,
            varied_tags=True, with_hv_payload=False,
        )
        flush_record_buffer(store)
        record("after-corpus")

        edges_inserted = _build_synthetic_edges(store, n, edge_factor, seed)
        flush_edge_buffer(store)
        record("after-edges")

        ids = _read_record_ids(store)
        gc.collect()
        baseline = record("before-borrow-loop")

        pool = RoConnPool(store.db._db_path, size=1)
        loop_result = _borrow_loop(
            pool, store, ids,
            m_borrows=m_borrows, sample_every=sample_every, record=record,
        )

        result: dict[str, Any] = {
            "ok": True,
            "n": n,
            "inserted_records": inserted,
            "inserted_edges": edges_inserted,
            "m_borrows": m_borrows,
            "distinct_key_window": loop_result["window"],
            "key_coverage_fraction": loop_result["key_coverage_fraction"],
            "distinct_ids_touched": loop_result["distinct_ids_touched"],
            "refreshed_mid_loop": loop_result["refreshed_mid_loop"],
            "slot_refresh_count": loop_result["slot_refresh_count"],
            "slot_reopen_count": loop_result["slot_reopen_count"],
            "baseline_total_rss": baseline["total_rss"],
            "settled_total_rss": loop_result["settled_sample"]["total_rss"],
            "residency_growth_bytes": (
                loop_result["settled_sample"]["total_rss"] - baseline["total_rss"]
            ),
            "rss_curve": loop_result["rss_curve"],
        }
        return result
    finally:
        try:
            if pool is not None:
                pool.close()
        except Exception:  # noqa: BLE001
            pass
        if out_fh is not None:
            try:
                out_fh.close()
            except Exception:  # noqa: BLE001
                pass
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp_root, ignore_errors=True)
        real_sig_after = _real_store_signature()
        if real_sig_before != real_sig_after:
            before = real_sig_before if isinstance(real_sig_before, dict) else {}
            after = real_sig_after if isinstance(real_sig_after, dict) else {}
            added = sorted(set(after) - set(before))
            removed = sorted(set(before) - set(after))
            modified = sorted(
                k for k in set(before) & set(after) if before[k] != after[k]
            )
            raise RuntimeError(
                "HERMETICITY VIOLATION: ~/.iai-mcp changed during the probe. "
                f"added={added!r} removed={removed!r} modified={modified!r}."
            )


# ---------------------------------------------------------------------------
# Parent -- spawn the worker, cap it, parse its result.
# ---------------------------------------------------------------------------


def _fmt_mib(b: int | None) -> str:
    if b is None:
        return "   n/a"
    return f"{b / _BYTES_PER_MIB:7.1f}"


def _run_capped_worker(
    *,
    n: int,
    seed: int,
    edge_factor: float,
    m_borrows: int,
    sample_every: int,
    cap_gib: float,
    sample_interval: float,
    out_dir: Path,
) -> dict[str, Any]:
    out_jsonl = out_dir / f"phases-n{n}.jsonl"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--n", str(n),
        "--seed", str(seed),
        "--edge-factor", str(edge_factor),
        "--m-borrows", str(m_borrows),
        "--sample-every", str(sample_every),
        "--worker-out", str(out_jsonl),
    ]
    cap_bytes = cap_gib * _BYTES_PER_GIB
    env = dict(os.environ)
    proc = subprocess.Popen(
        cmd,
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    peak_worker_rss = 0
    aborted = False
    abort_reason = ""
    try:
        ps_proc = psutil.Process(proc.pid)
    except Exception:  # noqa: BLE001
        ps_proc = None

    while True:
        ret = proc.poll()
        if ret is not None:
            break
        subtree_rss = 0
        if ps_proc is not None:
            try:
                subtree_rss = int(ps_proc.memory_info().rss)
                for ch in ps_proc.children(recursive=True):
                    try:
                        subtree_rss += int(ch.memory_info().rss)
                    except Exception:  # noqa: BLE001
                        continue
            except Exception:  # noqa: BLE001
                subtree_rss = 0
        peak_worker_rss = max(peak_worker_rss, subtree_rss)
        if subtree_rss > cap_bytes:
            aborted = True
            abort_reason = (
                f"worker subtree RSS {subtree_rss / _BYTES_PER_GIB:.3f} GiB "
                f"exceeded cap {cap_gib:.3f} GiB at N={n}"
            )
            _kill_tree(proc, ps_proc)
            break
        time.sleep(sample_interval)

    stdout, stderr = proc.communicate(timeout=60)

    if aborted:
        return {
            "ok": False, "aborted": True, "n": n, "reason": abort_reason,
            "peak_worker_rss": peak_worker_rss,
            "stderr_tail": (stderr or "")[-2000:],
        }

    # Parse the sentinel line BEFORE treating a non-zero exit as a hard error:
    # the worker exits 1 whenever its own result dict has ok=False (including
    # a clean, JSON-reported failure), so a real error message is only ever
    # visible by parsing the line first.
    result: dict[str, Any] | None = None
    for line in (stdout or "").splitlines():
        if line.startswith(_RESULT_SENTINEL):
            try:
                result = json.loads(line[len(_RESULT_SENTINEL):])
            except json.JSONDecodeError:
                result = None
            break

    if result is not None:
        result["peak_worker_rss_observed_by_parent"] = peak_worker_rss
        if not result.get("ok", False):
            result.setdefault("stderr_tail", (stderr or "")[-2000:])
        return result

    if proc.returncode != 0:
        return {
            "ok": False, "aborted": False, "n": n,
            "error": f"worker exited with code {proc.returncode}",
            "peak_worker_rss": peak_worker_rss,
            "stdout_tail": (stdout or "")[-2000:],
            "stderr_tail": (stderr or "")[-2000:],
        }
    return {
        "ok": False, "aborted": False, "n": n,
        "error": "worker produced no parseable result line",
        "peak_worker_rss": peak_worker_rss,
        "stdout_tail": (stdout or "")[-2000:],
        "stderr_tail": (stderr or "")[-2000:],
    }


def _self_check(result: dict[str, Any]) -> list[str]:
    """Arm-independent workload-validity checks. Returns a list of failures."""
    failures: list[str] = []
    coverage = result.get("key_coverage_fraction")
    if coverage is None:
        failures.append("key_coverage_fraction unavailable")
    elif coverage < _MIN_KEY_COVERAGE_FRACTION:
        failures.append(
            f"key_coverage_fraction={coverage:.3f} is below the "
            f"{_MIN_KEY_COVERAGE_FRACTION} floor -- the borrow loop's query "
            "keys covered too little of the id space; increase --m-borrows "
            "or the id window (fixture is not exercising the bound's page "
            "budget)"
        )
    if not result.get("refreshed_mid_loop"):
        failures.append("mid-loop refresh was never triggered")
    if result.get("slot_refresh_count", 0) < 1:
        failures.append(
            "slot_refresh_count == 0 -- the mid-loop write+mark_stale did not "
            "cause a real engine refresh"
        )
    if result.get("slot_reopen_count", 0) != 0:
        failures.append(
            f"slot_reopen_count={result.get('slot_reopen_count')} -- the pool "
            "fell back to reopen instead of the in-place refresh path"
        )
    return failures


def _print_result(result: dict[str, Any]) -> None:
    n = result.get("n")
    print(f"\n=== N={n} RO-pool-borrow residency ===")
    print(f"  m_borrows              {result.get('m_borrows')}")
    print(f"  distinct_key_window    {result.get('distinct_key_window')}")
    print(f"  key_coverage_fraction  {result.get('key_coverage_fraction')}")
    print(f"  distinct_ids_touched   {result.get('distinct_ids_touched')}")
    print(f"  refreshed_mid_loop     {result.get('refreshed_mid_loop')}")
    print(f"  slot_refresh_count     {result.get('slot_refresh_count')}")
    print(f"  slot_reopen_count      {result.get('slot_reopen_count')}")
    print(f"  baseline_total_rss     {_fmt_mib(result.get('baseline_total_rss'))} MiB")
    print(f"  settled_total_rss      {_fmt_mib(result.get('settled_total_rss'))} MiB")
    print(
        f"  residency_growth       "
        f"{_fmt_mib(result.get('residency_growth_bytes'))} MiB"
    )
    curve = result.get("rss_curve") or []
    if curve:
        print("  rss_curve (borrow -> total_rss MiB):")
        for s in curve:
            print(f"    {s['phase']:<14} {_fmt_mib(s['total_rss'])}")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ro_pool_residency_probe",
        description=(
            "Two-process hermetic probe measuring RO-pool-borrow page-cache "
            "residency: pins the RO pool to one slot, borrows it many times "
            "over a synthetic corpus's edges, and reports settled residency "
            "growth plus a mid-loop refresh proof."
        ),
    )
    parser.add_argument("--n", type=int, nargs="+", default=[3000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-factor", type=float, default=1.3)
    parser.add_argument("--m-borrows", type=int, default=4000)
    parser.add_argument(
        "--sample-every", type=int, default=500,
        help="sample RSS every this many borrows (0 disables the curve)",
    )
    parser.add_argument("--cap-gib", type=float, default=4.5)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--self-check", action="store_true",
        help=(
            "assert the workload's query keys covered enough of the id space "
            "and the mid-loop refresh engaged the real in-place re-arm path "
            "(arm-independent fixture validity, not a before/after verdict)"
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        n = args.n[0]
        try:
            result = _worker_run(
                n=n, seed=args.seed, edge_factor=args.edge_factor,
                m_borrows=args.m_borrows, sample_every=args.sample_every,
                out_jsonl=args.worker_out,
            )
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "n": n, "error": repr(exc)}
        sys.stdout.write(_RESULT_SENTINEL + json.dumps(result, default=str) + "\n")
        sys.stdout.flush()
        return 0 if result.get("ok") else 1

    out_dir = (args.out or Path(tempfile.mkdtemp(prefix="iai-ro-residency-out-"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ro_pool_residency_probe -- scales={args.n} m_borrows={args.m_borrows} out={out_dir}")

    overall_ok = True
    results: list[dict[str, Any]] = []
    for n in args.n:
        print(f"\n[scale N={n}] spawning capped worker ...")
        result = _run_capped_worker(
            n=n, seed=args.seed, edge_factor=args.edge_factor,
            m_borrows=args.m_borrows, sample_every=args.sample_every,
            cap_gib=args.cap_gib, sample_interval=args.sample_interval,
            out_dir=out_dir,
        )
        results.append(result)
        if not result.get("ok"):
            overall_ok = False
            if result.get("aborted"):
                print(f"[scale N={n}] ABORTED: {result.get('reason')}")
            else:
                print(f"[scale N={n}] ERROR: {result.get('error')}")
            if result.get("stderr_tail"):
                print("---- worker stderr tail ----")
                print(result["stderr_tail"])
            continue
        _print_result(result)
        if args.self_check:
            failures = _self_check(result)
            if failures:
                overall_ok = False
                print(f"[scale N={n}] SELF-CHECK FAILED:")
                for f in failures:
                    print(f"  - {f}")
            else:
                print(f"[scale N={n}] self-check PASSED")

    try:
        (out_dir / "summary.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nresults written to {out_dir / 'summary.json'}")
    except Exception as exc:  # noqa: BLE001
        print(f"failed to write summary: {exc!r}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(_main())
