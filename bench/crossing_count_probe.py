"""Honest per-recall synchronous engine execute() count against a real store.

Usage (against an isolated store copy — NEVER live prod):

    source .venv/bin/activate
    LILLI_STORAGE_DRIVER=lilli python bench/crossing_count_probe.py --arm both
    LILLI_STORAGE_DRIVER=lilli python bench/crossing_count_probe.py --warm-bundle --arm new

The store path defaults to $IAI_ISO; pass --store-path to override. Reads
the isolated copy only — never opens ~/.iai-mcp.

Four monkeypatched seams give the honest total (naively wrapping only the
caller-facing execute() undercounts by exactly the acquisition count):
  - HippoDB.ro_conn                        acquisition count
  - RoConnPool._FenceRetryingConn.execute  caller-issued pooled statements
  - RoConnPool._borrow_prepared            the pool's own borrow-time
                                            "SELECT 1 FROM records LIMIT 1"
                                            fence probe, which runs on
                                            slot.conn directly and bypasses
                                            _FenceRetryingConn entirely
  - _MutationSignallingConn.execute        still-synchronous writer executes

Every counter is keyed by thread identity and gated on being inside
dispatch() on the main thread, so the async write-queue drain thread's
activity never pollutes the recall path's own count.

The --arm flag toggles IAI_MCP_CROSSING_CONSOLIDATION_OFF so the same
harness measures the pre-consolidation ("legacy", switch ON) and
consolidated ("new", switch OFF) code paths back-to-back in one process.
Before any consolidation lands, both arms report the same total — that
equality is the harness's own self-check.

--warm-bundle builds the runtime graph bundle in-process before measuring,
so hop1/hop2 neighbor lookups serve from the in-RAM adjacency instead of
falling through to a DB scan (the cold-bundle artifact a bare MemoryStore
construction with no daemon otherwise always hits).

Reinforcement/provenance writes are a normal side effect of the real
dispatch() path with async writes enabled (matching bench/full_recall_
latency_probe.py's own methodology and the SC-2 probe) — the harness runs
the production recall path unmodified, it does not insert, delete, or
otherwise reshape any record itself.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import traceback
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

DEFAULT_CUE = "what did we discuss about the project roadmap recently"
CROSSING_KILL_SWITCH_ENV = "IAI_MCP_CROSSING_CONSOLIDATION_OFF"


class _CrossingCounters:
    """Thread-keyed counters for one measured arm."""

    def __init__(self) -> None:
        self.main_ident = threading.get_ident()
        self.in_dispatch = False
        self.ro_conn_acq_main = 0
        self.ro_conn_acq_other = 0
        self.ro_execute_main = 0
        self.ro_execute_other = 0
        self.probe_execute_main = 0
        self.probe_execute_other = 0
        self.writer_execute_main = 0
        self.writer_execute_other = 0
        self.call_sites: dict[str, int] = {}

    def reset(self) -> None:
        self.ro_conn_acq_main = self.ro_conn_acq_other = 0
        self.ro_execute_main = self.ro_execute_other = 0
        self.probe_execute_main = self.probe_execute_other = 0
        self.writer_execute_main = self.writer_execute_other = 0
        self.call_sites.clear()

    def on_main(self) -> bool:
        return threading.get_ident() == self.main_ident and self.in_dispatch

    @property
    def total_main(self) -> int:
        return self.ro_execute_main + self.probe_execute_main + self.writer_execute_main


def _install_patches(counters: _CrossingCounters):
    """Patch the four seams; return a restore callback."""
    from iai_mcp.hippo._db import HippoDB, _MutationSignallingConn
    from iai_mcp.hippo._ro_pool import RoConnPool

    orig_ro_conn = HippoDB.ro_conn
    orig_fence_execute = RoConnPool._FenceRetryingConn.execute
    orig_writer_execute = _MutationSignallingConn.execute
    orig_borrow_prepared = RoConnPool._borrow_prepared

    def counted_borrow_prepared(self, slot):
        if counters.on_main():
            counters.probe_execute_main += 1
        else:
            counters.probe_execute_other += 1
        return orig_borrow_prepared(self, slot)

    def counted_ro_conn(self):
        if counters.on_main():
            counters.ro_conn_acq_main += 1
            frame = traceback.extract_stack()[-2]
            key = f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}"
            counters.call_sites[key] = counters.call_sites.get(key, 0) + 1
        else:
            counters.ro_conn_acq_other += 1
        return orig_ro_conn(self)

    def counted_fence_execute(self, sql, params=()):
        if counters.on_main():
            counters.ro_execute_main += 1
        else:
            counters.ro_execute_other += 1
        return orig_fence_execute(self, sql, params)

    def counted_writer_execute(self, *args, **kwargs):
        if counters.on_main():
            counters.writer_execute_main += 1
        else:
            counters.writer_execute_other += 1
        return orig_writer_execute(self, *args, **kwargs)

    RoConnPool._borrow_prepared = counted_borrow_prepared
    HippoDB.ro_conn = counted_ro_conn
    RoConnPool._FenceRetryingConn.execute = counted_fence_execute
    _MutationSignallingConn.execute = counted_writer_execute

    def restore() -> None:
        RoConnPool._borrow_prepared = orig_borrow_prepared
        HippoDB.ro_conn = orig_ro_conn
        RoConnPool._FenceRetryingConn.execute = orig_fence_execute
        _MutationSignallingConn.execute = orig_writer_execute

    return restore


def _run_dispatch_n(store, counters: _CrossingCounters, n: int, cue: str) -> None:
    import iai_mcp.pipeline as _pm
    from iai_mcp import core

    for _ in range(n):
        _pm._last_recall_latency_ms = 0.0
        counters.in_dispatch = True
        try:
            core.dispatch(
                store, "memory_recall",
                {"cue": cue, "session_id": "crossing-count-probe"},
            )
        finally:
            counters.in_dispatch = False


def measure_arm(store, n: int, cue: str, *, legacy: bool) -> _CrossingCounters:
    """Measure one arm against an already-open store: legacy=True forces
    the pre-consolidation path (switch ON), legacy=False selects the
    default/new path (switch OFF)."""
    if legacy:
        os.environ[CROSSING_KILL_SWITCH_ENV] = "1"
    else:
        os.environ.pop(CROSSING_KILL_SWITCH_ENV, None)

    counters = _CrossingCounters()
    restore = _install_patches(counters)
    try:
        _run_dispatch_n(store, counters, 1, cue)  # warm-up, discarded
        counters.reset()
        _run_dispatch_n(store, counters, n, cue)
    finally:
        restore()
    return counters


def _print_arm_report(label: str, counters: _CrossingCounters, n: int) -> None:
    print(f"--- {label}: n={n} recalls, warm ---")
    print(
        f"  ro_conn() acquisitions: total={counters.ro_conn_acq_main}, "
        f"per-recall avg={counters.ro_conn_acq_main / n:.2f}"
    )
    print(
        f"  RO pooled conn.execute() (caller statements): "
        f"total={counters.ro_execute_main}, per-recall avg={counters.ro_execute_main / n:.2f}"
    )
    print(
        f"  RO pool borrow-time fence PROBE execute: "
        f"total={counters.probe_execute_main}, per-recall avg={counters.probe_execute_main / n:.2f}"
    )
    print(
        f"  writer conn.execute() (still-synchronous): "
        f"total={counters.writer_execute_main}, per-recall avg={counters.writer_execute_main / n:.2f}"
    )
    print(f"  TOTAL synchronous engine execute() calls: per-recall avg={counters.total_main / n:.2f}")
    print("  [ro_conn() acquisition call sites]")
    for site, cnt in sorted(counters.call_sites.items(), key=lambda kv: -kv[1]):
        print(f"    {site}: {cnt} total ({cnt / n:.2f}/recall)")


def run(store_path: str, n: int, arm: str, warm_bundle: bool, cue: str) -> dict[str, float]:
    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    sp = Path(store_path)
    if not sp.exists():
        print(f"ERROR: store_path does not exist: {sp}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening isolated store: {sp}")
    store = MemoryStore(path=sp, access_mode=AccessMode.SHARED)
    asyncio.run(store.enable_async_writes())

    try:
        bundle_label = "cold-bundle"
        if warm_bundle:
            from iai_mcp import retrieve
            retrieve.build_runtime_graph(store)
            bundle_label = "warm-bundle"
            print("Warm graph bundle built in-process (store._warm_graph_bundle populated).")

        results: dict[str, _CrossingCounters] = {}
        try:
            if arm in ("new", "both"):
                results["new"] = measure_arm(store, n, cue, legacy=False)
                _print_arm_report(f"{bundle_label} / new (switch OFF)", results["new"], n)
            if arm in ("legacy", "both"):
                results["legacy"] = measure_arm(store, n, cue, legacy=True)
                _print_arm_report(f"{bundle_label} / legacy (switch ON)", results["legacy"], n)
        finally:
            os.environ.pop(CROSSING_KILL_SWITCH_ENV, None)

        if "new" in results and "legacy" in results:
            new_avg = results["new"].total_main / n
            legacy_avg = results["legacy"].total_main / n
            print("--- delta (legacy - new) ---")
            print(f"  new={new_avg:.2f}  legacy={legacy_avg:.2f}  delta={legacy_avg - new_avg:.2f}")

        pool = getattr(store.db, "_ro_pool", None)
        if pool is not None:
            print(
                f"[RoConnPool health, cumulative whole run] "
                f"fence_reopen_count={pool.fence_reopen_count} "
                f"slot_refresh_count={pool.slot_refresh_count} "
                f"slot_reopen_count={pool.slot_reopen_count} "
                f"writer_fallback_count={pool.writer_fallback_count}"
            )

        return {k: v.total_main / n for k, v in results.items()}
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Honest per-recall synchronous engine execute() count."
    )
    parser.add_argument(
        "--store-path", default=os.environ.get("IAI_ISO"),
        help="Isolated real-store copy (defaults to $IAI_ISO, NEVER live prod)",
    )
    parser.add_argument("--n", type=int, default=20, help="Warm recalls per arm (default 20)")
    parser.add_argument(
        "--arm", choices=["new", "legacy", "both"], default="new",
        help="Which arm(s) to measure via IAI_MCP_CROSSING_CONSOLIDATION_OFF",
    )
    parser.add_argument(
        "--warm-bundle", action="store_true",
        help="Build the graph bundle in-process before measuring (warm-daemon steady-state)",
    )
    parser.add_argument("--cue", default=DEFAULT_CUE)
    args = parser.parse_args()

    if os.environ.get("LILLI_STORAGE_DRIVER") != "lilli":
        print(
            "ERROR: this probe measures the production driver's honest count; "
            "run with LILLI_STORAGE_DRIVER=lilli.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.store_path:
        print("ERROR: no store path -- set $IAI_ISO or pass --store-path", file=sys.stderr)
        sys.exit(1)

    run(args.store_path, args.n, args.arm, args.warm_bundle, args.cue)


if __name__ == "__main__":
    main()
