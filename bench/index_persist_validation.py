"""Persisted-index cold-start validation: sidecar-present vs sidecar-absent
fresh-process arms.

Usage (against an isolated store copy — NEVER live prod):

    source .venv/bin/activate
    LILLI_STORAGE_DRIVER=lilli .venv/bin/python bench/index_persist_validation.py \
        --run-all --source ~/.iai-mcp-isotest --workdir /tmp/scratch

The storage engine can build, persist, and load its secondary column index
(the ``col IN (...)`` candidate index used by recall) to an on-disk sidecar,
gated by a per-table write-generation fingerprint. A loaded sidecar is
complete by construction — every row's every indexed value was filed at
persist time — so the prediction under test is exact: a fresh process that
opens a copy with the sidecar present should serve ANY indexed read,
including one that never happened before the persist, with zero full-table
scans. This harness measures exactly that, contrasted against a control arm
that opens a copy with no sidecar (the lazy, incremental-build path).

Two modes:

  --arm control|warm --store <root> --out <json>
      The per-process arm body. Meant to be run inside a freshly spawned
      subprocess (never invoked directly against prod — see the refusal
      guard below). Opens the store EXCLUSIVE with deferred reinforce (the
      daemon's own recall model), then fires a fixed, frozen cue list as
      sequential recall dispatches, timing each and sampling the engine's
      full-table-scan counter (lilli only) after every recall. Never calls
      the host boot warm-up — that is a separate, unrelated accelerator.

  --arm prep --store <root> --out <json>
      The warm-arm's one-time preparation subprocess: opens the copy,
      builds and persists the column-index sidecar, records the persist
      duration and the sidecar's on-disk byte size, and closes cleanly
      without ever touching the recall path. Must run before the warm
      arm's measurement subprocess opens the SAME copy — no writer runs
      between prep-close and measurement-open, so the sidecar's stamp
      cannot diverge before it is read.

  --run-all --source <isotest root> --workdir <scratch>
      The orchestrating mode. Records the real prod store's stat pre/post
      (never opening it). For the control set, makes THREE separate fresh
      disposable working copies (one per run), deletes any sidecar the
      copy operation may have carried over, and runs one control-arm
      subprocess against each. For the warm set, makes THREE separate
      fresh disposable working copies, runs one prep subprocess against
      each to persist its own sidecar, then one warm-arm measurement
      subprocess against that same copy. Every measurement run gets its
      own copy — never reused — because a run's own committed writes
      (deferred-reinforce drain, event/telemetry rows) advance the
      write-generation and would silently discard a reused copy's sidecar
      for the next run. All six copies are made and consumed strictly
      sequentially — one process alive at a time.

Prod refusal: the arm body hard-exits before opening anything if the
resolved store root is the real home store, or if the isolation env is not
set. This guard lives in the arm body itself (not only the orchestrator) so
a mistaken direct invocation can never touch prod.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path preamble -- no-op when imported as package member
# ---------------------------------------------------------------------------
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

# ---------------------------------------------------------------------------
# The frozen cue set is IMPORTED from the prior validation harness so the
# sha256 is identical by construction (never copy-pasted, never re-picked).
# ---------------------------------------------------------------------------
from bench.boot_warmup_validation import FIXED_CUES, cue_set_sha256  # noqa: E402

_EXPECTED_CUE_SHA256 = (
    "a100d35abaa253a8fd4ec4a7f66783dff03414eb590c004e2c76e59bcce1f6ef"
)


def _verify_cue_sha256() -> None:
    actual = cue_set_sha256()
    if actual != _EXPECTED_CUE_SHA256:
        print(
            f"REFUSED: imported cue set sha256 mismatch (expected "
            f"{_EXPECTED_CUE_SHA256}, got {actual}) — the frozen cue tuple "
            "must never be re-picked.",
            file=sys.stderr,
        )
        sys.exit(2)


_verify_cue_sha256()


# ---------------------------------------------------------------------------
# Prod refusal guard (same pattern as boot_warmup_validation.py)
# ---------------------------------------------------------------------------


def _resolve_prod_root() -> Path:
    return Path.home() / ".iai-mcp"


def _refuse_if_prod(store_root: Path) -> None:
    """Hard-exit if store_root resolves to the real prod store.

    Lives in the arm body itself (not only the orchestrator) so a mistaken
    direct invocation of --arm can never open prod.
    """
    prod_root = _resolve_prod_root()
    try:
        resolved = store_root.expanduser().resolve()
    except OSError:
        resolved = store_root.expanduser()
    if resolved == prod_root.resolve():
        print(
            f"REFUSED: store root resolves to the prod store ({prod_root}). "
            "This harness never opens prod.",
            file=sys.stderr,
        )
        sys.exit(2)


def _require_isolation_env() -> None:
    """The arm body also refuses to run without the isolation env set.

    IAI_MCP_STORE must be set (even though --store is also passed
    explicitly) so a stray import that falls back to the env-derived
    default can never silently resolve to the home store.
    """
    if not os.environ.get("IAI_MCP_STORE"):
        print(
            "REFUSED: IAI_MCP_STORE is not set in this process's environment.",
            file=sys.stderr,
        )
        sys.exit(2)


def _sidecar_path(store_root: Path) -> Path:
    return store_root / "hippo" / "records.colindex"


# ---------------------------------------------------------------------------
# Per-process arm bodies
# ---------------------------------------------------------------------------


def _get_lilli_conn(db):
    """Return the underlying lilli engine connection, or None if unavailable."""
    conn = getattr(db, "_conn", None)
    if conn is None:
        return None
    if not hasattr(conn, "full_scan_count"):
        return None
    return conn


def _require_persist_surface(conn) -> None:
    if conn is None or not hasattr(conn, "persist_col_indexes"):
        print(
            "REFUSED: the engine Connection is missing persist_col_indexes "
            "— stale native build. Run `cd rust/iai_mcp_native && maturin "
            "develop --release` before measuring.",
            file=sys.stderr,
        )
        sys.exit(2)


def run_prep(store_root: Path) -> dict:
    """Build + persist the column-index sidecar on a fresh copy.

    Runs exactly once per warm-arm measurement run, on that run's own
    dedicated copy, immediately before the measurement subprocess opens it.
    Never fires a recall — only builds and persists the index.
    """
    _refuse_if_prod(store_root)
    _require_isolation_env()

    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    result: dict = {
        "mode": "prep",
        "store_root": str(store_root),
        "pid": os.getpid(),
    }

    store = MemoryStore(path=store_root, access_mode=AccessMode.EXCLUSIVE)
    conn = _get_lilli_conn(store.db)
    _require_persist_surface(conn)

    lock = getattr(store.db, "_conn_lock", None)
    from contextlib import nullcontext

    lock_ctx = lock if lock is not None else nullcontext()

    t0 = time.perf_counter()
    with lock_ctx:
        conn.persist_col_indexes()
    persist_ms = (time.perf_counter() - t0) * 1000.0
    result["persist_ms"] = persist_ms

    try:
        store.close()
    except Exception:  # noqa: BLE001 -- teardown must not mask the measurement
        pass

    sidecar = _sidecar_path(store_root)
    result["sidecar_exists"] = sidecar.exists()
    result["sidecar_bytes"] = sidecar.stat().st_size if sidecar.exists() else 0

    return result


def run_arm(arm: str, store_root: Path) -> dict:
    """Run one fresh-process measurement arm (control or warm).

    Must be called from a process that has just started — the cold-index
    tax is per-process, so this function assumes nothing has touched the
    store yet in this interpreter. Never calls the host boot warm-up (an
    unrelated accelerator, deliberately excluded from both arms here).
    """
    _refuse_if_prod(store_root)
    _require_isolation_env()

    from iai_mcp import core
    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    result: dict = {
        "arm": arm,
        "store_root": str(store_root),
        "cue_set_sha256": cue_set_sha256(),
        "pid": os.getpid(),
        "sidecar_present_at_open": _sidecar_path(store_root).exists(),
    }

    t_open0 = time.perf_counter()
    store = MemoryStore(path=store_root, access_mode=AccessMode.EXCLUSIVE)
    store.enable_reinforce_queue()
    result["store_open_ms"] = (time.perf_counter() - t_open0) * 1000.0

    lconn = _get_lilli_conn(store.db)
    result["lilli_scan_counter_available"] = lconn is not None
    if lconn is not None:
        lconn.reset_full_scan_count()

    recalls: list[dict] = []
    for i, cue in enumerate(FIXED_CUES):
        t0 = time.perf_counter()
        resp = core.dispatch(
            store, "memory_recall", {"cue": cue, "session_id": "index_persist_validation"}
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        hits = resp.get("hits", [])
        full_scan_count = lconn.full_scan_count() if lconn is not None else None
        recalls.append(
            {
                "index": i,
                "cue": cue,
                "elapsed_ms": elapsed_ms,
                "hits": len(hits),
                "full_scan_count_after": full_scan_count,
                "degraded": bool(resp.get("_degraded", False)),
            }
        )

    result["recalls"] = recalls

    try:
        store.close()
    except Exception:  # noqa: BLE001 -- teardown must not mask the measurement
        pass

    return result


def _arm_main(args: argparse.Namespace) -> None:
    store_root = Path(args.store).expanduser()
    if args.arm == "prep":
        result = run_prep(store_root)
    else:
        result = run_arm(args.arm, store_root)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Arm '{args.arm}' complete -> {out_path}")


# ---------------------------------------------------------------------------
# Orchestrating mode
# ---------------------------------------------------------------------------


def _stat_or_none(path: Path) -> dict | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _spawn(mode_args: list[str], store_root: Path, out_path: Path, label: str) -> dict:
    """Spawn one fresh subprocess and load its JSON result."""
    env = copy.deepcopy(dict(os.environ))
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(store_root / ".daemon.sock")
    env["LILLI_STORAGE_DRIVER"] = "lilli"

    cmd = [sys.executable, str(Path(__file__).resolve()), *mode_args]
    print(f"[run-all] spawning {label} ...")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    print(proc.stdout.strip())
    return json.loads(out_path.read_text())


def _fresh_copy(source: Path, workdir: Path, tag: str) -> Path:
    ts = int(time.time() * 1000)
    copy_root = workdir / f"iai-idxval-{tag}-{ts}"
    print(f"[run-all] making fresh working copy: {source} -> {copy_root}")
    subprocess.run(["cp", "-R", str(source), str(copy_root)], check=True)
    return copy_root


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _aggregate_arm(runs: list[dict]) -> dict:
    first_recall_ms = [r["recalls"][0]["elapsed_ms"] for r in runs]
    per_recall_medians = []
    n_recalls = len(runs[0]["recalls"]) if runs else 0
    for idx in range(n_recalls):
        per_recall_medians.append(_median([r["recalls"][idx]["elapsed_ms"] for r in runs]))
    full_scan_after = [[rc["full_scan_count_after"] for rc in r["recalls"]] for r in runs]
    hits_after = [[rc["hits"] for rc in r["recalls"]] for r in runs]
    return {
        "n_runs": len(runs),
        "first_recall_ms_all": first_recall_ms,
        "first_recall_ms_median": _median(first_recall_ms),
        "per_recall_ms_median": per_recall_medians,
        "full_scan_count_after_all": full_scan_after,
        "hits_all": hits_after,
    }


def run_all(source: Path, workdir: Path) -> dict:
    prod_root = _resolve_prod_root()
    prod_db = prod_root / "hippo" / "brain.sqlite3"
    prod_stat_pre = _stat_or_none(prod_db)

    workdir.mkdir(parents=True, exist_ok=True)

    control_runs: list[dict] = []
    control_copies: list[str] = []
    warm_runs: list[dict] = []
    warm_copies: list[str] = []
    prep_runs: list[dict] = []

    # -- CONTROL arm: 3 runs, each on its OWN fresh copy, sidecar absent. --
    for i in range(3):
        copy_root = _fresh_copy(source, workdir, f"control-{i}")
        control_copies.append(str(copy_root))
        sidecar = _sidecar_path(copy_root)
        if sidecar.exists():
            sidecar.unlink()
            print(f"[run-all] deleted carried-over sidecar at {sidecar}")

        out_path = workdir / f"control-{i}.json"
        result = _spawn(
            ["--arm", "control", "--store", str(copy_root), "--out", str(out_path)],
            copy_root,
            out_path,
            f"control arm run #{i}",
        )
        control_runs.append(result)
        shutil.rmtree(copy_root, ignore_errors=True)

    # -- WARM arm: 3 runs, each on its OWN fresh copy, own prep subprocess. --
    for i in range(3):
        copy_root = _fresh_copy(source, workdir, f"warm-{i}")
        warm_copies.append(str(copy_root))

        prep_out = workdir / f"prep-{i}.json"
        prep_result = _spawn(
            ["--arm", "prep", "--store", str(copy_root), "--out", str(prep_out)],
            copy_root,
            prep_out,
            f"warm arm prep #{i}",
        )
        prep_runs.append(prep_result)

        out_path = workdir / f"warm-{i}.json"
        result = _spawn(
            ["--arm", "warm", "--store", str(copy_root), "--out", str(out_path)],
            copy_root,
            out_path,
            f"warm arm measurement run #{i}",
        )
        warm_runs.append(result)
        shutil.rmtree(copy_root, ignore_errors=True)

    prod_stat_post = _stat_or_none(prod_db)

    control_agg = _aggregate_arm(control_runs)
    warm_agg = _aggregate_arm(warm_runs)

    control_first_median = control_agg["first_recall_ms_median"]
    warm_first_median = warm_agg["first_recall_ms_median"]

    # Flatness gate: every warm run's full_scan_count must be 0 after
    # EVERY recall (not just non-growing — flat AT ZERO).
    full_scan_flat_zero = True
    for run in warm_runs:
        for rc in run["recalls"]:
            if rc["full_scan_count_after"] not in (0, None):
                full_scan_flat_zero = False

    # per-recall warm medians all <= 3.0s
    warm_per_recall_medians = warm_agg["per_recall_ms_median"]
    warm_all_under_budget = all(m <= 3000.0 for m in warm_per_recall_medians)

    # hit-count equality between arms (sidecar changes scan cost, never
    # results): compare per-recall hit counts, control vs warm, per index.
    hit_counts_equal = True
    n_recalls = len(control_runs[0]["recalls"]) if control_runs else 0
    control_hits_by_idx = [
        [r["recalls"][idx]["hits"] for r in control_runs] for idx in range(n_recalls)
    ]
    warm_hits_by_idx = [
        [r["recalls"][idx]["hits"] for r in warm_runs] for idx in range(n_recalls)
    ]
    for idx in range(n_recalls):
        c_set = set(control_hits_by_idx[idx])
        w_set = set(warm_hits_by_idx[idx])
        if c_set != w_set:
            hit_counts_equal = False

    pass_conditions = {
        "full_scan_flat_zero_every_warm_recall": full_scan_flat_zero,
        "warm_first_recall_median_le_3s": warm_first_median <= 3000.0,
        "warm_per_recall_medians_all_le_3s": warm_all_under_budget,
        "hit_counts_equal_between_arms": hit_counts_equal,
    }
    verdict = "PASS" if all(pass_conditions.values()) else "ESCALATE"

    aggregate: dict = {
        "cue_set_sha256": cue_set_sha256(),
        "cues": list(FIXED_CUES),
        "control_copies": control_copies,
        "warm_copies": warm_copies,
        "prod_stat_pre": prod_stat_pre,
        "prod_stat_post": prod_stat_post,
        "prod_untouched": prod_stat_pre == prod_stat_post,
        "prep_runs": prep_runs,
        "control_runs": control_runs,
        "warm_runs": warm_runs,
        "control_aggregate": control_agg,
        "warm_aggregate": warm_agg,
        "control_first_recall_ms_median": control_first_median,
        "warm_first_recall_ms_median": warm_first_median,
        "pass_conditions": pass_conditions,
        "verdict": verdict,
    }
    return aggregate


def _run_all_main(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser()
    workdir = Path(args.workdir).expanduser()
    aggregate = run_all(source, workdir)
    print("\nJSON summary:")
    print(json.dumps(aggregate, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Persisted-index cold-start validation: sidecar-present vs "
            "sidecar-absent fresh-process arms on an isolated store copy. "
            "NEVER live prod."
        )
    )
    parser.add_argument(
        "--arm", choices=["control", "warm", "prep"], help="Run a single arm body"
    )
    parser.add_argument("--store", help="Store root for the --arm mode")
    parser.add_argument("--out", help="JSON output path for the --arm mode")
    parser.add_argument("--run-all", action="store_true", help="Run the full control+warm sequence")
    parser.add_argument("--source", help="Source isolated copy to duplicate (--run-all mode)")
    parser.add_argument("--workdir", help="Scratch directory for disposable working copies (--run-all mode)")
    args = parser.parse_args()

    if args.run_all:
        if not args.source or not args.workdir:
            parser.error("--run-all requires --source and --workdir")
        _run_all_main(args)
        return

    if args.arm:
        if not args.store or not args.out:
            parser.error("--arm requires --store and --out")
        _arm_main(args)
        return

    parser.error("must specify either --arm or --run-all")


if __name__ == "__main__":
    main()
