"""Boot warm-up efficacy validation: fresh-subprocess control vs warm arms.

Usage (against an isolated store copy — NEVER live prod):

    source .venv/bin/activate
    LILLI_STORAGE_DRIVER=lilli .venv/bin/python bench/boot_warmup_validation.py \
        --run-all --source ~/.iai-mcp-isotest --workdir /tmp/scratch

The cold-index tax the daemon boot warm-up is meant to collapse is a
per-process cost: a fresh process pays it once, the moment it first touches
each distinct query shape. Measuring that honestly requires spawning a real
fresh subprocess per arm — an in-process loop can never reproduce a cold
process, because the second call in the same process is already warm.

Two modes:

  --arm control|warm --store <root> --out <json>
      The per-process arm body. Meant to be run inside a freshly spawned
      subprocess (never invoked directly against prod — see the refusal
      guard below). Opens the store EXCLUSIVE with deferred reinforce (the
      daemon's own recall model), optionally runs the shipped boot warm-up
      first, then fires a fixed, frozen cue list as sequential recall
      dispatches, timing each and sampling the engine's full-table-scan
      counter (lilli only) after every recall.

  --run-all --source <isotest root> --workdir <scratch>
      The orchestrating mode. Makes one fresh disposable working copy of
      the source store, records the real prod store's stat pre/post (never
      opening it), then runs three control-arm subprocesses followed by
      three warm-arm subprocesses — strictly sequential, one process alive
      at a time — aggregates medians, and prints the verdict.

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
# The fixed cue set — frozen before the first run, never re-picked.
#
# Six varied realistic cues chosen to hit different vec_label neighborhoods:
# a preference, a decision, a project name, a temporal question, a technical
# term, and a person reference. Uses "alice", never real user data.
# ---------------------------------------------------------------------------
FIXED_CUES: tuple[str, ...] = (
    "what does alice prefer for her morning workflow",
    "the decision we made about the database migration rollback plan",
    "project codename for the new recommendation engine",
    "what did we discuss last week about the release schedule",
    "explain the hashing algorithm used for the projection matrix",
    "who is alice's main contact on the infrastructure team",
)


def cue_set_sha256() -> str:
    """Stable sha256 over the frozen cue tuple, emitted in every output."""
    joined = "\x1f".join(FIXED_CUES).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


# ---------------------------------------------------------------------------
# Prod refusal guard
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


# ---------------------------------------------------------------------------
# Per-process arm body
# ---------------------------------------------------------------------------


def _get_lilli_conn(db):
    """Return the underlying lilli engine connection, or None if unavailable."""
    conn = getattr(db, "_conn", None)
    if conn is None:
        return None
    if not hasattr(conn, "full_scan_count"):
        return None
    return conn


def run_arm(arm: str, store_root: Path) -> dict:
    """Run one fresh-process arm (control or warm) and return its result dict.

    Must be called from a process that has just started — the cold-index
    tax is per-process, so this function assumes nothing has touched the
    store yet in this interpreter.
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
    }

    t_open0 = time.perf_counter()
    store = MemoryStore(path=store_root, access_mode=AccessMode.EXCLUSIVE)
    store.enable_reinforce_queue()
    result["store_open_ms"] = (time.perf_counter() - t_open0) * 1000.0

    lconn = _get_lilli_conn(store.db)
    result["lilli_scan_counter_available"] = lconn is not None

    if arm == "warm":
        from iai_mcp.daemon._boot_warmup import run_boot_warmup

        t_warm0 = time.perf_counter()
        warmup_summary = run_boot_warmup(store)
        warmup_wall_ms = (time.perf_counter() - t_warm0) * 1000.0
        result["warmup_wall_ms"] = warmup_wall_ms
        result["warmup_summary"] = warmup_summary
        if lconn is not None:
            lconn.reset_full_scan_count()

    recalls: list[dict] = []
    for i, cue in enumerate(FIXED_CUES):
        t0 = time.perf_counter()
        resp = core.dispatch(store, "memory_recall", {"cue": cue, "session_id": "boot_warmup_validation"})
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


def _spawn_arm(arm: str, store_root: Path, out_path: Path, run_index: int) -> dict:
    """Spawn one fresh subprocess for a single arm run and load its JSON result."""
    env = copy.deepcopy(dict(os.environ))
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(store_root / ".daemon.sock")
    env["LILLI_STORAGE_DRIVER"] = "lilli"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--arm",
        arm,
        "--store",
        str(store_root),
        "--out",
        str(out_path),
    ]
    print(f"[run-all] spawning {arm} arm run #{run_index} ...")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{arm} arm run #{run_index} failed (exit {proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    print(proc.stdout.strip())
    return json.loads(out_path.read_text())


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _aggregate_arm(runs: list[dict]) -> dict:
    first_recall_ms = [r["recalls"][0]["elapsed_ms"] for r in runs]
    third_recall_ms = [r["recalls"][2]["elapsed_ms"] for r in runs]
    per_recall_medians = []
    n_recalls = len(runs[0]["recalls"]) if runs else 0
    for idx in range(n_recalls):
        per_recall_medians.append(_median([r["recalls"][idx]["elapsed_ms"] for r in runs]))
    full_scan_after = [
        [rc["full_scan_count_after"] for rc in r["recalls"]] for r in runs
    ]
    return {
        "n_runs": len(runs),
        "first_recall_ms_all": first_recall_ms,
        "first_recall_ms_median": _median(first_recall_ms),
        "third_recall_ms_all": third_recall_ms,
        "third_recall_ms_median": _median(third_recall_ms),
        "per_recall_ms_median": per_recall_medians,
        "full_scan_count_after_all": full_scan_after,
    }


def run_all(source: Path, workdir: Path) -> dict:
    prod_root = _resolve_prod_root()
    prod_db = prod_root / "hippo" / "brain.sqlite3"
    prod_stat_pre = _stat_or_none(prod_db)

    workdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    copy_root = workdir / f"iai-warmval-{ts}"
    print(f"[run-all] making fresh working copy: {source} -> {copy_root}")
    subprocess.run(["cp", "-R", str(source), str(copy_root)], check=True)

    control_runs: list[dict] = []
    warm_runs: list[dict] = []

    for i in range(3):
        out_path = workdir / f"control-{ts}-{i}.json"
        control_runs.append(_spawn_arm("control", copy_root, out_path, i))

    for i in range(3):
        out_path = workdir / f"warm-{ts}-{i}.json"
        warm_runs.append(_spawn_arm("warm", copy_root, out_path, i))

    prod_stat_post = _stat_or_none(prod_db)

    control_agg = _aggregate_arm(control_runs)
    warm_agg = _aggregate_arm(warm_runs)

    control_first_median = control_agg["first_recall_ms_median"]
    warm_first_median = warm_agg["first_recall_ms_median"]
    warm_third_median = warm_agg["third_recall_ms_median"]

    # full_scan flatness on the warm arm: for each run, check whether the
    # scan counter grows across its own recall sequence (fresh varied cues,
    # unrelated to whatever the warm-up itself touched).
    full_scan_flat = True
    for run in warm_runs:
        counts = [
            rc["full_scan_count_after"]
            for rc in run["recalls"]
            if rc["full_scan_count_after"] is not None
        ]
        if len(counts) >= 2 and counts[-1] > counts[0]:
            full_scan_flat = False

    pass_conditions = {
        "warm_first_recall_le_3s": warm_first_median <= 3000.0,
        "warm_first_recall_le_half_control": (
            control_first_median > 0
            and warm_first_median <= control_first_median / 2.0
        ),
        "full_scan_flat": full_scan_flat,
        "warm_recall3_le_3s": warm_third_median <= 3000.0,
    }
    verdict = "PASS" if all(pass_conditions.values()) else "ESCALATE"

    aggregate: dict = {
        "cue_set_sha256": cue_set_sha256(),
        "cues": list(FIXED_CUES),
        "copy_root": str(copy_root),
        "prod_stat_pre": prod_stat_pre,
        "prod_stat_post": prod_stat_post,
        "prod_untouched": prod_stat_pre == prod_stat_post,
        "control_runs": control_runs,
        "warm_runs": warm_runs,
        "control_aggregate": control_agg,
        "warm_aggregate": warm_agg,
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
            "Boot warm-up efficacy validation: fresh-subprocess control vs "
            "warm arms on an isolated store copy. NEVER live prod."
        )
    )
    parser.add_argument("--arm", choices=["control", "warm"], help="Run a single arm body")
    parser.add_argument("--store", help="Store root for the --arm mode")
    parser.add_argument("--out", help="JSON output path for the --arm mode")
    parser.add_argument("--run-all", action="store_true", help="Run the full control+warm sequence")
    parser.add_argument("--source", help="Source isolated copy to duplicate (--run-all mode)")
    parser.add_argument("--workdir", help="Scratch directory for the disposable working copy (--run-all mode)")
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
