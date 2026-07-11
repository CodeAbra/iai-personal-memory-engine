from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
    os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
        "iai-mcp-bench-falsifiability-deterministic-2026"
    )

from iai_mcp.types import EMBED_DIM, MemoryRecord  # noqa: E402


def _cur_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return _ru_maxrss_bytes()


def _ru_maxrss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(r)
    return int(r) * 1024


class _RSSSampler:

    def __init__(self, interval_sec: float = 0.05) -> None:
        self._interval = interval_sec
        self._peak = 0
        self._count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = _cur_rss_bytes()
            if v > self._peak:
                self._peak = v
            self._count += 1
            self._stop.wait(self._interval)

    def __enter__(self) -> "_RSSSampler":
        self._peak = _cur_rss_bytes()
        self._count = 1
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        v = _cur_rss_bytes()
        if v > self._peak:
            self._peak = v
        self._count += 1

    @property
    def peak_bytes(self) -> int:
        return self._peak

    @property
    def sample_count(self) -> int:
        return self._count


def _make_record(i: int, rng: np.random.Generator, dim: int) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    vec = rng.standard_normal(dim)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"alice daily routine note {i}: she logged a habit and a plan",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["bench", "rss-peak"],
        language="en",
    )


def _install_remote_stubs() -> None:
    def _boom(*_a: object, **_kw: object) -> "dict":
        raise RuntimeError(
            "remote subprocess call attempted inside RSS-peak bench "
            "(must never happen — the bench is local-only)"
        )

    import iai_mcp.claude_cli as cc

    cc.invoke_claude_sync = _boom  # type: ignore[assignment]
    cc.invoke_claude_once = _boom  # type: ignore[assignment]

    import iai_mcp.reconsolidation_critic as rc

    def _no_critic(_pool: object, **_kw: object) -> "dict":
        return {}

    rc.evaluate_batch_reconsolidation = _no_critic  # type: ignore[assignment]


def _real_store_signature() -> object:
    real = Path.home() / ".iai-mcp"
    if not real.exists():
        return None
    volatile_prefixes = (
        "wrappers/", "logs/", ".capture-state/", ".deferred-captures/",
        ".daemon.sock", ".daemon-state", ".session-start-payload",
        ".heartbeat", "wake.signal", ".wake.signal", ".locked", ".lock",
    )
    sig: dict[str, tuple[int, int]] = {}
    for p in real.rglob("*"):
        try:
            rel = str(p.relative_to(real))
        except ValueError:
            continue
        if any(rel.startswith(v) or rel == v.rstrip("/") for v in volatile_prefixes):
            continue
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        sig[rel] = (st.st_size, st.st_mtime_ns)
    return sig


def run_peak(
    n: int = 2000,
    dim: int = EMBED_DIM,
    seed: int = 42,
    *,
    with_embedder: bool = True,
) -> dict:
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
    from iai_mcp.store import MemoryStore

    _install_remote_stubs()

    real_sig_before = _real_store_signature()

    tmp_root = Path(tempfile.mkdtemp(prefix="iai-rss-peak-"))
    prev_store = os.environ.get("IAI_MCP_STORE")
    os.environ["IAI_MCP_STORE"] = str(tmp_root)
    prev_embed_dim = os.environ.get("IAI_MCP_EMBED_DIM")
    if dim != EMBED_DIM:
        os.environ["IAI_MCP_EMBED_DIM"] = str(dim)
    prev_user_model = os.environ.get("IAI_MCP_USER_MODEL_PATH")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(tmp_root / "user_model.json")

    store = None
    embedder = None
    try:
        store = MemoryStore(path=tmp_root)
        eff_dim = store.embed_dim

        if with_embedder:
            from iai_mcp.embed import Embedder

            embedder = Embedder()
            embedder.embed("alice daily routine warm-up prime encode")

        rng = np.random.default_rng(seed)
        for i in range(n):
            store.insert(_make_record(i, rng, dim=eff_dim))

        log_dir = tmp_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        event_log = LifecycleEventLog(log_dir=log_dir)
        pipeline = SleepPipeline(
            store=store,
            lifecycle_state_path=tmp_root / "lifecycle_state.json",
            event_log=event_log,
        )

        baseline_rss = _cur_rss_bytes()

        t0 = time.perf_counter()
        with _RSSSampler(interval_sec=0.05) as sampler:
            result = pipeline.run()
        duration = time.perf_counter() - t0

        peak_current = sampler.peak_bytes
        ru_maxrss = _ru_maxrss_bytes()
        sample_count = sampler.sample_count

        if embedder is not None:
            _ = embedder.DIM

        gc.collect()
        steady = _cur_rss_bytes()

        completed = [
            getattr(s, "name", str(s))
            for s in result.get("completed_steps", [])
        ]
        critic_calls = int(result.get("critic_calls", 0) or 0)

        schema_mine_ran = any("SCHEMA_MINE" in c for c in completed)
        optimize_ran = any("OPTIMIZE" in c for c in completed)

        return {
            "n_records": n,
            "peak_current_rss_bytes": int(peak_current),
            "ru_maxrss_bytes": int(ru_maxrss),
            "baseline_rss_bytes": int(baseline_rss),
            "steady_state_after_gc_bytes": int(steady),
            "embedder_resident": bool(with_embedder),
            "completed_steps": completed,
            "schema_mine_ran": schema_mine_ran,
            "optimize_step_ran": optimize_ran,
            "failed_step": (
                getattr(result.get("failed_step"), "name", None)
                if result.get("failed_step") is not None
                else None
            ),
            "error": result.get("error"),
            "critic_calls": critic_calls,
            "remote_rem_stubbed": True,
            "platform": sys.platform,
            "sample_count": int(sample_count),
            "duration_sec": round(duration, 3),
            "embed_dim": int(eff_dim),
        }
    finally:
        if prev_store is None:
            os.environ.pop("IAI_MCP_STORE", None)
        else:
            os.environ["IAI_MCP_STORE"] = prev_store
        if dim != EMBED_DIM:
            if prev_embed_dim is None:
                os.environ.pop("IAI_MCP_EMBED_DIM", None)
            else:
                os.environ["IAI_MCP_EMBED_DIM"] = prev_embed_dim
        if prev_user_model is None:
            os.environ.pop("IAI_MCP_USER_MODEL_PATH", None)
        else:
            os.environ["IAI_MCP_USER_MODEL_PATH"] = prev_user_model
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
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
                "HERMETICITY VIOLATION: ~/.iai-mcp changed during the bench. "
                f"added={added!r} removed={removed!r} modified={modified!r}. "
                "The bench must never touch the real user store."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="consolidation_rss_peak",
        description=(
            "Hermetic full-sleep-pipeline RSS-peak sampler. Seeds N "
            "synthetic records into a throwaway store, drives one in-process "
            "sleep cycle, and reports the peak RSS so a memory watchdog cap "
            "can be set above the legitimate consolidation peak."
        ),
    )
    parser.add_argument(
        "--n", "--n-records", dest="n", type=int, default=2000,
        help="record count to seed (default 2000)",
    )
    parser.add_argument(
        "--dim", type=int, default=EMBED_DIM,
        help=f"embedding dimension (default {EMBED_DIM})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed (default 42)",
    )
    parser.add_argument(
        "--no-embedder", dest="with_embedder", action="store_false",
        help=(
            "Do NOT hold the embedder resident. Off by default: the watched "
            "process keeps the embedder loaded, so the resident-inclusive peak "
            "is the faithful number. Use this only to isolate the "
            "consolidation working-set delta."
        ),
    )
    parser.set_defaults(with_embedder=True)
    parser.add_argument(
        "--out", type=str, default=None,
        help="Write the JSON result to this file (in addition to stdout).",
    )
    args = parser.parse_args(argv)
    result = run_peak(
        n=args.n, dim=args.dim, seed=args.seed,
        with_embedder=args.with_embedder,
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh)
    print(json.dumps(result))
    if result.get("critic_calls", 0) != 0:
        return 2
    if result.get("failed_step") is not None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
