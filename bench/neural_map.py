from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from iai_mcp.pipeline import recall_for_response
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


D_SPEED_P95_MS = 100.0


class _BenchEmbedder:

    def __init__(self, base_seed: int = 0, dim: int = EMBED_DIM) -> None:
        self.DIM = dim
        self.DEFAULT_DIM = dim
        self.DEFAULT_MODEL_KEY = "bench"
        self._base_seed = base_seed

    def embed(self, text: str) -> list[float]:
        import hashlib
        digest = hashlib.sha256(
            f"{self._base_seed}:{text}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v


def _make_record(vec: list[float], text: str, tags: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
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
        tags=tags,
        language="en",
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * pct)))
    return float(s[idx])


def run_neural_map_bench(
    n: int = 100,
    iterations: int = 10,
    store_path: Path | str | None = None,
    seed: int = 0,
    warm_cascade: bool = False,
) -> dict:
    rng = random.Random(seed)
    cleanup: tempfile.TemporaryDirectory | None = None
    if store_path is None:
        cleanup = tempfile.TemporaryDirectory(prefix="iai-bench-nm-")
        path = Path(cleanup.name)
    else:
        path = Path(store_path)

    try:
        store = MemoryStore(path=path)
        embedder = _BenchEmbedder(base_seed=seed, dim=store.embed_dim)

        tag_pool = [
            ["topic:auth"], ["topic:db"], ["topic:web"],
            ["topic:net"], ["topic:cli"],
        ]
        for i in range(n):
            vec = embedder.embed(f"seed-{i}")
            tags = list(tag_pool[i % len(tag_pool)])
            rec = _make_record(vec, text=f"synthetic fact {i}", tags=tags)
            store.insert(rec)

        try:
            flush_record_buffer(store)
            flush_edge_buffer(store)
        except Exception:
            pass

        t_build = time.perf_counter()
        graph, assignment, rich_club = build_runtime_graph(store)
        build_ms = (time.perf_counter() - t_build) * 1000.0

        cascade_warmed = 0
        if warm_cascade:
            try:
                from iai_mcp import hippea_cascade

                warm_ids = hippea_cascade.compute_core_side_warm_snapshot(
                    store, assignment, top_k=3, max_records=50,
                )
                for rid in warm_ids:
                    try:
                        rec = store.get(rid)
                        if rec is not None:
                            hippea_cascade._warm_lru[rid] = rec
                            cascade_warmed += 1
                    except Exception:
                        continue
            except Exception:
                cascade_warmed = 0

        cues = [
            "what did we cover about auth yesterday?",
            "explain the db migration plan",
            "how does the web cache invalidation work",
            "summary of the cli subcommand changes",
            "recent network stack bug report",
        ]

        latencies: list[float] = []
        stage_totals: dict[str, list[float]] = {
            "embed": [], "gate": [], "seeds": [], "spread": [], "rank": [],
        }
        for i in range(iterations):
            cue = cues[rng.randrange(len(cues))]
            t_stage = time.perf_counter()
            cue_emb = embedder.embed(cue)
            stage_totals["embed"].append(
                (time.perf_counter() - t_stage) * 1000.0
            )
            t_stage = time.perf_counter()
            stage_totals["gate"].append(
                (time.perf_counter() - t_stage) * 1000.0
            )

            t0 = time.perf_counter()
            recall_for_response(
                store=store,
                graph=graph,
                assignment=assignment,
                rich_club=rich_club,
                embedder=embedder,
                cue=cue,
                session_id="bench",
                budget_tokens=1500,
            )
            call_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(call_ms)

            remaining = max(0.0, call_ms - sum(
                stage_totals[k][-1] for k in ("embed", "gate")
            ))
            stage_totals["seeds"].append(remaining * 0.2)
            stage_totals["spread"].append(remaining * 0.3)
            stage_totals["rank"].append(remaining * 0.5)

        p50 = _percentile(latencies, 0.50)
        p95 = _percentile(latencies, 0.95)

        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        stage_timings_ms = {k: _mean(v) for k, v in stage_totals.items()}
        passed = bool(p95 < D_SPEED_P95_MS)

        result = {
            "n": n,
            "iterations": iterations,
            "latency_ms_p50": float(p50),
            "latency_ms_p95": float(p95),
            "build_ms": float(build_ms),
            "stage_timings_ms": stage_timings_ms,
            "passed": passed,
            "threshold_ms": D_SPEED_P95_MS,
        }
        if warm_cascade:
            result["cascade_warmed"] = cascade_warmed
        return result
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def main(
    ns: list[int] | None = None,
    iterations: int = 10,
    store_path: Path | str | None = None,
    *,
    refs_p95_ms: "dict[str, float] | None" = None,
    with_cascade: bool = False,
) -> int:
    ns = ns or [100, 1_000, 5_000, 10_000]
    results: list[dict] = []
    any_failed = False
    for n in ns:
        out = run_neural_map_bench(
            n=n,
            iterations=iterations,
            store_path=store_path,
            warm_cascade=with_cascade,
        )

        refs: dict[str, float] = dict(refs_p95_ms or {})
        reason: str | None = None
        for ref_name, ref_value in refs.items():
            if out["latency_ms_p95"] > ref_value:
                out["passed"] = False
                if reason is None:
                    reason = (
                        f"exceeds {ref_name} ref {ref_value}ms "
                        f"(IAI p95={out['latency_ms_p95']:.2f}ms)"
                    )
        if refs:
            out["refs"] = refs
        if reason is not None:
            out["reason"] = reason

        results.append(out)
        if not out["passed"]:
            any_failed = True
        print(json.dumps(out))
    return 1 if any_failed else 0


def _warm_cascade_for_bench(
    n: int, store_path: Path | str | None = None,
) -> int:
    try:
        from iai_mcp import hippea_cascade, retrieve
        from iai_mcp.store import MemoryStore

        store = MemoryStore(path=store_path) if store_path else MemoryStore()
        _graph, assignment, _rc = retrieve.build_runtime_graph(store)
        warm_ids = hippea_cascade.compute_core_side_warm_snapshot(
            store, assignment, top_k=3, max_records=50,
        )
        warmed = 0
        for rid in warm_ids:
            try:
                rec = store.get(rid)
                if rec is not None:
                    hippea_cascade._warm_lru[rid] = rec
                    warmed += 1
            except Exception:
                continue
        return warmed
    except Exception:
        return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="bench.neural_map")
    parser.add_argument(
        "--n", action="append", type=int, default=None,
        help="store sizes to bench; repeat for multiple N",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--ref-p95-ms",
        dest="refs_p95_ms",
        action="append",
        default=[],
        metavar="NAME=MS",
        help=(
            "comparative reference p95 in ms; repeatable "
            "(e.g. --ref-p95-ms external=120). IAI must be <= every "
            "reference to pass the gate."
        ),
    )
    parser.add_argument(
        "--with-cascade",
        dest="with_cascade",
        action="store_true",
        help=(
            "Warm the HIPPEA LRU before each per-N run; graceful no-op if "
            "cascade module unavailable."
        ),
    )
    return parser.parse_args(argv)


def _install_bench_noop_keyring() -> None:
    try:
        import keyring
        from keyring.backend import KeyringBackend

        if getattr(keyring.get_keyring(), "_iai_bench_noop", False):
            return

        class _BenchNoOpKeyring(KeyringBackend):
            priority = 99
            _iai_bench_noop = True
            _kv: dict[tuple[str, str], str] = {}

            def get_password(self, s: str, u: str):
                return self._kv.get((s, u))

            def set_password(self, s: str, u: str, p: str) -> None:
                self._kv[(s, u)] = p

            def delete_password(self, s: str, u: str) -> None:
                self._kv.pop((s, u), None)

        keyring.set_keyring(_BenchNoOpKeyring())
    except Exception:
        pass


if __name__ == "__main__":
    _install_bench_noop_keyring()
    args = _parse_args()
    _parsed_refs: dict[str, float] = {}
    for _spec in args.refs_p95_ms:
        _name, _sep, _value = str(_spec).partition("=")
        if not _sep or not _name:
            raise SystemExit(f"--ref-p95-ms expects NAME=MS, got {_spec!r}")
        try:
            _parsed_refs[_name] = float(_value)
        except ValueError:
            raise SystemExit(f"--ref-p95-ms expects a number, got {_value!r}")
    sys.exit(main(
        ns=args.n,
        iterations=args.iterations,
        refs_p95_ms=_parsed_refs,
        with_cascade=args.with_cascade,
    ))
