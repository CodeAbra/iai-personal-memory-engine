from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import MAX_PAIRS_PER_CLUSTER, SleepStep

logger = logging.getLogger(__name__)


def step_cluster_replay(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.CLUSTER_REPLAY, 0, interrupt_check,
    ):
        return False, {}

    from iai_mcp.daemon_config import _load_sleep_overhaul_config
    cfg = _load_sleep_overhaul_config()
    window_sec = cfg.cluster_window_sec
    delta = cfg.cluster_replay_initial_weight
    dry_run = cfg.dry_run
    lookback_windows = 5

    from iai_mcp.events import write_event

    now = self._now()
    lookback_cutoff = now - timedelta(seconds=window_sec * lookback_windows)

    # Stream only (id, last_reviewed) within the lookback window instead of
    # materializing the full record corpus. The streaming surface cannot sort
    # in SQL, so order in Python; the secondary id tie-break makes the window
    # walk deterministic on equal timestamps.
    lookback_cutoff_str = lookback_cutoff.strftime("%Y-%m-%d %H:%M:%S")
    try:
        rows = list(
            self._store.iter_record_columns(
                ["id", "last_reviewed"],
                batch_size=2048,
                where=f"last_reviewed >= '{lookback_cutoff_str}'",
            )
        )
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("cluster_replay query failed: %s", exc)
        rows = []
    rows.sort(key=lambda r: (r["last_reviewed"], r["id"]))

    clusters: list[list[Any]] = []
    if rows:
        window_td = timedelta(seconds=window_sec)
        current_cluster: list[Any] = []
        current_window_end = None
        for row in rows:
            ts = row["last_reviewed"]
            try:
                py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            except (TypeError, ValueError, AttributeError):
                py = ts
            if isinstance(py, str):
                try:
                    py = datetime.fromisoformat(py)
                except (TypeError, ValueError):
                    continue
            if getattr(py, "tzinfo", None) is None:
                try:
                    py = py.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError, AttributeError):
                    continue
            if current_window_end is None or py > current_window_end:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [row["id"]]
                current_window_end = py + window_td
            else:
                current_cluster.append(row["id"])
        if current_cluster:
            clusters.append(current_cluster)

    from itertools import combinations
    import uuid as _uuid

    replay_clusters = [c for c in clusters if len(c) >= 2]
    total_pairs = 0
    capped_count = 0
    all_pairs: list[tuple[Any, Any]] = []
    for c in replay_clusters:
        uids = [
            _uuid.UUID(str(x)) if not isinstance(x, _uuid.UUID) else x
            for x in c
        ]
        cluster_pairs = list(combinations(uids, 2))
        if len(cluster_pairs) > MAX_PAIRS_PER_CLUSTER:
            cluster_pairs = cluster_pairs[:MAX_PAIRS_PER_CLUSTER]
            capped_count += 1
        all_pairs.extend(cluster_pairs)
        total_pairs += len(cluster_pairs)

    edges_boosted = 0
    if all_pairs and not dry_run:
        try:
            result_map = self._store.boost_edges(
                all_pairs,
                delta=delta,
                edge_type="hebbian_cluster_replay",
            )
            edges_boosted = len(result_map)
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.error("cluster_replay boost_edges failed: %s", exc, exc_info=True)
            write_event(
                self._store,
                "cluster_replay_pass",
                {
                    "clusters_replayed": len(replay_clusters),
                    "total_edges_boosted": 0,
                    "avg_cluster_size": (
                        sum(len(c) for c in replay_clusters)
                        / len(replay_clusters)
                        if replay_clusters else 0.0
                    ),
                    "window_sec": int(window_sec),
                    "lookback_windows": int(lookback_windows),
                    "max_pairs_per_cluster_applied": int(capped_count),
                    "dry_run_mode": False,
                    "mutation_error": str(exc)[:500],
                },
                severity="warning",
            )
            raise
    elif all_pairs and dry_run:
        edges_boosted = total_pairs

    avg_cluster_size = (
        sum(len(c) for c in replay_clusters) / len(replay_clusters)
        if replay_clusters else 0.0
    )
    write_event(
        self._store,
        "cluster_replay_pass",
        {
            "clusters_replayed": int(len(replay_clusters)),
            "total_edges_boosted": int(edges_boosted),
            "avg_cluster_size": float(avg_cluster_size),
            "window_sec": int(window_sec),
            "lookback_windows": int(lookback_windows),
            "max_pairs_per_cluster_applied": int(capped_count),
            "dry_run_mode": bool(dry_run),
        },
        severity="info",
    )

    seq_pairs_boosted = 0
    if replay_clusters and not dry_run:
        try:
            sequential_pairs: list[tuple[Any, Any]] = []
            for c in replay_clusters:
                uids = [
                    _uuid.UUID(str(x)) if not isinstance(x, _uuid.UUID) else x
                    for x in c
                ]
                for i in range(len(uids) - 1):
                    sequential_pairs.append((uids[i], uids[i + 1]))
            if sequential_pairs:
                self._store.boost_edges(
                    sequential_pairs,
                    delta=delta * 1.5,
                    edge_type="temporal_sequence",
                )
                seq_pairs_boosted = len(sequential_pairs)
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("non-critical temporal trajectory replay failed: %s", exc)

    return True, {
        "clusters_replayed": int(len(replay_clusters)),
        "sequential_pairs": int(seq_pairs_boosted),
        "dry_run": bool(dry_run),
    }
