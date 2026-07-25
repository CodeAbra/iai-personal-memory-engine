from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Literal

RuntimeGraphProof = Literal["detection", "centrality"]

_RSS_ARM_WORKER = textwrap.dedent(
    """
    import json, resource, sys
    from datetime import datetime, timezone
    from pathlib import Path
    from uuid import uuid4

    import numpy as np

    import iai_mcp.community as _cm
    from iai_mcp import retrieve, runtime_graph_cache
    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer
    from iai_mcp.types import MemoryRecord

    proof = sys.argv[1]
    seed_base = int(sys.argv[2])
    in_parent = sys.argv[3] == "in_parent"
    root = Path(sys.argv[4])
    n_records = int(sys.argv[5])

    store = MemoryStore(path=root / "store")
    store.root = root / "root"
    store.root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    for index in range(n_records):
        seed = seed_base + index
        rng = np.random.default_rng(seed)
        vector = rng.random(store.embed_dim).astype(np.float32)
        vector = (vector / np.linalg.norm(vector)).tolist()
        store.insert(MemoryRecord(
            id=uuid4(), tier="episodic",
            literal_surface=f"surface number {seed} carrying real text",
            aaak_index="", embedding=vector, community_id=None, centrality=0.0,
            detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
            last_reviewed=None, never_decay=False, never_merge=False,
            provenance=[], created_at=now, updated_at=now,
            tags=[f"tag{seed % 3}"], language="en",
        ))
    flush_record_buffer(store)
    flush_edge_buffer(store)

    if proof == "detection" and in_parent:
        def _in_parent_detect(graph, **kwargs):
            assignment = _cm.detect_communities(
                graph, prior=None, prior_mode="seeded"
            )
            if kwargs.get("with_centrality"):
                return assignment, graph.centrality()
            return assignment

        runtime_graph_cache.compute_assignment_in_child = _in_parent_detect
    elif proof == "centrality" and in_parent:
        def _local_detect(store, graph, *, with_centrality=False):
            assignment = _cm.detect_communities(
                graph, prior=None, prior_mode="seeded"
            )
            if with_centrality:
                return assignment, None
            return assignment

        retrieve._detect_communities_isolated = _local_detect
        runtime_graph_cache.compute_centrality_in_child = (
            lambda graph, **kwargs: graph.centrality()
        )

    retrieve.build_runtime_graph(store)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"peak_rss_raw": peak}))
    """
)


def _ru_maxrss_mb(raw: int) -> float:
    if platform.system() == "Darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def run_runtime_graph_rss_arm(
    *,
    proof: RuntimeGraphProof,
    driver: str,
    seed_base: int,
    in_parent: bool,
    root: Path,
    n_records: int,
) -> float:
    env = os.environ.copy()
    env["LILLI_STORAGE_DRIVER"] = driver
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-passphrase-not-secret"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _RSS_ARM_WORKER,
            proof,
            str(seed_base),
            "in_parent" if in_parent else "isolated",
            str(root),
            str(n_records),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, (
        f"RSS arm subprocess failed (rc={proc.returncode}):\n{proc.stderr}"
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return _ru_maxrss_mb(int(payload["peak_rss_raw"]))
