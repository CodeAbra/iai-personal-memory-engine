"""Attribution probe: split numba JIT MAP_JIT region growth into
import-time vs first-call-time on a fixture above the SMALL_N_FLAT
short-circuit (so the mosaic path is actually exercised).

Run as a standalone script (NOT a pytest test):
    python tests/rss_rca/probes/numba_attribution.py

Exit codes:
    0 — PROCEED — runtime per-call leak confirmed; cold-caller isolation
                  will reduce VM_ALLOCATE residence in the parent daemon.
    2 — PIVOT  — leak is import-time preallocation; per-caller isolation
                  cannot fix this surface (the parent loads mosaic via
                  transitive import); re-plan required.
    3 — PIVOT  — attribution unclear; surface new evidence to orchestrator.

Mechanism:
    Spawns a fresh Python child via subprocess.run so the measurement is
    uncontaminated by the parent's already-imported modules. The child
    snapshots vmmap region counts at three checkpoints:
      A) before any project import,
      B) after importing iai_mcp.community (which transitively loads
         mosaic + the 6 @njit kernels at module-load time),
      C) after one detect_communities call on a 300-node fixture above
         SMALL_N_FLAT=200 (forces the mosaic branch, not the flat
         short-circuit).

    The probe parses the child's JSON line and applies a decision tree.

Evidence file:
    tests/rss_rca/probes/numba_attribution-result.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


CHILD_SCRIPT = r'''
import json, os, subprocess, sys
from uuid import uuid4

def vmmap_va_count(pid):
    out = subprocess.run(
        ["/usr/bin/vmmap", str(pid)],
        capture_output=True, text=True, timeout=20,
    )
    return sum(1 for line in out.stdout.splitlines() if line.startswith("VM_ALLOCATE"))

pid = os.getpid()

# Checkpoint A: before any project import.
regions_pre_import = vmmap_va_count(pid)

# Checkpoint B: import iai_mcp.community (transitively loads mosaic + numba).
from iai_mcp.community import detect_communities, SMALL_N_FLAT
from iai_mcp.graph import MemoryGraph

regions_post_import = vmmap_va_count(pid)

# Build a deterministic 300-node fixture with ~900 edges (3x edges:nodes).
# Node count MUST be > SMALL_N_FLAT (200) so detect_communities enters
# the mosaic branch, not the flat short-circuit.
import random
random.seed(42)
n_nodes = max(300, SMALL_N_FLAT + 100)

# Use a deterministic float32 embedding per node (cosine doesn't matter
# for this probe — we just need a valid graph that exercises mosaic).
node_ids = [uuid4() for _ in range(n_nodes)]
embedding_dim = 16  # tiny — embeddings are not the leak target
nodes = []
for nid in node_ids:
    emb = [random.random() for _ in range(embedding_dim)]
    nodes.append((nid, None, emb, {}))

# ~900 edges = 3 per node on average; random pairs with weight=1.0.
edges = []
n_edges_target = n_nodes * 3
for _ in range(n_edges_target):
    src = random.choice(node_ids)
    dst = random.choice(node_ids)
    if src == dst:
        continue
    edges.append((src, dst, 1.0, "hebbian"))

graph = MemoryGraph()
graph.clear_and_rebuild(nodes, edges)

# Checkpoint C: one detect_communities call (cold prior).
assignment = detect_communities(graph, prior=None, prior_mode="cold")
regions_post_call = vmmap_va_count(pid)

# Capture community labels (sorted by node UUID bytes -> tuple of int label
# indices, canonical relabel done in tests).
n_to_c = assignment.node_to_community
comms_sorted = sorted(set(n_to_c.values()), key=lambda u: u.bytes)
comm_index = {cu: i for i, cu in enumerate(comms_sorted)}
labels_sorted_by_node = [
    comm_index[n_to_c[nid]] for nid in sorted(n_to_c.keys(), key=lambda u: u.bytes)
]

result = {
    "regions_pre_import": regions_pre_import,
    "regions_post_import": regions_post_import,
    "regions_post_call": regions_post_call,
    "import_delta": regions_post_import - regions_pre_import,
    "call_delta": regions_post_call - regions_post_import,
    "fixture_n_nodes": n_nodes,
    "small_n_flat": SMALL_N_FLAT,
    "n_communities": len(comms_sorted),
    "labels_sorted_by_node": labels_sorted_by_node,
    "backend": str(assignment.backend),
}
print("PROBE_RESULT:" + json.dumps(result))
'''


def main() -> int:
    here = Path(__file__).resolve().parent
    result_path = here / "numba_attribution-result.json"

    print(f"[probe] spawning fresh Python child to measure attribution...")
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        capture_output=True, text=True, timeout=600,
    )

    if proc.returncode != 0:
        print(f"[probe] child exited non-zero: {proc.returncode}", file=sys.stderr)
        print(f"[probe] stderr tail:\n{proc.stderr[-2000:]}", file=sys.stderr)
        return 4

    result_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE_RESULT:"):
            result_line = line[len("PROBE_RESULT:"):]
            break
    if result_line is None:
        print("[probe] child did not emit PROBE_RESULT line", file=sys.stderr)
        print(f"[probe] stdout tail:\n{proc.stdout[-2000:]}", file=sys.stderr)
        return 4

    data = json.loads(result_line)
    result_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    import_delta = data["import_delta"]
    call_delta = data["call_delta"]
    n_nodes = data["fixture_n_nodes"]
    small_n_flat = data["small_n_flat"]

    print(f"[probe] fixture_n_nodes={n_nodes} (SMALL_N_FLAT={small_n_flat})")
    print(f"[probe] regions_pre_import  = {data['regions_pre_import']}")
    print(f"[probe] regions_post_import = {data['regions_post_import']}  "
          f"(import_delta={import_delta})")
    print(f"[probe] regions_post_call   = {data['regions_post_call']}  "
          f"(call_delta={call_delta})")
    print(f"[probe] backend={data['backend']}  "
          f"n_communities={data['n_communities']}")
    print(f"[probe] evidence file: {result_path}")

    if n_nodes < 300 or n_nodes <= small_n_flat:
        print("DECISION: PIVOT — fixture too small to exercise mosaic path",
              file=sys.stderr)
        return 3

    if import_delta > 300:
        print("DECISION: PIVOT — leak is import-time preallocation, "
              "per-caller isolation will not fix it")
        return 2
    if import_delta < 50 and call_delta > 100:
        print("DECISION: PROCEED — runtime per-call leak confirmed")
        return 0
    print("DECISION: PIVOT — attribution unclear, surface new evidence to "
          "orchestrator")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
