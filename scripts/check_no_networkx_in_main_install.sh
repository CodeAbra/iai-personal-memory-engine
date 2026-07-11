#!/usr/bin/env bash
#
# Verify the iai_mcp main install (no [dev] extras) does NOT pull
# networkx transitively AND DOES ship the iai_mcp_native Rust
# extension required by the σ assembly. Run from a clean venv before
# committing any change to the dependency tree.
#
# Usage:
#   python -m venv /tmp/iai-mcp-clean-check && \
#     /tmp/iai-mcp-clean-check/bin/pip install -e . && \
#     /tmp/iai-mcp-clean-check/bin/python scripts/check_no_networkx_in_main_install.sh
#
# Or, on an existing venv, uninstall networkx + hypothesis-networkx
# first to simulate the no-[dev] state.
#
# Exit codes:
#   0  — main install is networkx-free AND iai_mcp_native importable AND
#        MemoryGraph public API stays networkx-free at runtime.
#   1  — iai_mcp import failed.
#   2  — networkx loaded after `import iai_mcp` (transitive regression).
#   3  — iai_mcp_native missing or broken.
#   4  — iai_mcp_native.graph.__file__ resolves to an empty path, or
#        the algorithm surface is incomplete.
#   5  — MemoryGraph runtime invariant failed (either an API call regressed
#        or a lazy `import networkx` was reintroduced into the graph
#        backend's hot path).
set -euo pipefail

if ! python -c "import iai_mcp" 2>/dev/null; then
    echo "ERROR: iai_mcp import failed; cannot verify the dependency tree"
    exit 1
fi

NETWORKX_LOADED=$(python -c "import iai_mcp; import sys; print('YES' if 'networkx' in sys.modules else 'NO')")
if [[ "$NETWORKX_LOADED" == "YES" ]]; then
    echo "ERROR: networkx loaded after 'import iai_mcp' — the main dependency"
    echo "tree has regressed. networkx must remain in [dev] extras only."
    exit 2
fi
echo "OK: 'import iai_mcp' does not pull networkx"

# Clean install MUST ship iai_mcp_native (Rust extension).
# The σ assembly imports `from iai_mcp_native import graph as lilli_graph`;
# without the wheel, any σ call would raise ImportError.
if ! python -c "import iai_mcp_native; import iai_mcp_native.graph" 2>/dev/null; then
    echo "ERROR: iai_mcp_native (or iai_mcp_native.graph) not importable —"
    echo "Rust extension missing from the install. Build via:"
    echo "  cd rust/iai_mcp_native && maturin develop --release"
    exit 3
fi
# iai_mcp_native.graph is a registered PyO3 sub-module (no own __file__);
# probe it via the parent extension's __file__ and a smoke-call into the
# Rust kernel.
NATIVE_SO_PATH=$(python -c "import iai_mcp_native; print(iai_mcp_native.__file__)")
if [[ -z "$NATIVE_SO_PATH" ]]; then
    echo "ERROR: iai_mcp_native.__file__ resolves to an empty path —"
    echo "extension is in a broken intermediate state. Re-run maturin develop."
    exit 4
fi
if ! python -c "from iai_mcp_native import graph as g; assert callable(g.is_connected); assert callable(g.average_clustering); assert callable(g.gnm_random_graph)" 2>/dev/null; then
    echo "ERROR: iai_mcp_native.graph algorithm surface incomplete —"
    echo "expected is_connected / average_clustering / gnm_random_graph callable."
    exit 4
fi
echo "OK: iai_mcp_native installed at $NATIVE_SO_PATH"

echo
echo "=== MemoryGraph runtime invariant check ==="
# Exercises the full MemoryGraph public API in a single python -c block and
# asserts at the end that `networkx` was never pulled into `sys.modules`.
# Complements the pip-level check above: that one verifies networkx is not
# in the install tree; this one verifies it stays out of the runtime even
# when the in-process graph backend is fully exercised.
if ! python -c '
import sys
from uuid import uuid4
from iai_mcp.graph import MemoryGraph
g = MemoryGraph()
a, b, c = uuid4(), uuid4(), uuid4()
for n in (a, b, c):
    g.add_node(n, community_id=None, embedding=[0.0] * 384)
g.add_edge(a, b)
g.add_edge(b, c)
assert g.node_count() == 3
assert g.has_node(a)
indptr, indices, data = g.to_csr_arrays()
assert len(indptr) == 4
cen = g.centrality()
assert b in cen
list(g.iter_edges_with_weight())
list(g.iter_nodes())
list(g.degrees())
g.rich_club_coefficient()
g.two_hop_neighborhood([a], top_k=5)
assert "networkx" not in sys.modules, "networkx in sys.modules after MemoryGraph full-API exercise"
print("OK: MemoryGraph public API exercised without networkx")
' 2>&1; then
    echo "ERROR: MemoryGraph runtime invariant check failed —"
    echo "either an API call regressed or a lazy ``import networkx`` was reintroduced."
    exit 5
fi

echo
echo "Main install is networkx-free, ships the Rust extension, and MemoryGraph"
echo "runtime stays networkx-free across its full public API."
