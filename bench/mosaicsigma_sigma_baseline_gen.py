#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import networkx as nx  # noqa: E402

from iai_mcp.sigma import classify_regime, fast_sigma  # noqa: E402


FROZEN_GENERATED_AT: str = "2026-05-27T00:00:00+00:00"

FROZEN_NETWORKX_VERSION: str = "3.6.1"

SEED: int = 42

N_RANDOM: int = 3


def _canonical_edges(g: nx.Graph) -> list[list[int]]:
    edges = [tuple(sorted((int(u), int(v)))) for (u, v) in g.edges()]
    edges.sort()
    return [list(e) for e in edges]


def _relabel_consecutive(g: nx.Graph) -> nx.Graph:
    return nx.convert_node_labels_to_integers(g, first_label=0, ordering="sorted")


def _sigma_record(
    g: nx.Graph,
    *,
    source: str,
    literature_anchor: str | None,
    status: str,
    include_edges: bool = True,
) -> dict:
    g = _relabel_consecutive(g)
    n = int(g.number_of_nodes())
    m = int(g.number_of_edges())
    sigma_val, C, L, Cr, Lr = fast_sigma(g, n_random=N_RANDOM, seed=SEED)
    if isinstance(sigma_val, float) and (sigma_val != sigma_val):
        sigma_for_json: float | None = None
    else:
        sigma_for_json = float(sigma_val)
    regime = classify_regime(n, sigma_for_json)
    record: dict = {
        "n": n,
        "m": m,
        "C": float(C),
        "L": float(L),
        "sigma": sigma_for_json,
        "Cr": float(Cr),
        "Lr": float(Lr),
        "regime": regime,
        "source": source,
        "literature_anchor": literature_anchor,
        "status": status,
        "edges": _canonical_edges(g) if include_edges else None,
    }
    return record


def build_fixtures() -> dict[str, dict]:
    fixtures: dict[str, dict] = {}


    fixtures["tiny_10_ws_k4"] = _sigma_record(
        nx.watts_strogatz_graph(10, 4, 0.0, seed=SEED),
        source="hand-verified",
        literature_anchor="Watts-Strogatz 1998 Nature 393:440-442 Table 1",
        status="mandatory",
    )

    fixtures["tiny_20_ws_p010"] = _sigma_record(
        nx.watts_strogatz_graph(20, 4, 0.1, seed=SEED),
        source="hand-verified",
        literature_anchor="Watts-Strogatz 1998 Nature 393:440-442 Table 1",
        status="informational-boundary",
    )


    fixtures["karate"] = _sigma_record(
        nx.karate_club_graph(),
        source="networkx-frozen",
        literature_anchor="Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1",
        status="mandatory",
    )

    fixtures["les_miserables"] = _sigma_record(
        nx.les_miserables_graph(),
        source="networkx-frozen",
        literature_anchor="Humphries-Gurney 2008 PLOS ONE 3(4):e0002051 Table 1",
        status="mandatory",
    )


    for n_nodes, m_edges, key in [
        (200, 400, "er_200"),
        (500, 1000, "er_500"),
        (1000, 2000, "er_1000"),
    ]:
        fixtures[key] = _sigma_record(
            nx.gnm_random_graph(n_nodes, m_edges, seed=SEED),
            source="networkx-frozen",
            literature_anchor=None,
            status="mandatory",
        )


    fixtures["ws_2500_k4_p0"] = _sigma_record(
        nx.watts_strogatz_graph(2500, 4, 0.0, seed=SEED),
        source="networkx-frozen",
        literature_anchor="Watts-Strogatz 1998 Nature 393:440-442 Table 1",
        status="mandatory",
    )


    snapshot_path = "bench/snapshots/live_n2000.npz"
    if Path(snapshot_path).exists():
        fixtures["live_n2000"] = {
            "n": -1,
            "m": -1,
            "C": 0.0,
            "L": 0.0,
            "sigma": None,
            "Cr": 0.0,
            "Lr": 0.0,
            "regime": "unavailable",
            "source": "snapshot",
            "literature_anchor": None,
            "status": "optional",
            "edges": None,
            "snapshot_path": snapshot_path,
            "note": "live_n2000 snapshot present at snapshot_path; "
                    "downstream tests load and re-compute fields.",
        }
    else:
        fixtures["live_n2000"] = {
            "n": 0,
            "m": 0,
            "C": 0.0,
            "L": 0.0,
            "sigma": None,
            "Cr": 0.0,
            "Lr": 0.0,
            "regime": "unavailable",
            "source": "missing-snapshot",
            "literature_anchor": None,
            "status": "optional",
            "edges": None,
            "snapshot_path": None,
            "note": "live_n2000 unavailable; the 8 mandatory fixtures "
                    "remain the gated set.",
        }

    return fixtures


def build_doc() -> dict:
    fixtures = build_fixtures()
    return {
        "schema_version": "1",
        "generator_meta": {
            "networkx_version": FROZEN_NETWORKX_VERSION,
            "seed": SEED,
            "n_random": N_RANDOM,
            "generated_at": FROZEN_GENERATED_AT,
            "host_python": "frozen",
            "consilium_reference": "4/4 unanimous (multi-channel) — 3-round multi-Socratic",
        },
        "fixtures": fixtures,
    }


def canonical_bytes(doc_without_hash: dict) -> bytes:
    return json.dumps(doc_without_hash, sort_keys=True, indent=2).encode()


def main() -> int:
    doc = build_doc()
    digest = hashlib.sha256(canonical_bytes(doc)).hexdigest()
    doc["sha256_self_check"] = digest

    if len(sys.argv) > 1 and sys.argv[1] not in ("-", "--stdout"):
        target = Path(sys.argv[1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {target}", file=sys.stderr)
        print(f"sha256_self_check = {digest}", file=sys.stderr)
    else:
        sys.stdout.write(json.dumps(doc, sort_keys=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
