"""Every served hit's reason must reconcile with its served score.

Property test at the RecallResponse boundary — the surface users read —
over randomized corpora. The oracle parses the reason grammar the scorer
emits and recomputes the score; any future mutator that changes a served
score without marking itself in the reason turns this red without anyone
remembering to update a list.

Printed values are rounded (3 decimals per term) and multipliers amplify
that rounding; tolerance is 2% of |score| with a 2e-3 absolute floor
(3-decimal printing leaves up to ~5e-4 per active term) — wholesale
replacements and forgotten multipliers shift scores far beyond it.
"""
from __future__ import annotations

import re

import numpy as np

from iai_mcp.retrieve import STALE_DOWNWEIGHT_FACTOR

from tests.test_recall_for_response import (
    _FakeEmbedder,
    _build_store_and_graph,
    _flat_assignment,
)

# Reasons that make no arithmetic claim: fixed labels from non-ranked lanes.
_FIXED_REASONS = (
    "pending-recency",
    "contradicts-edge neighbour",
    "L0 identity (always skipped)",
    "low-similarity baseline anti-hit",
)
# Markers that declare a wholesale score replacement — arithmetic is
# intentionally not reconcilable past them, but they MUST be present.
_REPLACEMENT_MARKERS = (
    " | anchored-below-corrector",
    " | capped-below-superseder",
)

_BASE_RE = re.compile(
    r"^cos (?P<cos>-?\d+\.\d+)\*(?P<wc>[\d.e+-]+)"
    r" \+ aaak (?P<aaak>-?\d+\.\d+)\*(?P<wa>[\d.e+-]+)"
    r" \+ deg_norm (?P<deg>-?\d+\.\d+)\*(?P<wd>[\d.e+-]+)"
    r" - age (?P<age>-?\d+\.\d+)\*(?P<wg>[\d.e+-]+)"
)
_TERM_RES = {
    "spread": re.compile(r" \+ spread (-?\d+\.\d+)"),
    "community": re.compile(r" \+ community (-?\d+\.\d+)"),
    "structural": re.compile(r" \| structural (-?\d+\.\d+) \(w=(-?\d+\.\d+)\)"),
    "xgain": re.compile(r" \| xgain (-?\d+\.\d+)"),
    "stab": re.compile(r" \+ stab (-?\d+\.\d+)"),
    "xval": re.compile(r" \| xval (-?\d+\.\d+)"),
    "trigram": re.compile(r" \| x2\.0 trigram"),
    "fts": re.compile(r" \| x3\.0 fts"),
    "lex": re.compile(r" \+ lex (-?\d+\.\d+)"),
    "xtier": re.compile(r" \| xtier ([\d.eE+-]+)"),
    "xtemp": re.compile(r" \| xtemp ([\d.eE+-]+)"),
}


def reconcile(reason: str, score: float) -> tuple[bool, bool, str]:
    """Return (claim_checked, ok, detail). claim_checked=False means the
    reason declares no arithmetic claim (fixed lane or replacement)."""
    if reason in _FIXED_REASONS:
        return False, True, "fixed-lane reason"
    if any(m in reason for m in _REPLACEMENT_MARKERS):
        return False, True, "wholesale replacement declared"
    if reason.startswith("exact-cosine"):
        sep = " | score from pipeline rank: "
        if sep not in reason:
            return False, True, "bare authority reason"
        reason = reason.split(sep, 1)[1]

    stale_count = reason.count(" · stale")
    reason = reason.replace(" · stale", "")

    m = _BASE_RE.match(reason)
    assert m, f"unparseable reason: {reason!r}"
    val = (
        float(m["cos"]) * float(m["wc"])
        + float(m["aaak"]) * float(m["wa"])
        + float(m["deg"]) * float(m["wd"])
        - float(m["age"]) * float(m["wg"])
    )
    if s := _TERM_RES["spread"].search(reason):
        val += float(s.group(1))
    if s := _TERM_RES["community"].search(reason):
        val += float(s.group(1))
    if s := _TERM_RES["structural"].search(reason):
        sw = float(s.group(2))
        val = (1.0 - sw) * val + sw * float(s.group(1))
    if s := _TERM_RES["xgain"].search(reason):
        val *= float(s.group(1))
    if s := _TERM_RES["stab"].search(reason):
        val += float(s.group(1))
    if s := _TERM_RES["xval"].search(reason):
        val *= float(s.group(1))
    if _TERM_RES["trigram"].search(reason):
        val *= 2.0
    if _TERM_RES["fts"].search(reason):
        val *= 3.0
    if s := _TERM_RES["lex"].search(reason):
        val += float(s.group(1))
    if s := _TERM_RES["xtier"].search(reason):
        val *= float(s.group(1))
    if s := _TERM_RES["xtemp"].search(reason):
        val *= float(s.group(1))
    val *= STALE_DOWNWEIGHT_FACTOR ** stale_count

    tol = max(2e-3, 0.02 * abs(score))
    ok = abs(val - score) <= tol
    return True, ok, (
        f"reconstructed {val:.4f} vs served {score:.4f} (tol {tol:.4f}) "
        f"from {reason!r}"
    )


def test_reason_grammar_oracle_full_grammar() -> None:
    # A synthetic reason exercising EVERY grammar term, in the exact order
    # the scorer applies them; the oracle must reconstruct it to the digit.
    base = 1.0 * 0.9 + 0.3 * 0.5 + 0.05 * 0.8 - 0.05 * 0.4 + 0.041 + 0.052
    lerped = 0.7 * base + 0.3 * 0.45
    full = ((lerped * 0.973 + 0.037) * 1.2) * 2.0 * 3.0 + 0.081
    full = full * 1.05 * 1.15 * 0.5
    reason = (
        "cos 0.900*1 + aaak 0.50*0.3 + deg_norm 0.800*0.05 - age 0.40*0.05"
        " + spread 0.041 + community 0.052 | structural 0.450 (w=0.30)"
        " | xgain 0.973 + stab 0.037 | xval 1.20 | x2.0 trigram | x3.0 fts"
        " + lex 0.081 | xtier 1.05 | xtemp 1.15 · stale"
    )
    checked, ok, detail = reconcile(reason, full)
    assert checked and ok, detail


def test_reason_grammar_oracle_self_check() -> None:
    # A live-shaped served string — must reconcile.
    checked, ok, detail = reconcile(
        "cos 0.888*1 + aaak 0.00*0.3 + deg_norm 0.486*0.0289 "
        "- age 0.00*0.05 + stab 0.050 | xtier 1.05",
        0.999,
    )
    assert checked and ok, detail
    # A wrong score must NOT reconcile — the oracle has teeth.
    checked, ok, _ = reconcile(
        "cos 0.888*1 + aaak 0.00*0.3 + deg_norm 0.486*0.0289 "
        "- age 0.00*0.05 + stab 0.050 | xtier 1.05",
        1.90,
    )
    assert checked and not ok
    checked, _, _ = reconcile("pending-recency", 0.0)
    assert not checked
    checked, _, _ = reconcile(
        "cos 1.000*1 + aaak 0.00*0.3 + deg_norm 1.000*0.1 - age 0.00*0.05 "
        "| capped-below-superseder",
        0.42,
    )
    assert not checked


def test_every_served_reason_reconciles_over_random_corpora(tmp_path) -> None:
    from iai_mcp.pipeline import recall_for_response

    failures: list[str] = []
    claims_checked = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        store, graph, recs = _build_store_and_graph(
            tmp_path / f"seed{seed}", n=18, surface_len=40,
        )
        cue_vec = rng.normal(size=len(recs[0].embedding))
        cue_vec = (cue_vec / np.linalg.norm(cue_vec)).tolist()
        for rec in recs:
            v = rng.normal(size=len(rec.embedding))
            rec.embedding = (v / np.linalg.norm(v)).tolist()
            rec.stability = float(rng.uniform(0.0, 1.0))
            if rng.uniform() < 0.3:
                rec.tags = ["doc:notes.md"]
            store.update(rec)
            graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        for _ in range(8):
            a, b = rng.choice(len(recs), size=2, replace=False)
            store.boost_edges(
                [(recs[int(a)].id, recs[int(b)].id)],
                edge_type="hebbian", delta=1.0,
            )
            graph.add_edge(
                recs[int(a)].id, recs[int(b)].id,
                edge_type="hebbian", weight=1.0,
            )
        for _ in range(3):
            a, b = rng.choice(len(recs), size=2, replace=False)
            store.boost_edges(
                [(recs[int(a)].id, recs[int(b)].id)],
                edge_type="contradicts", delta=1.0,
            )

        resp = recall_for_response(
            store=store, graph=graph, assignment=_flat_assignment(recs),
            rich_club=[], embedder=_FakeEmbedder(vec=cue_vec),
            cue=f"random probe {seed} about notes.md",
            session_id=f"s-recon-{seed}",
        )
        for hit in resp.hits:
            checked, ok, detail = reconcile(hit.reason, hit.score)
            if checked:
                claims_checked += 1
                if not ok:
                    failures.append(f"seed {seed}: {detail}")

    assert claims_checked >= 20, (
        f"the corpora must exercise real ranked hits; only {claims_checked} "
        "arithmetic claims were checked"
    )
    assert not failures, (
        "served reasons that do not reconcile with served scores:\n  "
        + "\n  ".join(failures[:10])
    )
