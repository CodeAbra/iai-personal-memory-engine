"""Multi-cue foresight before/after eval over a labelled fixture, against a
read-only COPY of the operator's own store.

Runs `foresight.refresh_pack` twice per labelled cue (single-cue baseline via
`IAI_MCP_FORESIGHT_MULTI_CUE_OFF=1`, then the multi-cue default) and reports
`pack_hit_rate` before/after plus a negative-control noise check. Mirrors
`bench/foresight_real_eval.py`'s shape; not wired into the pytest gate (bench
scripts are manual/report tooling).

The labelled fixture (`~/.iai-mcp/eval-fixtures/labelled_multicue_drown_cues.json`)
lives OUTSIDE the repo tree and is generated on first run if absent, by mining
real cases from the read-only copy: bare record-id anchors and mining logic
live here (committed); any real cue text or record content is written only
into the local fixture file, never into this source. When mining a lane finds
no suitable real-store case within its bounded search, a synthetic-only
fallback entry is used instead (marked as such, `relevant_record_ids: []`) —
the gap is reported, not papered over with fabricated "real" evidence.

Run: .venv/bin/python bench/foresight_multicue_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from uuid import UUID

_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
_REPO_PATH = str(Path(__file__).resolve().parent.parent)
if _REPO_PATH not in sys.path:
    sys.path.insert(0, _REPO_PATH)

from bench.recall_accuracy_real import (  # noqa: E402
    _copy_real_store,
    _open_copy_store_shared,
    _toggles,
)

_FIXTURE_DEFAULT_PATH = Path(
    "~/.iai-mcp/eval-fixtures/labelled_multicue_drown_cues.json"
).expanduser()

# Bare record-id anchor only — never the literal memory text it points to.
# The real cue text/content is read from the store COPY at run time and
# written only into the local, uncommitted fixture JSON. Override with a
# local-only id via env so no live-store id has to ship in source.
_HUMANIZER_LONG_PROMPT_ID = os.environ.get(
    "IAI_MCP_FORESIGHT_HUMANIZER_ANCHOR_ID",
    "8a6364a8-1a09-4b18-81f6-88a8f79603b3",
)

_LATIN_MINE_TOKEN_RE = re.compile(r"[A-Za-z]{4,}")
_CYRILLIC_MINE_TOKEN_RE = re.compile(r"[а-яёЀ-ӿ]{5,}")
_MINE_SAMPLE_CAP = 600
_MINE_CANDIDATE_CAP = 150
_LATIN_MINE_MIN_IDF = 2.0
# Ground truth for a mined/reproduced case is the ANN neighborhood of the
# EXACT vector refresh_pack's own derived-cue lane would query with — the
# same retrieval mechanism under test, not a proxy. A BM25/anchor-embedding
# proxy was tried first and produced silent false negatives: its neighbor
# set diverged from what the actual ANN-embedding derived-cue query returns.
_GROUND_TRUTH_COS_FLOOR = 0.72

# Synthetic wrapper templates (own crafted prose, no real store content) that
# drown a mined or fallback rare token among unrelated topics, mirroring
# tests/test_foresight_pack.py's DROWN_LONG_PROMPT shape. The Cyrillic
# template is non-English DATA under test, not narrative prose — same
# public-scrub exemption convention as tests/test_foresight_pack.py.
_LATIN_WRAPPER_TEMPLATE = (
    "Quick roundup across a few unrelated threads: the release notes draft "
    "is nearly done, we tried a new pasta recipe last night, the forecast "
    "says rain tomorrow, and by the way don't forget {tok} before the day "
    "wraps up — also the team stand-up moved to Thursday and the plants "
    "still need watering."
)
_CYRILLIC_WRAPPER_TEMPLATE = (
    "Коротко про несколько несвязанных тем: черновик отчёта почти готов, "
    "вечером идём гулять, завтра обещают дождь, и кстати не забудь про "
    "{tok} до конца дня — встречу перенесли на четверг, а цветы всё ещё "
    "нужно полить."
)
_LATIN_SYNTHETIC_FALLBACK_TOKEN = "zzqorvexthon"
_CYRILLIC_SYNTHETIC_FALLBACK_TOKEN = "жмышковатость"


def fixture_path() -> Path:
    env_val = os.environ.get("IAI_MCP_FORESIGHT_MULTICUE_FIXTURE_PATH")
    if env_val:
        return Path(env_val).expanduser()
    return _FIXTURE_DEFAULT_PATH


def _sample_episodic_records(store, cap: int) -> list:
    """Mirrors scripts/label_real_thread_cues.py's `_mine_candidates` shape."""
    records = [
        r for r in store.iter_records(where="tier = 'episodic'")
        if "role:user" in (r.tags or [])
    ]
    if len(records) <= cap:
        return records
    step = max(1, len(records) // cap)
    return records[::step][:cap]


def _derived_cue_vecs(store, embedder, long_prompt: str) -> "list[tuple[str, list[float]]]":
    """Reproduce refresh_pack's own prefilter pool + precision re-rank
    exactly (same constants, same ascending-distance-from-primary-cue sort),
    so mining validates against the SAME derived cues refresh_pack would
    actually pick — not a proxy."""
    from iai_mcp.embed import embed_query
    from iai_mcp.foresight import (
        FORESIGHT_CUE_CAP_DEFAULT,
        FORESIGHT_CUE_PREFILTER_CEILING,
        _cos,
        _derive_short_cues,
    )

    cue_vec = list(embed_query(embedder, long_prompt[:512]))
    cue_cap = FORESIGHT_CUE_CAP_DEFAULT
    prefilter_cap = min(FORESIGHT_CUE_PREFILTER_CEILING, max(cue_cap, cue_cap * 2))
    pool = _derive_short_cues(long_prompt, prefilter_cap, store=store)
    if not pool:
        return []
    pool_vecs = embedder.embed_batch(pool, input_type="query")
    ranked = sorted(zip(pool, pool_vecs), key=lambda cv: _cos(cv[1], cue_vec))
    return ranked[:cue_cap]


def _ground_truth_ids(store, vec: "list[float]", *, k: int = 8) -> "list[str]":
    try:
        hits = store.exact_top_k(list(vec), k=k, build_if_cold=True)
    except Exception:  # noqa: BLE001 -- mining is best-effort, never blocking
        return []
    return [str(rid) for rid, cos in hits if cos >= _GROUND_TRUTH_COS_FLOOR]


def _mine_humanizer_case(store, embedder) -> "dict | None":
    from iai_mcp.foresight import _record_session

    rec = store.get(UUID(_HUMANIZER_LONG_PROMPT_ID))
    if rec is None:
        return None
    long_prompt = rec.literal_surface
    if not long_prompt:
        return None
    session_id = _record_session(rec)

    derived = _derived_cue_vecs(store, embedder, long_prompt)
    if not derived:
        return None
    chosen_tok, chosen_vec = derived[0]
    relevant_ids = _ground_truth_ids(store, chosen_vec)
    if not relevant_ids:
        return None

    return {
        "cue_id": "humanizer_real_8a6364a8",
        "case_type": "humanizer",
        "long_prompt": long_prompt,
        "short_cue": chosen_tok,
        "relevant_record_ids": relevant_ids,
        "session_id": session_id,
        "notes": (
            "reproduced real drown case: a long multi-topic real prompt "
            "whose relevant entity the whole-prompt cue misses; "
            "relevant_record_ids = the ANN neighborhood of the derived "
            "cue refresh_pack's own precision re-rank actually selects"
        ),
    }


def _mine_rare_token_case(store, embedder, *, kind: str) -> "dict | None":
    """Mine a real rare token whose synthetic-wrapper prompt reproduces the
    drown condition end-to-end: the token must (a) exist in the real store,
    (b) survive refresh_pack's own derivation + precision re-rank as the #1
    (most-distant) derived cue when embedded in the wrapper — the only slot
    guaranteed to be tried under the default single-slot reserve — and
    (c) have a non-empty real-store ANN neighborhood as ground truth."""
    from iai_mcp.entity_anchors import _CAP_DENYLIST
    from iai_mcp.foresight import _record_session

    token_re = _LATIN_MINE_TOKEN_RE if kind == "latin" else _CYRILLIC_MINE_TOKEN_RE
    wrapper = _LATIN_WRAPPER_TEMPLATE if kind == "latin" else _CYRILLIC_WRAPPER_TEMPLATE

    sample = _sample_episodic_records(store, _MINE_SAMPLE_CAP)
    seen: set = set()
    tried = 0
    for rec in sample:
        text = rec.literal_surface or ""
        for m in token_re.finditer(text):
            tok = m.group(0).lower()
            if tok in seen or tok in _CAP_DENYLIST:
                continue
            seen.add(tok)
            tried += 1
            if tried > _MINE_CANDIDATE_CAP:
                return None

            if kind == "latin":
                # Cheap pre-check: _derive_short_cues' own Latin lane gate —
                # a token that fails this could never enter the pool at all.
                try:
                    hits = store.lexical_query_warm(tok, k=1, min_idf=_LATIN_MINE_MIN_IDF)
                except Exception:  # noqa: BLE001
                    continue
                if not hits:
                    continue

            wrapped_prompt = wrapper.format(tok=tok)
            derived = _derived_cue_vecs(store, embedder, wrapped_prompt)
            if not derived or derived[0][0] != tok:
                continue
            _, chosen_vec = derived[0]
            relevant_ids = _ground_truth_ids(store, chosen_vec)
            if not relevant_ids:
                continue
            top_rec = store.get(UUID(relevant_ids[0]))
            top_session = _record_session(top_rec) if top_rec is not None else None

            return {
                "cue_id": f"mined_{kind}_{tok[:12]}",
                "case_type": kind,
                "long_prompt": wrapped_prompt,
                "short_cue": tok,
                "relevant_record_ids": relevant_ids,
                "session_id": top_session or f"multicue-eval-mined-{kind}",
                "notes": (
                    f"real rare {kind} token mined from the store copy, "
                    "drowned in a synthetic wrapper prompt; confirmed as the "
                    "#1 derived cue under the default reserve"
                ),
            }
    return None


def _synthetic_fallback_case(kind: str) -> dict:
    tok = (
        _LATIN_SYNTHETIC_FALLBACK_TOKEN if kind == "latin"
        else _CYRILLIC_SYNTHETIC_FALLBACK_TOKEN
    )
    wrapper = _LATIN_WRAPPER_TEMPLATE if kind == "latin" else _CYRILLIC_WRAPPER_TEMPLATE
    return {
        "cue_id": f"{kind}_synthetic_fallback",
        "case_type": f"{kind}_synthetic_fallback",
        "long_prompt": wrapper.format(tok=tok),
        "short_cue": tok,
        "relevant_record_ids": [],
        "session_id": f"multicue-eval-fallback-{kind}",
        "notes": (
            f"mining found no real-store {kind} candidate within the bounded "
            "search; entirely synthetic, no real content, informational only"
        ),
    }


def _negative_control_case() -> dict:
    return {
        "cue_id": "negative_control_no_drowned_rule",
        "case_type": "negative_control",
        "long_prompt": (
            "Notes for later: refill the printer paper, check the recycling "
            "schedule, the bus route changed on Mondays, remember to charge "
            "the flashlight batteries, and the neighborhood picnic is next "
            "weekend."
        ),
        "short_cue": "",
        "relevant_record_ids": [],
        "session_id": "multicue-eval-negative-control",
        "notes": "no drowned rule present — checks multi-cue injects no noise",
    }


def _generate_fixture(store, embedder) -> dict:
    cues: list = []
    humanizer = _mine_humanizer_case(store, embedder)
    if humanizer:
        cues.append(humanizer)
    else:
        print(
            "[foresight_multicue_eval] humanizer anchor record not found in "
            "this store copy — skipping that case", flush=True,
        )

    for kind in ("latin", "cyrillic"):
        mined = _mine_rare_token_case(store, embedder, kind=kind)
        if mined is None:
            print(
                f"[foresight_multicue_eval] no real-store {kind} candidate found "
                f"within {_MINE_CANDIDATE_CAP} tried tokens — using synthetic fallback",
                flush=True,
            )
            mined = _synthetic_fallback_case(kind)
        cues.append(mined)

    cues.append(_negative_control_case())
    return {"schema_version": 1, "cues": cues}


def _reset_session_pack_state(store, session_id: str) -> None:
    from iai_mcp.foresight import _state_path, pack_path

    pack_path(store, session_id).unlink(missing_ok=True)
    _state_path(store, session_id).unlink(missing_ok=True)


def run_eval(copy_root: "Path | None" = None, *, regenerate: bool = False) -> dict:
    from iai_mcp import foresight
    from iai_mcp.embed import embed_query, embedder_for_store

    owns_copy = copy_root is None
    if owns_copy:
        copy_root = Path(tempfile.mkdtemp(prefix="foresight-multicue-eval-"))
        _copy_real_store(copy_root)
    store = _open_copy_store_shared(copy_root)
    try:
        try:
            active_count = store.active_records_count()
            print(
                f"[foresight_multicue_eval] copy active records: {active_count}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 -- diagnostic only
            print(f"[foresight_multicue_eval] active-count probe failed: {exc}", flush=True)

        embedder = embedder_for_store(store)
        try:
            # One-time, offline-only build of the warm lexical index (never
            # done on the recall/foresight hot path) so the Latin mining
            # lane's `lexical_query_warm` has a generation-current index.
            store.lexical_search("_multicue_eval_warmup_", k=1)
        except Exception as exc:  # noqa: BLE001 -- warm-up is best-effort
            print(f"[foresight_multicue_eval] lexical warm-up skipped: {exc}", flush=True)

        fpath = fixture_path()
        if regenerate or not fpath.exists():
            fixture = _generate_fixture(store, embedder)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(
                json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            print(
                f"[foresight_multicue_eval] wrote {len(fixture['cues'])} cues to {fpath}",
                flush=True,
            )
        else:
            fixture = json.loads(fpath.read_text(encoding="utf-8"))

        cues = fixture["cues"]
        per_cue: list = []
        for cue in cues:
            long_prompt = cue["long_prompt"]
            relevant = set(str(r) for r in (cue.get("relevant_record_ids") or []))
            session_id = cue.get("session_id") or f"multicue-eval-{cue['cue_id']}"
            emb = list(embed_query(embedder, long_prompt[:512]))

            _reset_session_pack_state(store, session_id)
            with _toggles(IAI_MCP_FORESIGHT_MULTI_CUE_OFF="1"):
                single = foresight.refresh_pack(
                    store, cue_text=long_prompt, cue_embedding=emb, session_id=session_id,
                )
            single_ids = set(single.get("packed_ids") or [])

            _reset_session_pack_state(store, session_id)
            multi = foresight.refresh_pack(
                store, cue_text=long_prompt, cue_embedding=emb, session_id=session_id,
            )
            multi_ids = set(multi.get("packed_ids") or [])

            entry = {
                "cue_id": cue["cue_id"],
                "case_type": cue.get("case_type"),
                "single_packed": len(single_ids),
                "multi_packed": len(multi_ids),
                "multi_skipped_already_injected": multi.get("skipped_already_injected"),
            }
            if relevant:
                entry["single_hit"] = bool(single_ids & relevant)
                entry["multi_hit"] = bool(multi_ids & relevant)
            else:
                entry["single_hit"] = None
                entry["multi_hit"] = None
                entry["noise_injected"] = sorted(multi_ids - single_ids)
            per_cue.append(entry)

        scored = [c for c in per_cue if c["single_hit"] is not None]
        n_scored = len(scored)
        pack_hit_rate_before = (
            round(sum(1 for c in scored if c["single_hit"]) / n_scored, 3) if n_scored else None
        )
        pack_hit_rate_after = (
            round(sum(1 for c in scored if c["multi_hit"]) / n_scored, 3) if n_scored else None
        )
        neg_controls = [c for c in per_cue if c["single_hit"] is None]
        negative_control_clean = (
            all(not c.get("noise_injected") for c in neg_controls) if neg_controls else None
        )

        return {
            "cue_count": len(cues),
            "scored_cases": n_scored,
            "pack_hit_rate_before": pack_hit_rate_before,
            "pack_hit_rate_after": pack_hit_rate_after,
            "negative_control_clean": negative_control_clean,
            "per_cue": per_cue,
        }
    finally:
        store.close()
        if owns_copy:
            shutil.rmtree(copy_root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Measure multi-cue foresight pack_hit_rate before/after against a "
            "read-only copy of the operator's own Hippo store."
        ),
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="rebuild the labelled fixture even if one already exists",
    )
    args = parser.parse_args()
    result = run_eval(regenerate=args.regenerate)
    print(json.dumps(result, indent=2, default=str))
