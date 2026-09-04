"""Time-held-out, episodic-tail-gold cue labelling.

Sibling of ``label_real_thread_cues.py``: reuses its mining, auto-suggest,
LOCAL-only fixture write, and repo-path-refusal machinery (never modified
in place), extended with two filters the original lacks:

1. Time-held-out draw -- candidate cue text is mined from natural,
   already-existing episodic ``role:user`` records, excluding anything
   that is itself a tail-gold candidate or a cluster summary, so a cue is
   never engineered around the exact gap it is meant to test.
2. Episodic-tail gold -- the confirmable gold set is restricted to
   records that are members of an already-consolidated cluster, ranked
   6th-or-later by recency within that cluster (the members a cluster's
   own summary text never embeds, since the sleep pipeline only folds the
   first 5 into ``summary_text``). Every member still gets a
   ``consolidated_from`` edge to its summary regardless of rank, so the
   tail set is reconstructed from those edges, not re-derived by
   re-clustering.

Each labelled cue records the gold record id AND that record's cluster's
summary id, for a downstream cluster-stratified bootstrap. NEVER commits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from uuid import UUID

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.recall_accuracy_real import _copy_real_store, _open_copy_store_shared  # noqa: E402
from iai_mcp.embed import embedder_for_store  # noqa: E402
from iai_mcp.sleep import _existing_summary_members  # noqa: E402
from iai_mcp.store import EDGES_TABLE  # noqa: E402

_DEFAULT_FIXTURE_OUT = Path("~/.iai-mcp/eval-fixtures/labelled_holdout_tail_cues.json").expanduser()
_DEFAULT_TARGET_COUNT = 50
_AUTO_SUGGEST_K = 20
_SNIPPET_CHARS = 120

#: A cluster summary embeds only its first-5 members by recency
#: (``sleep.py``'s ``cluster_recs[:5]``) -- the tail begins at the next
#: rank (zero-indexed position 5, recency rank 6).
_TAIL_RANK_FLOOR = 5


def cluster_tail_members(store) -> "dict[UUID, list]":
    """Reconstruct real consolidated clusters from ``consolidated_from``
    edges and return, per summary id, only the tail members (recency rank
    6+) -- the exact records that summary's own text never embedded."""
    edges_df = store.db.open_table(EDGES_TABLE).to_pandas()
    semantic_ids: "set[UUID]" = set()
    for row in store.iter_record_columns(["id", "tier"], batch_size=2048):
        if row.get("tier") == "semantic":
            try:
                semantic_ids.add(UUID(str(row["id"])))
            except (ValueError, TypeError):
                continue
    members_by_summary = _existing_summary_members(edges_df, semantic_ids)

    tails: "dict[UUID, list]" = {}
    for summary_id, member_ids in members_by_summary.items():
        if not member_ids:
            continue
        batch = store.get_batch(list(member_ids))
        cluster_recs = list(batch.values())
        if len(cluster_recs) <= _TAIL_RANK_FLOOR:
            continue  # every member is within the summary's own first-5 -- no tail
        cluster_recs.sort(key=lambda r: r.updated_at or r.created_at, reverse=True)
        tails[summary_id] = cluster_recs[_TAIL_RANK_FLOOR:]
    return tails


def mine_holdout_cues(store, sample_n: int, excluded_ids: "set[UUID]") -> list:
    """Episodic ``role:user`` cues, excluding any record that is itself a
    tail-gold candidate -- a cue text drawn from the same record a
    labeller might pick as gold would engineer the query around the exact
    gap under test rather than holding it out."""
    all_role_user = [
        r for r in store.iter_records(where="tier = 'episodic'")
        if "role:user" in (r.tags or []) and r.id not in excluded_ids
    ]
    if len(all_role_user) <= sample_n:
        return all_role_user
    step = max(1, len(all_role_user) // sample_n)
    return all_role_user[::step][:sample_n]


def _snippet(text: str) -> str:
    return text[:_SNIPPET_CHARS] + ("..." if len(text) > _SNIPPET_CHARS else "")


def _present_and_confirm_tail(
    cue_record, store, embedder, tail_summary_by_id: dict, index: int,
) -> "dict | None":
    print(f"\n{'=' * 72}")
    print(f"[candidate {index}] id={cue_record.id}")
    print(f"cue text: {cue_record.literal_surface}")
    print(f"{'=' * 72}")

    decision = input("Use this cue? [y/n/q]: ").strip()
    if decision.lower() == "q":
        return {"_quit": True}
    if not decision or decision.lower() == "n":
        return None

    cue_vec = embedder.embed(cue_record.literal_surface)
    suggestions = store.exact_top_k(list(cue_vec), k=_AUTO_SUGGEST_K, build_if_cold=True)

    print("\nAuto-suggested candidates (exact_top_k), tail-eligible ones marked [TAIL]:")
    suggestion_records = []
    for i, (rid, cosine) in enumerate(suggestions):
        rec = store.get(rid)
        if rec is None:
            continue
        is_tail = rid in tail_summary_by_id
        suggestion_records.append((i, rid, is_tail))
        marker = " [TAIL]" if is_tail else ""
        print(f"  [{i}] {rid} | cos={cosine:.3f}{marker} | {_snippet(rec.literal_surface)}")

    picked_raw = input(
        'Gold index (a TAIL-marked candidate only; empty=skip; "+ <uuid>" '
        "for a manually-verified tail id): "
    ).strip()

    gold_id: "str | None" = None
    if picked_raw:
        if picked_raw.startswith("+"):
            manual_raw = picked_raw[1:].strip()
            try:
                manual_uuid = UUID(manual_raw) if manual_raw else None
            except ValueError:
                manual_uuid = None
            if manual_uuid is not None and manual_uuid in tail_summary_by_id:
                gold_id = str(manual_uuid)
        else:
            try:
                idx = int(picked_raw)
            except ValueError:
                idx = None
            if idx is not None:
                for i, rid, is_tail in suggestion_records:
                    if i == idx and is_tail:
                        gold_id = str(rid)
                        break

    if gold_id is None:
        print("No tail-eligible gold selected; skipping this cue.")
        return None

    cluster_id = tail_summary_by_id[UUID(gold_id)]
    notes = input("notes (optional): ").strip()

    return {
        "cue_id": f"holdout_cue_{index:04d}",
        "cue_text": cue_record.literal_surface,
        "gold_record_id": gold_id,
        "cluster_id": str(cluster_id),
        "notes": notes,
    }


def _write_fixture(entries: list, out_path: Path, *, force_in_repo: bool) -> None:
    resolved_out = out_path.expanduser().resolve()
    is_in_repo = resolved_out.is_relative_to(_REPO_ROOT)

    if is_in_repo and not force_in_repo:
        print(
            f"REFUSED: --fixture-out ({resolved_out}) resolves inside the repo tree "
            f"({_REPO_ROOT}). The labelled fixture must be LOCAL-ONLY. Pass "
            "--force-in-repo to override (NOT recommended — .gitignore is the "
            "final catch, not the intended safety mechanism).",
            file=sys.stderr,
        )
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"schema_version": 1, "cues": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {len(entries)} labelled tail-gold cues to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive labelling: mine natural episodic cues from a "
            "read-only COPY of your Hippo, restrict gold candidates to "
            "episodic-tail members of an already-consolidated cluster "
            "(recency rank 6+, never embedded by that cluster's own "
            "summary), confirm per cue, write a LOCAL-only JSON fixture "
            "recording the gold id and its cluster id. NEVER commits."
        ),
    )
    parser.add_argument("--fixture-out", type=Path, default=_DEFAULT_FIXTURE_OUT)
    parser.add_argument("--target-count", type=int, default=_DEFAULT_TARGET_COUNT)
    parser.add_argument("--driver", choices=["stdlib", "lilli"], default="stdlib")
    parser.add_argument(
        "--force-in-repo",
        action="store_true",
        help=(
            "Bypass the safety refusal when --fixture-out points inside the repo "
            "(NOT recommended — .gitignore is the final catch)."
        ),
    )
    args = parser.parse_args()
    if args.driver == "lilli":
        os.environ["LILLI_STORAGE_DRIVER"] = "lilli"

    with tempfile.TemporaryDirectory(prefix="iai-mcp-holdout-label-copy-") as td:
        copy_root = _copy_real_store(Path(td) / "copy")
        store = _open_copy_store_shared(copy_root)
        try:
            tails_by_summary = cluster_tail_members(store)
            tail_summary_by_id = {
                member.id: summary_id
                for summary_id, members in tails_by_summary.items()
                for member in members
            }
            embedder = embedder_for_store(store)
            # Oversample generously: the operator can raise --target-count
            # past the original 30-50 floor to reach the verdict's
            # eligible-and-missed power floor -- the real corpus carries
            # many large consolidated clusters, ample headroom for a
            # larger held-out set when the operator needs one.
            candidates = mine_holdout_cues(
                store, sample_n=args.target_count * 3,
                excluded_ids=set(tail_summary_by_id.keys()),
            )
            entries: list = []
            for cand in candidates:
                if len(entries) >= args.target_count:
                    break
                entry = _present_and_confirm_tail(
                    cand, store, embedder, tail_summary_by_id, len(entries) + 1,
                )
                if entry is None:
                    continue
                if entry.get("_quit"):
                    break
                entries.append(entry)
            _write_fixture(entries, args.fixture_out, force_in_repo=args.force_in_repo)
        finally:
            store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
