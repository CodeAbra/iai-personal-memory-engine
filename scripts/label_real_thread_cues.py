"""Interactive labelling: mine cues from a read-only COPY of the owner's Hippo,
auto-suggest relevant memories via exact_top_k, confirm per cue, write a
LOCAL-only JSON fixture. NEVER commits.

The default output path is outside the repository tree
(``~/.iai-mcp/eval-fixtures/labelled_real_thread_cues.json``); the write
step refuses to write inside the repo tree unless ``--force-in-repo`` is
passed, and ``.gitignore`` is the final catch even then.

No LLM-assisted labelling: the only automation here is
``store.exact_top_k`` — the owner confirms or rejects every suggestion in
the terminal, and every suggestion is displayed verbatim (first-N-chars,
never paraphrased).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench.recall_accuracy_real import _copy_real_store, _open_copy_store_shared  # noqa: E402
from iai_mcp.embed import embedder_for_store  # noqa: E402

_DEFAULT_FIXTURE_OUT = Path("~/.iai-mcp/eval-fixtures/labelled_real_thread_cues.json").expanduser()
_DEFAULT_TARGET_COUNT = 30
_AUTO_SUGGEST_K = 20
_SNIPPET_CHARS = 120


def _mine_candidates(store, sample_n: int) -> list:
    all_role_user = [
        r for r in store.iter_records(where="tier = 'episodic'")
        if "role:user" in (r.tags or [])
    ]
    if len(all_role_user) <= sample_n:
        return all_role_user
    step = max(1, len(all_role_user) // sample_n)
    return all_role_user[::step][:sample_n]


def _snippet(text: str) -> str:
    return text[:_SNIPPET_CHARS] + ("..." if len(text) > _SNIPPET_CHARS else "")


def _present_and_confirm(cue_record, store, embedder, index: int) -> dict | None:
    print(f"\n{'=' * 72}")
    print(f"[candidate {index}] id={cue_record.id}")
    print(f"cue text: {cue_record.literal_surface}")
    print(f"{'=' * 72}")

    decision = input("Use this cue? [y/n/q/notes-then-y]: ").strip()
    if decision.lower() == "q":
        return {"_quit": True}
    if not decision or decision.lower() == "n":
        return None

    notes = ""
    accept = decision.lower() == "y"
    if not accept and decision.lower().startswith("y"):
        # "y notes: ..." form
        accept = True
        if ":" in decision:
            notes = decision.split(":", 1)[1].strip()
    if not accept:
        return None

    cue_vec = embedder.embed(cue_record.literal_surface)
    suggestions = store.exact_top_k(list(cue_vec), k=_AUTO_SUGGEST_K, build_if_cold=True)

    print(f"\nTop {len(suggestions)} auto-suggested candidates (exact_top_k):")
    suggestion_records = []
    for i, (rid, cosine) in enumerate(suggestions):
        rec = store.get(rid)
        if rec is None:
            continue
        suggestion_records.append((i, rid, rec))
        print(f"  [{i}] {rid} | cos={cosine:.3f} | {_snippet(rec.literal_surface)}")

    picked_raw = input(
        'Relevant indices (comma-sep, empty=none, e.g. "0,2,5"; or "+ <uuid>" to add manually): '
    ).strip()

    relevant_ids: list[str] = []
    if picked_raw:
        for token in picked_raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token.startswith("+"):
                manual_id = token[1:].strip()
                if manual_id:
                    relevant_ids.append(manual_id)
                continue
            try:
                idx = int(token)
            except ValueError:
                continue
            for i, rid, _rec in suggestion_records:
                if i == idx:
                    relevant_ids.append(str(rid))
                    break

    if not relevant_ids:
        print("No relevant ids selected; skipping this cue.")
        return None

    expects_structural_raw = input("expects_structural? [y/N]: ").strip().lower()
    expects_structural = expects_structural_raw == "y"

    if not notes:
        notes = input("notes (optional): ").strip()

    return {
        "cue_id": f"cue_{index:04d}",
        "cue_text": cue_record.literal_surface,
        "relevant_record_ids": relevant_ids,
        "notes": notes,
        "expects_structural": expects_structural,
    }


def _write_fixture(entries: list[dict], out_path: Path, *, force_in_repo: bool) -> None:
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
    print(f"\nWrote {len(entries)} labelled cues to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive labelling: mine cues from a read-only COPY of your Hippo, "
            "auto-suggest relevant memories via exact_top_k, confirm per cue, "
            "write a LOCAL-only JSON fixture. NEVER commits."
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

    with tempfile.TemporaryDirectory(prefix="iai-mcp-label-copy-") as td:
        copy_root = _copy_real_store(Path(td) / "copy")
        store = _open_copy_store_shared(copy_root)
        try:
            embedder = embedder_for_store(store)
            candidates = _mine_candidates(store, sample_n=args.target_count * 3)
            entries: list[dict] = []
            for cand in candidates:
                if len(entries) >= args.target_count:
                    break
                entry = _present_and_confirm(cand, store, embedder, len(entries) + 1)
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
