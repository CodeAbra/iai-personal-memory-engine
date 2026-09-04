"""Single-store render A/B token witness for the render-only cls_summary
lever (rich_club strip-then-cap at session.py).

ONE build_runtime_graph pass on a read-only real-store copy. The admitted-id
list is derived by calling the shipped admission function directly
(session.py's ``_rich_club_admission``) — never a reimplementation of its
selection loop. BEFORE and AFTER are both rendered from that SAME admitted
list with a single shared ``now``, so the admitted-id set is identical
across them by construction and the two passes cannot desync at a minute
boundary. Because storage and embeddings are never touched by this lever,
this is a pure render-side comparison, not a before/after store mutation.
Never opens or mutates the operator's live store: every step runs against a
read-only-sourced TemporaryDirectory copy
(bench/recall_accuracy_real.py::open_eval_copy_store).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bench.recall_accuracy_real import open_eval_copy_store
from bench.token_breakdown import _measure_text, _tiktoken_encoder
from iai_mcp import runtime_graph_cache
from iai_mcp.retrieve import build_runtime_graph
from iai_mcp.session import (
    RICH_CLUB_BUDGET_TOKENS,
    RICH_CLUB_CLS_SUMMARY_CAP,
    _clean_surface,
    _rich_club_admission,
    _rich_club_segment_with_budget,
)
from iai_mcp.types import CLS_SUMMARY_PREFIX_RE


def _real_store_present() -> bool:
    from iai_mcp.hippo import _operator_home

    return (_operator_home() / ".iai-mcp" / "hippo" / "brain.sqlite3").exists()


def _run_witness(store) -> None:
    runtime_graph_cache.invalidate(store)
    graph, assignment, rich_club = build_runtime_graph(store)
    assert rich_club, "real store's rich_club membership is empty — nothing to measure"

    now = datetime.now(timezone.utc)
    admission = _rich_club_admission(store, rich_club, budget=RICH_CLUB_BUDGET_TOKENS, now=now)
    admitted = [uid for uid, *_ in admission]
    assert admitted, "the shipped admission selected zero records — nothing to measure"

    after_text = _rich_club_segment_with_budget(
        store, rich_club, budget=RICH_CLUB_BUDGET_TOKENS, now=now
    )
    after_lines = after_text.splitlines()

    # Anchor: both passes call the SAME admission function with the SAME
    # `now`, so an equal count pins the identical admitted id-set by
    # construction — not a re-derivation of shipped internals.
    assert len(after_lines) == len(admitted), (
        f"admitted set ({len(admitted)} ids) does not match the shipped "
        f"render's line count ({len(after_lines)})"
    )

    by_uuid = store.get_batch(admitted)
    before_lines = []
    matched_pre_real: list[int] = []
    matched_post_real: list[int] = []
    for uid, after_line in zip(admitted, after_lines):
        prefix_part, after_content = after_line.split(": ", 1)
        rec = by_uuid[uid]
        cleaned = _clean_surface(rec.literal_surface)
        before_content = cleaned[:60]
        before_lines.append(f"{prefix_part}: {before_content}")

        if "cls_summary" in rec.tags:
            m = CLS_SUMMARY_PREFIX_RE.match(rec.literal_surface)
            if m is not None:
                assert len(after_content) <= RICH_CLUB_CLS_SUMMARY_CAP, (
                    f"matched cls_summary content exceeds the cap: "
                    f"{len(after_content)} > {RICH_CLUB_CLS_SUMMARY_CAP}"
                )
                prefix_len = len(m.group(0))
                matched_pre_real.append(max(0, len(before_content) - prefix_len))
                matched_post_real.append(len(after_content))

    # Checked BEFORE the token-reduction assertion: with zero matched
    # cls_summary lines in the admitted set (e.g. a degraded/neutral
    # centrality this cycle seats no cls_summary hub in rich_club), BEFORE
    # and AFTER are byte-identical by construction and the reduction
    # assertion below would fail for a harness reason, not a lever reason —
    # this state must skip, never hard-fail.
    if not matched_pre_real:
        pytest.skip(
            "no matched cls_summary line reached the admitted rich_club set "
            "on this real-store copy — the token-reduction and "
            "real-content-per-line witnesses would be vacuous on this "
            "corpus state"
        )

    before_text = "\n".join(before_lines)

    encoder = _tiktoken_encoder()
    tokens_before = _measure_text(before_text, encoder)["tiktoken_tokens"]
    tokens_after = _measure_text(after_text, encoder)["tiktoken_tokens"]

    assert tokens_after < tokens_before, (
        f"rich_club tiktoken tokens did not go down under the render-only "
        f"strip-then-cap: before={tokens_before} after={tokens_after} "
        f"admitted={len(admitted)}"
    )

    mean_pre = sum(matched_pre_real) / len(matched_pre_real)
    mean_post = sum(matched_post_real) / len(matched_post_real)
    assert mean_post >= mean_pre, (
        "mean visible real-content chars per matched cls_summary line "
        f"decreased: pre={mean_pre:.2f} post={mean_post:.2f} "
        f"n={len(matched_pre_real)}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_real_store_single_build_render_ab_token_witness(driver, monkeypatch):
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    # open_eval_copy_store mutates os.environ directly (unscoped); prime
    # monkeypatch's restore BEFORE it runs so the original env is restored
    # at teardown regardless of that direct mutation.
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    if not _real_store_present():
        pytest.skip(
            "real store not found; this is a real-store witness test, not a "
            "fresh-clone CI gate"
        )

    # Scoped tightly to the store-OPEN step: a failure here (on-disk format
    # mismatch under the lilli driver) is a second, distinct skip from the
    # unbuilt-extension ImportError skip above. A failure raised by the
    # witness logic itself (below) must propagate as a real test failure,
    # never get reinterpreted as a skip.
    cm = open_eval_copy_store(driver=driver)
    try:
        store = cm.__enter__()
    except Exception as exc:
        if driver == "lilli":
            pytest.skip(
                "lilli driver could not open the real-store copy (on-disk "
                f"format mismatch or driver error): {exc}"
            )
        raise

    try:
        _run_witness(store)
    finally:
        cm.__exit__(None, None, None)
