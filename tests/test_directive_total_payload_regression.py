"""Session-start total payload stays within the measured token baseline: no
directive truncated, the directive block stays within its own budget, and a
minimum reservation survives for the rest of the payload.

The baseline is MEASURED dynamically on a representative fixture composed
through the real production path (assemble_session_start ->
format_payload_as_markdown), never a hard-coded assumed figure.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp import working_tier as wt
from iai_mcp.capture import capture_turn
from iai_mcp.community import CommunityAssignment
from iai_mcp.core import _seed_l0_identity
from iai_mcp.daemon_state import MAX_RUNNING_AGENTS, register_running_agent
from iai_mcp.directive_budget import (
    AGENT_REGISTRY_BUDGET_TOKENS,
    AGENT_REGISTRY_LINE_CHAR_CAP,
    AGENT_REGISTRY_MAX_RENDERED,
    CONT_BUDGET_TOKENS,
    DIRECTIVE_BUDGET_TOKENS,
    DIRECTIVE_MAX_COUNT,
    DIRECTIVE_LINE_CHAR_CAP,
)
from iai_mcp.session import (
    _approx_tokens,
    assemble_session_start,
    format_payload_as_markdown,
    render_agent_registry_segment,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
from iai_mcp.working_tier import WORKING_TIER_MAX_GOAL_CHARS

_STANDARD = {"wake_depth": "standard"}

_SESSION_START_TOTAL_BUDGET_TOKENS = 3000

_FILLER_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet "


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    wt._reset()
    yield
    wt._reset()


def _near_cap_live_state_fields() -> tuple[str, str, str]:
    """Focus and next_action sized to exactly the per-field char cap they
    fold through uncut; goal (unbounded) absorbs the remainder so the
    rendered live-state block still lands just under CONT_BUDGET_TOKENS."""
    focus_chars = WORKING_TIER_MAX_GOAL_CHARS
    next_chars = WORKING_TIER_MAX_GOAL_CHARS
    overhead_chars = len("goal: \nfocus: \nnext action: ")
    safety_margin_chars = 8
    target_total_chars = CONT_BUDGET_TOKENS * 4 - safety_margin_chars
    goal_chars = max(target_total_chars - overhead_chars - focus_chars - next_chars, 0)
    filler = _FILLER_WORDS * 200
    goal = ("joint budget regression focal goal: " + filler)[:goal_chars]
    focus = ("joint budget regression focus: " + filler)[:focus_chars]
    next_action = ("joint budget regression next action: " + filler)[:next_chars]
    assert len(focus.strip()) == WORKING_TIER_MAX_GOAL_CHARS, (
        "focus slice must land on a non-whitespace boundary to exercise the cap"
    )
    assert len(next_action.strip()) == WORKING_TIER_MAX_GOAL_CHARS, (
        "next_action slice must land on a non-whitespace boundary to exercise the cap"
    )
    return goal, focus, next_action


def _near_cap_text(tag: str) -> str:
    prefix = f"standing order {tag}: "
    body = (_FILLER_WORDS * 20)[: DIRECTIVE_LINE_CHAR_CAP - len(prefix)]
    text = prefix + body
    assert len(text) == DIRECTIVE_LINE_CHAR_CAP
    return text


def _record(text: str, *, pinned: bool = False, detail_level: int = 1, community_id=None) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=community_id,
        centrality=0.5,
        detail_level=detail_level,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _seed_representative_store(store: MemoryStore) -> list[UUID]:
    """L0 + pinned L1 + rich club at a realistic operator-payload scale --
    not the empty-store trivial case."""
    _seed_l0_identity(store)

    for i in range(8):
        rec = _record(
            f"Standing project fact #{i}: representative pinned detail "
            "content used for the baseline measurement.",
            pinned=True, detail_level=5,
        )
        store.insert(rec)

    rich_club_ids: list[UUID] = []
    for i in range(15):
        rec = _record(
            f"rich-club representative memory entry {i}: " + ("realistic content " * 6)
        )
        store.insert(rec)
        rich_club_ids.append(rec.id)

    return rich_club_ids


def _seed_community_assignment(store: MemoryStore) -> CommunityAssignment:
    assignment = CommunityAssignment()
    for i in range(5):
        cid = uuid4()
        members: list[UUID] = []
        for j in range(2):
            rec = _record(f"community {i} member {j} representative content", community_id=cid)
            store.insert(rec)
            members.append(rec.id)
        assignment.top_communities.append(cid)
        assignment.mid_regions[cid] = members
        assignment.community_centroids[cid] = [0.0] * EMBED_DIM
    return assignment


def _measure_payload_tokens(store: MemoryStore, assignment: CommunityAssignment, rich_club_ids: list[UUID]) -> int:
    payload = assemble_session_start(store, assignment, rich_club_ids, profile_state=_STANDARD)
    rendered = format_payload_as_markdown(payload)
    return _approx_tokens(rendered)


def test_baseline_measured(tmp_path):
    store = MemoryStore(path=tmp_path)
    rich_club_ids = _seed_representative_store(store)
    assignment = _seed_community_assignment(store)

    measured_baseline = _measure_payload_tokens(store, assignment, rich_club_ids)
    assert measured_baseline > 0, "baseline must be a real measurement, not zero"
    assert measured_baseline == _measure_payload_tokens(store, assignment, rich_club_ids), (
        "the measurement must be deterministic across repeated composition"
    )

    reserved_for_directives = (
        _SESSION_START_TOTAL_BUDGET_TOKENS - CONT_BUDGET_TOKENS - measured_baseline
    )
    assert reserved_for_directives >= DIRECTIVE_BUDGET_TOKENS, (
        f"measured baseline {measured_baseline} leaves only {reserved_for_directives} "
        f"tokens for the directive tier, below its {DIRECTIVE_BUDGET_TOKENS}-token budget "
        f"(total={_SESSION_START_TOTAL_BUDGET_TOKENS}, CONT reserve={CONT_BUDGET_TOKENS})"
    )


def test_full_directive_tier_no_truncation_within_total_budget(tmp_path):
    store = MemoryStore(path=tmp_path)
    rich_club_ids = _seed_representative_store(store)
    assignment = _seed_community_assignment(store)
    baseline = _measure_payload_tokens(store, assignment, rich_club_ids)

    n_seed = DIRECTIVE_MAX_COUNT - 1
    assert n_seed < DIRECTIVE_MAX_COUNT
    seeded_texts: list[str] = []
    for i in range(n_seed):
        text = _near_cap_text(f"seed{i:02d}")
        result = capture_turn(
            store=store, cue="c", text=text,
            directive=True, session_id="s1", role="user",
        )
        assert result["status"] == "inserted", result
        rec = store.get(UUID(result["record_id"]))
        assert rec is not None and rec.directive is True
        seeded_texts.append(text)

    payload = assemble_session_start(store, assignment, rich_club_ids, profile_state=_STANDARD)
    rendered = format_payload_as_markdown(payload)
    total_tokens = _approx_tokens(rendered)

    assert total_tokens <= _SESSION_START_TOTAL_BUDGET_TOKENS, (
        f"total payload {total_tokens} exceeds the {_SESSION_START_TOTAL_BUDGET_TOKENS}-token ceiling"
    )

    for text in seeded_texts:
        assert text in payload.directives, (
            f"directive {text!r} missing from the rendered block -- truncation occurred"
        )
        assert text in rendered

    directive_block_tokens = _approx_tokens(payload.directives)
    assert directive_block_tokens <= DIRECTIVE_BUDGET_TOKENS, (
        f"directive block {directive_block_tokens} tokens exceeds its "
        f"{DIRECTIVE_BUDGET_TOKENS}-token budget"
    )

    reserved_for_cont = (
        _SESSION_START_TOTAL_BUDGET_TOKENS - baseline - directive_block_tokens
    )
    assert reserved_for_cont >= CONT_BUDGET_TOKENS, (
        f"baseline {baseline} + directive block {directive_block_tokens} leaves only "
        f"{reserved_for_cont} tokens for CONT, below the {CONT_BUDGET_TOKENS}-token reserve"
    )


def test_full_directive_tier_and_live_state_joint_budget_no_truncation(tmp_path):
    """The joint DIR+CONT budget proven against a REAL measured live-state
    block (not the reserved constant) alongside a full directive tier --
    nothing truncated, everything under the 3000-token session-start ceiling.
    """
    store = MemoryStore(path=tmp_path)
    rich_club_ids = _seed_representative_store(store)
    assignment = _seed_community_assignment(store)
    baseline = _measure_payload_tokens(store, assignment, rich_club_ids)

    n_seed = DIRECTIVE_MAX_COUNT - 1
    seeded_texts: list[str] = []
    for i in range(n_seed):
        text = _near_cap_text(f"joint{i:02d}")
        result = capture_turn(
            store=store, cue="c", text=text,
            directive=True, session_id="s1", role="user",
        )
        assert result["status"] == "inserted", result
        seeded_texts.append(text)

    goal, focus, next_action = _near_cap_live_state_fields()
    wt.open_task(goal, session_id="s1")
    wt.update_task(focus=focus, next_action=next_action, session_id="s1")

    payload = assemble_session_start(store, assignment, rich_club_ids, profile_state=_STANDARD)
    rendered = format_payload_as_markdown(payload)
    total_tokens = _approx_tokens(rendered)

    assert total_tokens <= _SESSION_START_TOTAL_BUDGET_TOKENS, (
        f"joint payload {total_tokens} exceeds the {_SESSION_START_TOTAL_BUDGET_TOKENS}-token ceiling"
    )

    for text in seeded_texts:
        # _clean_surface collapses/strips whitespace at render time -- a
        # near-cap-length seed can land its exact-length cut on a trailing
        # space; comparing the same normalized form on both sides still
        # proves no CONTENT truncation occurred.
        expected = text.rstrip()
        assert expected in payload.directives, (
            f"directive {expected!r} missing from the rendered block -- truncation occurred"
        )
        assert expected in rendered

    # open_task/update_task strip() the stored fields (verbatim body,
    # boundary whitespace only) -- compare against the same normalized form.
    goal, focus, next_action = goal.strip(), focus.strip(), next_action.strip()
    assert goal in payload.live_state, "live-state goal truncated"
    assert focus in payload.live_state, "live-state focus truncated"
    assert next_action in payload.live_state, "live-state next_action truncated"
    assert goal in rendered
    assert focus in rendered
    assert next_action in rendered

    directive_block_tokens = _approx_tokens(payload.directives)
    live_state_tokens = _approx_tokens(payload.live_state)

    assert live_state_tokens <= CONT_BUDGET_TOKENS, (
        f"live-state block {live_state_tokens} tokens exceeds its "
        f"{CONT_BUDGET_TOKENS}-token budget"
    )

    # Fourth joint-budget term: the agent-registry block is never folded
    # into assemble_session_start/format_payload_as_markdown (single
    # injection source is the eager file, not the composed payload), so it
    # is measured SEPARATELY off a real render_agent_registry_segment()
    # call and added arithmetically -- never searched for inside `rendered`.
    # Worst case: MAX_RUNNING_AGENTS pending agents, each pushing its
    # rendered line to the per-line char cap.
    for i in range(MAX_RUNNING_AGENTS):
        register_running_agent(
            agent_id=f"joint-budget-agent-{i:02d}",
            role="r" * 80,
            expected_artifact="e" * 200,
            agent_model="claude-sonnet-5",
        )
    agent_registry_rendered = render_agent_registry_segment()
    agent_registry_lines = agent_registry_rendered.split("\n")
    assert len(agent_registry_lines) == AGENT_REGISTRY_MAX_RENDERED, (
        "worst-case seeding must exercise the max-rendered-agents cap"
    )
    for line in agent_registry_lines:
        assert len(line) == AGENT_REGISTRY_LINE_CHAR_CAP, (
            f"line {line!r} was not truncated to the per-line char cap -- "
            "not the worst case"
        )
    agent_registry_tokens = _approx_tokens(agent_registry_rendered)
    assert agent_registry_tokens <= AGENT_REGISTRY_BUDGET_TOKENS, (
        f"agent-registry block {agent_registry_tokens} tokens exceeds its "
        f"{AGENT_REGISTRY_BUDGET_TOKENS}-token budget"
    )

    joint_total = baseline + directive_block_tokens + live_state_tokens
    assert joint_total <= _SESSION_START_TOTAL_BUDGET_TOKENS, (
        f"measured baseline {baseline} + directive block {directive_block_tokens} + "
        f"live-state block {live_state_tokens} = {joint_total} exceeds the "
        f"{_SESSION_START_TOTAL_BUDGET_TOKENS}-token ceiling"
    )

    joint_total_with_agent_registry = joint_total + agent_registry_tokens
    assert joint_total_with_agent_registry <= _SESSION_START_TOTAL_BUDGET_TOKENS, (
        f"joint payload {joint_total} + worst-case agent-registry block "
        f"{agent_registry_tokens} = {joint_total_with_agent_registry} exceeds the "
        f"{_SESSION_START_TOTAL_BUDGET_TOKENS}-token ceiling"
    )
