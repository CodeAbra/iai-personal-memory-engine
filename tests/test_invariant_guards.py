"""Grep-based static guards for the runtime's non-negotiable invariants.

Catalog:
- No paid API: ANTHROPIC_API_KEY and the Anthropic SDK must not appear in
  runtime code — the only LLM channel is the `claude -p` subprocess billed
  to the user's subscription.
- No fcntl.lockf (close-fd trap) anywhere in src/iai_mcp/.
- Verbatim preservation: no assignment to `.literal_surface` in daemon-side
  modules.
- No hardcoded Western clock-time in quiet_window.py — the quiet window is
  learned from event history.
- Sealed registry: PROFILE_KNOBS has exactly 11 entries (daemon does NOT
  add knobs).
- identity_audit.py does NOT import ProcessLock / concurrency module.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "iai_mcp"

# Daemon-side modules; the list tolerates absent entries so it can be
# declared ahead of the module. We scan whichever ones exist today.
DAEMON_MODULES: tuple[str, ...] = (
    "daemon/__init__.py",
    "daemon/__main__.py",
    "daemon/_watchdog.py",
    "dream.py",
    "identity_audit.py",
    "bedtime.py",
    "claude_cli.py",
    "insight.py",
    "quiet_window.py",
    "daemon_state.py",
    "concurrency.py",
    "hippea_cascade.py",
)


def _existing_daemon_files() -> list[Path]:
    return [SRC / n for n in DAEMON_MODULES if (SRC / n).exists()]


def test_daemon_files_are_actually_scanned():
    """The daemon-side guard scan filters on `.exists()` and FAILS OPEN: if the
    daemon source path stops resolving, its entry is silently dropped and the
    no-paid-API-key / verbatim-preservation guards go dark with no error. Pin
    the daemon package file into the resolved scan list so a bad repath fails
    loud instead of losing coverage."""
    scanned = _existing_daemon_files()
    assert scanned, "daemon-side guard scan resolved to an empty file list"
    assert (SRC / "daemon" / "__init__.py") in scanned, (
        "daemon package source is not in the scanned guard list — "
        "the no-paid-API-key / verbatim guards would silently skip it"
    )


# ---------------------------------------------------------------------------
# ANTHROPIC_API_KEY must never appear in daemon-side code
# ---------------------------------------------------------------------------

def test_no_api_key_in_daemon():
    """Zero paid-API cost. ANTHROPIC_API_KEY must not appear in ANY
    daemon-side module. Insight module uses `claude -p` subprocess with the
    user's subscription instead."""
    offenders: list[str] = []
    for f in _existing_daemon_files():
        text = f.read_text()
        if "ANTHROPIC_API_KEY" in text:
            offenders.append(f.name)
    assert not offenders, f"paid-API violation: ANTHROPIC_API_KEY found in {offenders}"


# ---------------------------------------------------------------------------
# Wide-scan guards over ALL of src/iai_mcp/**/*.py — a module whitelist can
# silently miss new files, so the paid-API guards below glob everything.
# ---------------------------------------------------------------------------


def _all_iai_mcp_files() -> list[Path]:
    """All Python source files under src/iai_mcp/, recursive — every src
    module is scanned by default, no explicit allow-listing."""
    return sorted(SRC.rglob("*.py"))


def test_no_api_key_anywhere_in_src():
    """ANTHROPIC_API_KEY must not appear in ANY file under src/iai_mcp/."""
    offenders: list[str] = []
    for f in _all_iai_mcp_files():
        text = f.read_text()
        if "ANTHROPIC_API_KEY" in text:
            offenders.append(str(f.relative_to(SRC.parent.parent)))
    assert not offenders, (
        f"paid-API violation: ANTHROPIC_API_KEY found in {offenders}. "
        "claude_cli.invoke_claude_sync via subscription is the only LLM channel."
    )


def test_no_anthropic_sdk_import_anywhere_in_src():
    """`import anthropic` and `from anthropic` are forbidden anywhere under
    src/iai_mcp/ — the SDK is not a runtime dependency.

    `claude_cli.py` may legitimately reference the string "anthropic" inside
    its env-deny-list (built from fragments, never as a literal import) and
    inside docstrings; this guard greps for actual import statements only.
    """
    import_pattern = re.compile(r"^(?:from anthropic\b|import anthropic\b)", re.MULTILINE)
    offenders: list[tuple[str, list[str]]] = []
    for f in _all_iai_mcp_files():
        text = f.read_text()
        matches = import_pattern.findall(text)
        if matches:
            offenders.append((str(f.relative_to(SRC.parent.parent)), matches))
    assert not offenders, (
        f"paid-API violation: `import anthropic` / `from anthropic` in "
        f"{offenders}. The SDK is not a runtime dependency."
    )


def test_no_anthropic_client_construction_anywhere_in_src():
    """`anthropic.Anthropic(...)` client construction is forbidden — an SDK
    client exists only to make paid-API calls."""
    offenders: list[tuple[str, str]] = []
    for f in _all_iai_mcp_files():
        text = f.read_text()
        if "anthropic.Anthropic(" in text:
            # Surface the surrounding line for diagnostic clarity.
            for line in text.splitlines():
                if "anthropic.Anthropic(" in line:
                    offenders.append(
                        (str(f.relative_to(SRC.parent.parent)), line.strip()),
                    )
    assert not offenders, (
        f"paid-API violation: anthropic.Anthropic() construction in {offenders}"
    )


def test_no_anthropic_messages_sdk_calls_anywhere_in_src():
    """Anthropic SDK method patterns are forbidden — nightly evaluation goes
    through the batched subscription path in
    `reconsolidation_critic.evaluate_batch_reconsolidation`, never through
    the SDK batch/messages surface.
    """
    forbidden_patterns = (
        "messages.batches.create",
        "messages.batches.retrieve",
        "messages.batches.results",
        # A per-record critic loop over messages.create would be a paid-API
        # runaway; flag any emergence. Note: this matches the SDK method, NOT
        # the local claude_cli subprocess (which uses subprocess.run / asyncio).
        ".messages.create(",
    )
    offenders: list[tuple[str, str]] = []
    for f in _all_iai_mcp_files():
        text = f.read_text()
        for pat in forbidden_patterns:
            if pat in text:
                for line in text.splitlines():
                    if pat in line:
                        offenders.append(
                            (str(f.relative_to(SRC.parent.parent)), line.strip()),
                        )
    assert not offenders, (
        f"paid-API violation: Anthropic SDK call pattern in {offenders}"
    )


def test_reconsolidation_critic_does_not_modify_literal_surface():
    """Cognitive integrity (Mottron EPF verbatim invariant): the Tier-1
    critic must never paraphrase, smooth, or otherwise rewrite the
    `literal_surface` of a memory record. It is permitted to ANNOTATE via
    `prediction_error` (a separate float field) and via
    `append_provenance({"prediction_error": ...})`, but must not assign to
    `.literal_surface` or push a new surface into the record via
    `store.insert`.

    This guard greps the reconsolidation_critic module for forbidden write
    patterns — both direct attribute assignment and any pattern that would
    rebuild + reinsert a record with mutated surface.
    """
    f = SRC / "reconsolidation_critic.py"
    assert f.exists(), "reconsolidation_critic.py missing"
    text = f.read_text()
    forbidden = (
        re.compile(r"\.literal_surface\s*="),
        re.compile(r"store\.insert\b"),
        re.compile(r"rec\.literal_surface\s*="),
    )
    offenders: list[str] = []
    for pat in forbidden:
        if pat.search(text):
            offenders.append(pat.pattern)
    assert not offenders, (
        f"cognitive-integrity violation in reconsolidation_critic.py: "
        f"forbidden write patterns {offenders}"
    )


def test_reconsolidation_critic_cap_constant_present():
    """`MAX_RECORDS_PER_CALL` cap is the load-bearing safety knob that turns
    the batched critic from a runaway per-record loop into the '1 call/night'
    invariant. Guard against accidental removal."""
    from iai_mcp.reconsolidation_critic import MAX_RECORDS_PER_CALL

    assert isinstance(MAX_RECORDS_PER_CALL, int)
    assert 1 <= MAX_RECORDS_PER_CALL <= 200, (
        f"cap drifted: MAX_RECORDS_PER_CALL={MAX_RECORDS_PER_CALL}. "
        "Tunable but must stay bounded — a runaway per-record loop is exactly "
        "what the cap exists to prevent."
    )


def test_no_anthropic_dependency_in_pyproject():
    """`anthropic` must not appear as a runtime dependency in
    pyproject.toml; this guard prevents accidental re-pin."""
    pyproject = SRC.parent.parent / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml missing"
    text = pyproject.read_text()
    # Block actual dependency lines like `"anthropic>=0.40.0",`, but allow
    # comments mentioning the SDK.
    dep_pattern = re.compile(r'^\s*"anthropic[>=<~!]', re.MULTILINE)
    offenders = dep_pattern.findall(text)
    assert not offenders, (
        f"paid-API violation: anthropic dependency re-pinned in pyproject.toml: "
        f"{offenders}"
    )


# ---------------------------------------------------------------------------
# fcntl.lockf must never be used (POSIX close-fd trap)
# ---------------------------------------------------------------------------

def test_no_lockf_anywhere():
    """POSIX fcntl.lockf is released when ANY fd referring to the same file
    is closed (apenwarr 2010). We must use BSD fcntl.flock which is bound to
    the open file description. Scan ALL iai_mcp/*.py, not just daemon
    modules -- mixing the two is also a bug."""
    offenders: list[str] = []
    for f in SRC.glob("*.py"):
        text = f.read_text()
        if "fcntl.lockf" in text:
            offenders.append(f.name)
    assert not offenders, f"close-fd-trap violation: fcntl.lockf in {offenders}"


# ---------------------------------------------------------------------------
# Daemon must NEVER assign to record.literal_surface
# ---------------------------------------------------------------------------

def test_no_literal_surface_mutation_in_daemon():
    """Literal preservation. Daemon-side modules must not contain
    `.literal_surface =` assignment syntax. Reading `.literal_surface` is
    allowed; writing is forbidden."""
    pattern = re.compile(r"\.literal_surface\s*=")
    offenders: list[tuple[str, list[str]]] = []
    for f in _existing_daemon_files():
        text = f.read_text()
        matches = pattern.findall(text)
        if matches:
            offenders.append((f.name, matches))
    assert not offenders, f"verbatim-preservation violation: {offenders}"


# ---------------------------------------------------------------------------
# No hardcoded Western 9-5 / clock-time in quiet_window.py
# ---------------------------------------------------------------------------

def test_no_hardcoded_clock_time_in_quiet_window():
    """Global-product mandate: quiet window must be LEARNED from event
    history, never hardcoded. Flag obvious clock-time literals."""
    f = SRC / "quiet_window.py"
    if not f.exists():
        return  # module not yet created
    text = f.read_text()
    # Look for common patterns that would indicate clock-based decisions.
    forbidden = [
        r"\b22:00\b",
        r"\b02:00\b",
        r"hour\s*==\s*22\b",
        r"hour\s*==\s*2\b",
    ]
    offenders: list[str] = []
    for pat in forbidden:
        if re.search(pat, text):
            offenders.append(pat)
    assert not offenders, (
        f"learned-quiet-window violation: hardcoded clock-time patterns in "
        f"quiet_window.py: {offenders}"
    )


# ---------------------------------------------------------------------------
# Sealed registry: PROFILE_KNOBS has exactly 11 entries
# (10 autistic-kernel + 1 operator wake_depth MCP-12)
# ---------------------------------------------------------------------------

def test_profile_knobs_still_sealed():
    """11-knob registry is sealed. Daemon must not add new knobs. Transient
    state (hebbian-rate boost during developmental sigma, etc.) belongs in
    events or .daemon-state.json, never in PROFILE_KNOBS."""
    from iai_mcp import profile
    assert len(profile.PROFILE_KNOBS) == 11, (
        f"PROFILE_KNOBS unseal: expected 11, got {len(profile.PROFILE_KNOBS)}"
    )


# ---------------------------------------------------------------------------
# Profile knob names must NEVER appear in the session-start payload at any
# wake_depth. Knobs are applied server-side via
# response_decorator.apply_profile; their names must not cross the MCP wire.
# ---------------------------------------------------------------------------


def test_no_profile_knob_in_session_start_payload(tmp_path):
    """Knob names must not leak into the lazy pointer fields at
    wake_depth=minimal (<=30 raw tok design budget).

    The legacy L0 identity kernel (`_seed_l0_identity`) historically recites
    a handful of autistic-kernel defaults inline in the literal_surface
    ('literal_preservation=strong, masking_off=true, ...'). That predates
    this guard and lives inside the user's identity record itself, not a
    decorator output — so it's scoped into the standard/deep l0 segment and
    explicitly exempt from this grep guard.

    The invariant this guard DEFENDS is: the lazy minimal payload
    (identity_pointer / brain_handle / topic_cluster_hint) MUST NOT contain
    knob names. Knobs are applied server-side by response_decorator; knob
    names must never reach the MCP wire.
    """
    from iai_mcp import profile
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.core import _seed_l0_identity
    from iai_mcp.session import assemble_session_start
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    assignment = CommunityAssignment()

    for mode in ("minimal", "standard", "deep"):
        state = profile.default_state()
        state["wake_depth"] = mode
        payload = assemble_session_start(
            store, assignment, [], profile_state=state,
        )
        # Only scan the lazy fields. Legacy l0 / l1 / l2 / rich_club carry
        # user-authored identity content and remain exempt per design.
        lazy_text = " ".join(
            [
                payload.identity_pointer,
                payload.brain_handle,
                payload.topic_cluster_hint,
            ],
        )
        for knob_name in profile.PROFILE_KNOBS:
            # wake_depth is the operator-facing knob; its echo in the
            # payload field `wake_depth` is a meta-attribute, not inline
            # knob text in the lazy pointers.
            assert knob_name not in lazy_text, (
                f"knob-leak violation: knob name '{knob_name}' found in "
                f"lazy session-start payload at wake_depth={mode} "
                f"(identity_pointer/brain_handle/topic_cluster_hint)"
            )


# ---------------------------------------------------------------------------
# wake_depth=minimal payload (<=30 raw tok) is below the Anthropic prompt-
# cache minimum (2048 tok). Adding cache_control in session.py would be
# silently ignored — wastes a breakpoint slot. Guard against regression.
# ---------------------------------------------------------------------------


def test_no_cache_control_in_session_assembler():
    """session.py must not set cache_control (the minimal prefix cannot be
    cached; standard+deep caching lives in the TS wrapper, not the Python
    assembler).
    """
    f = SRC / "session.py"
    assert f.exists(), "session.py missing"
    text = f.read_text()
    # Comments that mention "cache_control" are fine (they document the
    # pitfall). We only guard against actual code references like setattr/
    # cache_control=... — scan for the pattern with an equals sign.
    pattern = re.compile(r"cache_control\s*[:=]")
    offenders = pattern.findall(text)
    assert not offenders, (
        f"cache-minimum violation: cache_control assignment/kwarg in "
        f"session.py: {offenders}"
    )


# ---------------------------------------------------------------------------
# response_decorator must be pure-local. No Anthropic SDK import, no
# ANTHROPIC_API_KEY read, no paid-API coupling.
# ---------------------------------------------------------------------------


def test_no_api_key_in_response_decorator():
    """response_decorator.py stays local-only."""
    f = SRC / "response_decorator.py"
    assert f.exists(), "response_decorator.py missing"
    text = f.read_text()
    lower = text.lower()
    assert "anthropic" not in lower, (
        "paid-API violation: response_decorator references 'anthropic'"
    )
    assert "ANTHROPIC_API_KEY" not in text, (
        "paid-API violation: response_decorator references ANTHROPIC_API_KEY"
    )
    assert "import anthropic" not in lower, (
        "paid-API violation: response_decorator imports anthropic SDK"
    )


# ---------------------------------------------------------------------------
# identity_audit.py must not import ProcessLock
# ---------------------------------------------------------------------------

def test_identity_audit_has_no_lock_import():
    """Continuous audit runs even when the daemon is paused. To make that
    invariant mechanical, identity_audit.py must NOT import the concurrency
    module -- the only way to accidentally take a lock is to import it."""
    f = SRC / "identity_audit.py"
    if not f.exists():
        return
    text = f.read_text()
    # No import of iai_mcp.concurrency, no `ProcessLock` symbol reference.
    assert "iai_mcp.concurrency" not in text, (
        "lock-free-audit violation: identity_audit.py imports iai_mcp.concurrency"
    )
    assert "ProcessLock" not in text, (
        "lock-free-audit violation: identity_audit.py references ProcessLock"
    )
    # Also: no `fcntl.` calls (belt-and-braces).
    assert "fcntl." not in text, (
        "lock-free-audit violation: identity_audit.py uses fcntl directly"
    )


# ---------------------------------------------------------------------------
# HIPPEA cascade module guards
# ---------------------------------------------------------------------------

def test_no_api_key_in_hippea_cascade():
    """HIPPEA cascade is pure-local. ANTHROPIC_API_KEY and `anthropic` SDK
    imports are forbidden in hippea_cascade.py."""
    f = SRC / "hippea_cascade.py"
    if not f.exists():
        return  # module not yet created
    text = f.read_text()
    assert "ANTHROPIC_API_KEY" not in text, (
        "paid-API violation: ANTHROPIC_API_KEY in hippea_cascade.py"
    )
    assert "import anthropic" not in text, (
        "paid-API violation: `import anthropic` in hippea_cascade.py"
    )
    assert "from anthropic" not in text, (
        "paid-API violation: `from anthropic` in hippea_cascade.py"
    )


def test_hippea_cascade_is_read_only_against_store():
    """Cascade prefetch never mutates the store.

    Grep for store-mutating call patterns (with trailing open-paren so the
    module's own enumerated-forbidden list in the docstring does not trip
    this guard).
    """
    f = SRC / "hippea_cascade.py"
    if not f.exists():
        return
    text = f.read_text()
    forbidden_calls = [
        "store.insert(",
        "store.append_provenance(",
        "store.append_provenance_batch(",
        "store.update(",
        "store.boost_edges(",
        "store.add_contradicts_edge(",
    ]
    offenders = [p for p in forbidden_calls if p in text]
    assert not offenders, (
        f"read-only violation: hippea_cascade.py contains store mutators: {offenders}"
    )
