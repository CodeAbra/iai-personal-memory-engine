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
- Sealed registry: PROFILE_KNOBS has exactly 10 entries (daemon does NOT
  add knobs).
- identity_audit.py does NOT import ProcessLock / concurrency module.
- live/tombstoned_at co-occurrence: any source write that sets
  `tombstoned_at` must set `live` in the same write, covering both
  `values=` dict writes and SQL SET clauses (including f-strings).
- migrate staging seam: within src/iai_mcp/migrate/, no `.add(...)` call
  may be fed a row built by `store._to_row(...)` -- a full-table migrate
  swap must stage byte-for-byte from the SQL row so storage-only columns
  survive.
- generational cache counter purity: graph.py's incremental mutators
  (add_node, set_node_payload, remove_node, add_edge) may only touch the
  allow-listed pair `_pool_content_version` / `_dirty_since_centrality` as
  an incremented counter or boolean dirty-flag -- a third such attribute
  would desync a generational cache keyed on the version stamp.
- live-graph mutation locality: no assignment or pop/clear/setdefault/
  update call against `._node_payload[...]` / `._adj[...]` may appear
  outside graph.py -- every live-graph write must route through graph.py's
  own bump-carrying mutators.
"""
from __future__ import annotations

import ast
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
# Sealed registry: PROFILE_KNOBS has exactly 10 entries
# (9 autistic-kernel + 1 operator wake_depth MCP-12)
# ---------------------------------------------------------------------------

def test_profile_knobs_still_sealed():
    """10-knob registry is sealed. Daemon must not add new knobs. Transient
    state (hebbian-rate boost during developmental sigma, etc.) belongs in
    events or .daemon-state.json, never in PROFILE_KNOBS."""
    from iai_mcp import profile
    assert len(profile.PROFILE_KNOBS) == 10, (
        f"PROFILE_KNOBS unseal: expected 10, got {len(profile.PROFILE_KNOBS)}"
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


# ---------------------------------------------------------------------------
# live/tombstoned_at co-occurrence guard
#
# Rule: any source write that sets `tombstoned_at` must set `live` in the
# same write. Two shapes are checked:
#
#   Shape 1 -- a dict literal passed as the `values=` keyword of a call
#   (`tbl.update(where=..., values={"tombstoned_at": ..., "live": ...})`).
#   Gated on the `values=` keyword specifically so a plain dict that
#   happens to carry a `tombstoned_at` key for an unrelated purpose (a
#   journal record, a log payload) is never flagged.
#
#   Shape 2 -- a SQL string with a `SET` clause, covering both a plain
#   `ast.Constant` string and an `ast.JoinedStr` (f-string). Only the
#   segment between `SET` and `WHERE` is examined, so a WHERE clause that
#   mentions `tombstoned_at` (a reconciliation pass correcting drift, or
#   any read filter) is never flagged.
#
# Known blind spot: a dict passed positionally inside a list of tuples
# (e.g. a hypothetical `update_many_by_id([(rid, {"tombstoned_at": now})])`)
# is invisible to Shape 1, which gates on the `values=` keyword. Widening
# Shape 1 to "any dict literal with a tombstoned_at key" would false-positive
# on the journal-record shape above and require an allow-list that rots. No
# seam uses the positional-tuple shape today; each verb's own
# `COUNT(*) WHERE tombstoned_at IS NOT NULL AND live = 1 == 0` regression
# assertion is the backstop for that blind spot.
# ---------------------------------------------------------------------------

# Anchored to the `UPDATE ... SET` shape (not a bare `SET`) so a docstring
# or comment containing the English word "set" ahead of an unrelated
# `tombstoned_at =` cannot false-positive and break the build.
_SET_RE = re.compile(r"\bUPDATE\b.*?\bSET\b", re.IGNORECASE | re.DOTALL)
_WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)
_TOMBSTONED_AT_ASSIGN_RE = re.compile(r"tombstoned_at\s*=")
_LIVE_ASSIGN_RE = re.compile(r"\blive\s*=")


def _joinedstr_skeleton(node: ast.JoinedStr) -> str:
    """Concatenate the constant fragments of an f-string; interpolated
    values become a single-space gap so surrounding literal text stays
    adjacent for a `SET ... WHERE` scan."""
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            parts.append(" ")
    return "".join(parts)


def _set_segment(text: str) -> str | None:
    """Return the substring between the first `SET` and the next `WHERE`
    (case-insensitive), or None if the text carries no `SET` clause."""
    set_match = _SET_RE.search(text)
    if set_match is None:
        return None
    start = set_match.end()
    where_match = _WHERE_RE.search(text, start)
    end = where_match.start() if where_match is not None else len(text)
    return text[start:end]


class _TombstoneLiveVisitor(ast.NodeVisitor):
    def __init__(self, file: Path) -> None:
        self.file = file
        self.violations: list[tuple[Path, str, int]] = []
        self.candidates = 0

    def visit_Call(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg == "values" and isinstance(kw.value, ast.Dict):
                self._check_dict(kw.value)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._check_sql(_joinedstr_skeleton(node), node.lineno)
        # Do not generic_visit: the constant fragments below this node have
        # already been folded into the skeleton above, and visiting them
        # separately would both double-count candidates and lose the
        # cross-fragment SET...WHERE span.

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._check_sql(node.value, node.lineno)

    def _check_dict(self, node: ast.Dict) -> None:
        keys = {
            k.value
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        if "tombstoned_at" not in keys:
            return
        self.candidates += 1
        if "live" not in keys:
            self.violations.append((
                self.file,
                "values= dict sets tombstoned_at without live",
                node.lineno,
            ))

    def _check_sql(self, text: str, lineno: int) -> None:
        segment = _set_segment(text)
        if segment is None:
            return
        if not _TOMBSTONED_AT_ASSIGN_RE.search(segment):
            return
        self.candidates += 1
        if not _LIVE_ASSIGN_RE.search(segment):
            self.violations.append((
                self.file,
                "SQL SET clause sets tombstoned_at without live",
                lineno,
            ))


def test_tombstoned_at_write_derives_live_in_same_write():
    """A write that sets `tombstoned_at` must also set `live` in the same
    write -- a removed record left with `live = 1` stays reachable on the
    live-indexed direct recency rail even after this scan is green."""
    all_violations: list[tuple[Path, str, int]] = []
    total_candidates = 0
    for f in _all_iai_mcp_files():
        tree = ast.parse(f.read_text(), filename=str(f))
        visitor = _TombstoneLiveVisitor(f)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)
        total_candidates += visitor.candidates

    assert not all_violations, "a write sets tombstoned_at without live:\n" + "\n".join(
        f"  {path}:{lineno} -- {detail}" for path, detail, lineno in all_violations
    )
    assert total_candidates >= 6, (
        f"only found {total_candidates} tombstoned_at write candidates across "
        "src/iai_mcp/ -- expected at least 6; the scan is broken (renamed "
        "column, moved file, or a regex that stopped matching), not clean"
    )


def test_tombstoned_at_guard_scan_is_actually_populated():
    """The guard above fails silent (a scan resolving to zero files, or a
    regex that stopped matching, passes forever). Pin the scanned file list
    non-empty and confirm two known seams are actually in it."""
    scanned = _all_iai_mcp_files()
    assert scanned, "live/tombstoned_at guard scan resolved to an empty file list"
    assert (SRC / "migrate" / "_blob_quarantine.py") in scanned
    assert (SRC / "migrate" / "_dedupe.py") in scanned


# ---------------------------------------------------------------------------
# migrate staging seam guard: no `.add(...)` call in src/iai_mcp/migrate/ may
# be fed a row built by `store._to_row(...)`. `_to_row` is an insert
# serializer -- it drops storage-only columns (tombstoned_at, live,
# embedding_pending, valence) and regenerates vec_label. A full-table
# migrate swap must stage byte-for-byte from the SQL row instead.
# ---------------------------------------------------------------------------

MIGRATE_DIR = SRC / "migrate"


def _migrate_files() -> list[Path]:
    return sorted(MIGRATE_DIR.glob("*.py"))


class _ToRowIntoAddVisitor(ast.NodeVisitor):
    def __init__(self, file: Path) -> None:
        self.file = file
        self.violations: list[tuple[Path, int]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            for arg in node.args:
                candidates = arg.elts if isinstance(arg, ast.List) else [arg]
                for candidate in candidates:
                    if self._contains_to_row_call(candidate):
                        self.violations.append((self.file, node.lineno))
                        break
        self.generic_visit(node)

    def _contains_to_row_call(self, node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "_to_row"
            ):
                return True
        return False


def test_migrate_add_never_fed_by_to_row():
    """A full-table migrate swap (crypto recovery, re-embedding, etc.) must
    stage rows byte-for-byte from the source SQL row. `store._to_row(...)`
    is the fresh-insert serializer; feeding its output into a staging
    `.add(...)` drops tombstoned_at/live/embedding_pending/valence and
    regenerates vec_label."""
    all_violations: list[tuple[Path, int]] = []
    for f in _migrate_files():
        tree = ast.parse(f.read_text(), filename=str(f))
        visitor = _ToRowIntoAddVisitor(f)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)

    assert not all_violations, (
        "migrate staging seam violation: .add(...) fed by store._to_row(...):\n"
        + "\n".join(f"  {path}:{lineno}" for path, lineno in all_violations)
    )


def test_migrate_guard_scan_is_actually_populated():
    """Non-vacuity: a repath that empties the migrate scan must fail loud,
    not silently pass every migrate file forever."""
    scanned = _migrate_files()
    assert scanned, "migrate staging-seam guard scan resolved to an empty file list"
    for f in scanned:
        ast.parse(f.read_text(), filename=str(f))
    assert (MIGRATE_DIR / "_crypto_mig.py") in scanned


def test_to_row_into_add_visitor_flags_planted_violation():
    """Positive control: prove the visitor actually flags a planted
    violation, not just passes vacuously because no real file trips it."""
    source = "tbl.add([o._to_row(r)])\n"
    tree = ast.parse(source, filename="<synthetic>")
    visitor = _ToRowIntoAddVisitor(Path("<synthetic>"))
    visitor.visit(tree)
    assert len(visitor.violations) == 1, (
        f"expected exactly one violation for a planted .add([o._to_row(r)]) "
        f"call, got {visitor.violations}"
    )


# ---------------------------------------------------------------------------
# rich_club render-time label compaction (IAI_MCP_RICH_CLUB_COMPACT_LABEL):
# the derived aaak_index+age display label shrinks at the render site;
# generate_aaak_index and the stored aaak_index column stay untouched. Two-
# part lossless guard (byte-identity of stored fields + verbatim content-
# side) plus the token-reduction / saturated-identity gate the owner's
# decision requires.
# ---------------------------------------------------------------------------

_RICH_CLUB_ENV = "IAI_MCP_RICH_CLUB_COMPACT_LABEL"


def _seed_rich_club_record(
    store,
    i: int,
    *,
    entity_tags: "list[str] | None" = None,
    plain_tags: "list[str] | None" = None,
    pinned: bool = False,
    never_merge: bool = False,
    s5_trust_score: float = 0.5,
    detail_level: int = 3,
    created_at=None,
):
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    now = datetime.now(timezone.utc)
    ts = created_at if created_at is not None else now - timedelta(days=12)
    tags = list(plain_tags or []) + [f"entity:{e}" for e in (entity_tags or [])]
    # Fixed raw length so _clean_surface(...)[:60] is identical width across
    # every seeded record; the numeral (within the first 60 chars) makes each
    # record's content side unique so rendered lines can be traced back to
    # their source record.
    prefix = f"alice memory record {i:04d} "
    literal_surface = (prefix + "z" * 80)[:80]
    rid = uuid4()
    rec = MemoryRecord(
        id=rid,
        tier="semantic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=detail_level,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=never_merge,
        provenance=[],
        created_at=ts,
        updated_at=ts,
        tags=tags,
        language="en",
        s5_trust_score=s5_trust_score,
    )
    store.insert(rec)
    return rec


def _compose_standard(store, rich_club_ids, *, compact: bool, monkeypatch):
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.profile import default_state
    from iai_mcp.session import assemble_session_start

    monkeypatch.setenv(_RICH_CLUB_ENV, "1" if compact else "0")
    state = {**default_state(), "wake_depth": "standard"}
    return assemble_session_start(
        store, CommunityAssignment(), rich_club_ids,
        session_id="rich-club-lever-test", profile_state=state,
    )


def _rich_club_lines(markdown: str) -> list[str]:
    if "## Key memories" not in markdown:
        return []
    block = markdown.split("## Key memories", 1)[1]
    block = block.split("\n\n", 1)[0]
    return [ln for ln in block.strip("\n").splitlines() if ln.strip()]


def test_compact_aaak_label_matches_owner_chosen_format():
    """Direct fidelity check against the owner-chosen examples from the
    decision record: tier as one char, entities only when present (dropped
    'E:' prefix), room hash and tags dropped entirely. Built directly from
    record fields (tier, tags) -- never a generate_aaak_index round trip
    (ME-02: a tag/entity value containing '/' can otherwise mis-split into a
    spoofed field)."""
    from iai_mcp.session import _compact_aaak_label

    assert _compact_aaak_label("semantic", ["semantic", "cls_summary"]) == "S"
    assert (
        _compact_aaak_label(
            "episodic", ["capture", "entity:provenance_json", "entity:session_id"]
        )
        == "E ·provenance_json,session_id"
    )


def test_compact_aaak_label_caps_entity_list_with_marker():
    """Entity-heavy records (real-corpus provenance dumps can carry a dozen+
    tags) are capped at 4 displayed entities -- an uncapped list can grow
    long enough to hit the shared 88-char aaak truncation guard in BOTH
    label forms, at which point the compact form stops saving anything over
    the legacy one. A record with exactly 4 entities shows all 4 with no
    marker (nothing was actually dropped); a record with more than 4 is
    capped AND carries a trailing marker so a capped line is distinguishable
    from one that genuinely has only 4 entities."""
    from iai_mcp.session import _compact_aaak_label

    exactly_four = _compact_aaak_label(
        "episodic", ["capture"] + [f"entity:{c}" for c in "abcd"]
    )
    assert exactly_four == "E ·a,b,c,d"

    more_than_four = _compact_aaak_label(
        "episodic", ["capture"] + [f"entity:{c}" for c in "abcdef"]
    )
    assert more_than_four == "E ·a,b,c,d…"
    assert "e" not in more_than_four.split("·", 1)[1].rstrip("…").split(",")


def test_compact_aaak_label_caps_entity_string_by_characters():
    """ME-03: the entity cap bounds COUNT (4) but four long entity names can
    still produce a joined string long enough to erode the session-start
    budget margin. The joined string is additionally capped by CHARACTERS
    (with the truncation marker), so an entity-heavy line actually shrinks
    even when it has 4 or fewer entities."""
    from iai_mcp.session import _COMPACT_LABEL_ENTITY_CHAR_CAP, _compact_aaak_label

    long_entities = [f"entity:very_long_provenance_entity_name_{i:02d}" for i in range(4)]
    label = _compact_aaak_label("episodic", long_entities)
    body = label.split("·", 1)[1]
    assert body.endswith("…"), f"long entity string must carry the truncation marker: {body!r}"
    assert len(body.rstrip("…")) <= _COMPACT_LABEL_ENTITY_CHAR_CAP

    short_entities = ["entity:a", "entity:b"]
    short_label = _compact_aaak_label("episodic", short_entities)
    assert not short_label.endswith("…"), "a short entity string must not be marked truncated"


def test_rich_club_two_part_lossless_guard(tmp_path, monkeypatch):
    """Guard 1: seeded records' literal_surface/pinned/never_merge/trust/
    detail_level are byte-identical after composition (no write-back).
    Guard 2: every rendered '## Key memories' line's content side (after the
    first ': ') is the verbatim _clean_surface(literal_surface)[:60] for its
    record -- the lever must only ever touch the index side."""
    from iai_mcp.session import _clean_surface
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setattr("iai_mcp.capture.read_pending_live_events", lambda *a, **k: [])

    store = MemoryStore(path=tmp_path / "store")
    records = [
        _seed_rich_club_record(
            store, i,
            entity_tags=["provenance_json", "session_id"] if i == 3 else None,
            plain_tags=["semantic"],
            pinned=(i == 1),
            never_merge=(i == 2),
            s5_trust_score=0.91,
            detail_level=4,
        )
        for i in range(6)
    ]
    rich_club_ids = [r.id for r in records]
    before = {
        r.id: (r.literal_surface, r.pinned, r.never_merge, r.s5_trust_score, r.detail_level)
        for r in records
    }

    for compact in (False, True):
        payload = _compose_standard(store, rich_club_ids, compact=compact, monkeypatch=monkeypatch)
        assert payload.rich_club, "rich_club segment must be composed from the seeded ids"

        from iai_mcp.session import format_payload_as_markdown
        markdown = format_payload_as_markdown(payload)
        lines = _rich_club_lines(markdown)
        assert lines, "no rendered rich_club lines found under '## Key memories'"
        assert len(lines) <= len(records)

        # Position-based recovery: _rich_club_segment_with_budget iterates
        # rich_club_ids in order and its skip filter (record missing / empty
        # cleaned surface) is label-independent -- every seeded record here
        # survives that filter, so the Nth rendered line is always the Nth
        # seeded record. Never recover identity by matching content text
        # (repeated tags/content can collide across records).
        for idx, line in enumerate(lines):
            assert ": " in line, f"malformed rich_club line (no index/content separator): {line!r}"
            index_side, content_side = line.split(": ", 1)
            expected = _clean_surface(records[idx].literal_surface)[:60]
            assert content_side == expected, (
                f"line {idx} content side is not the verbatim source record's "
                f"literal_surface[:60]: got {content_side!r}, expected {expected!r}"
            )
            # The lever must only ever touch the index side: compact ON drops
            # R:/T: entirely, compact OFF (the kill-switch) keeps the legacy
            # verbose fields -- proves the toggle actually changes rendering.
            if compact:
                assert "R:" not in index_side and "T:" not in index_side, (
                    f"compact label ON must drop R:/T: from the index side: {index_side!r}"
                )
            else:
                assert "R:" in index_side and "T:" in index_side, (
                    f"compact label OFF (legacy) must retain R:/T: rendering: {index_side!r}"
                )

    for r in records:
        stored = store.get(r.id)
        assert (
            stored.literal_surface,
            stored.pinned,
            stored.never_merge,
            stored.s5_trust_score,
            stored.detail_level,
        ) == before[r.id], f"stored record {r.id} mutated by composition"


def test_rich_club_compact_label_token_reduction_and_saturated_identity(tmp_path, monkeypatch):
    """Token reduction: the same rich_club composition yields a strictly
    lower tiktoken cl100k_base count with the compact label ON (default)
    than OFF. Saturated-branch identity: HI-01's decoupled design selects
    by the legacy (verbose) line cost regardless of the label toggle, so
    the admitted record id-set is IDENTICAL OFF vs ON *by construction* --
    a structural regression guard, not a hand-fitted count."""
    import tiktoken

    from iai_mcp.session import format_payload_as_markdown
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setattr("iai_mcp.capture.read_pending_live_events", lambda *a, **k: [])

    store = MemoryStore(path=tmp_path / "store")
    # 80 candidates with VARIED per-line cost -- mimics the real corpus's
    # mix of plain and entity-heavy records (a provenance-tag dump, a long
    # variable-name list) so the budget boundary is exercised naturally
    # rather than fixture lengths hand-fitted to a specific budget value.
    records = []
    for i in range(80):
        entity_tags = (
            [f"provenance_entity_{i}_{j}" for j in range(8)]
            if i % 5 == 0
            else ["context_data"]
        )
        records.append(
            _seed_rich_club_record(
                store, i,
                plain_tags=["semantic", "cls_summary_pad1"],
                entity_tags=entity_tags,
            )
        )
    rich_club_ids = [r.id for r in records]

    payload_off = _compose_standard(store, rich_club_ids, compact=False, monkeypatch=monkeypatch)
    payload_on = _compose_standard(store, rich_club_ids, compact=True, monkeypatch=monkeypatch)

    md_off = format_payload_as_markdown(payload_off)
    md_on = format_payload_as_markdown(payload_on)

    lines_off = _rich_club_lines(md_off)
    lines_on = _rich_club_lines(md_on)
    assert lines_off, "OFF branch produced no rich_club lines"
    assert lines_on, "ON branch produced no rich_club lines"
    # Saturation witness: neither branch admitted every candidate.
    assert len(lines_off) < len(records)
    assert len(lines_on) < len(records)

    # Position-based recovery (never content-text match -- see
    # test_rich_club_two_part_lossless_guard for the rationale): every
    # seeded record here survives the render skip filter, so the first
    # len(lines) rich_club_ids, in order, are exactly the admitted set.
    ids_off = rich_club_ids[: len(lines_off)]
    ids_on = rich_club_ids[: len(lines_on)]
    assert ids_off == ids_on, (
        "admitted-id-set is not IDENTICAL OFF vs ON -- selection must be "
        "keyed to the legacy line cost regardless of the label toggle "
        f"(off={len(ids_off)}, on={len(ids_on)})"
    )

    encoder = tiktoken.get_encoding("cl100k_base")
    tok_off = len(encoder.encode(payload_off.rich_club))
    tok_on = len(encoder.encode(payload_on.rich_club))
    assert tok_on < tok_off, (
        f"compact label ON must strictly reduce the rich_club tiktoken count: "
        f"off={tok_off} on={tok_on}"
    )

    # Format spot-check: OFF keeps the legacy verbose fields, ON drops them.
    assert "T:" in md_off and "R:" in md_off
    assert "T:" not in md_on and "R:" not in md_on


def test_deep_branch_admitted_set_identical_regardless_of_label_toggle(tmp_path, monkeypatch):
    """HI-01 decouples selection from rendering: the deep-branch budget is a
    single legacy-cost constant that does not depend on the label toggle,
    so there is no OFF/ON ratio left to lock (ME-04 dropped the 1e-9 ratio
    lock, which blocked future recalibration). Confirms the SAME budget
    value is used on both toggle states and the admitted record id-set is
    identical, mirroring the standard-branch structural identity guard."""
    import iai_mcp.session as session_mod
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.profile import default_state
    from iai_mcp.session import format_payload_as_markdown
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setattr("iai_mcp.capture.read_pending_live_events", lambda *a, **k: [])

    store = MemoryStore(path=tmp_path / "store")
    records = []
    for i in range(80):
        entity_tags = (
            [f"provenance_entity_{i}_{j}" for j in range(6)] if i % 5 == 0 else ["context_data"]
        )
        records.append(
            _seed_rich_club_record(
                store, i,
                plain_tags=["semantic", "cls_summary_pad1"],
                entity_tags=entity_tags,
            )
        )
    rich_club_ids = [r.id for r in records]

    seen_budgets: list[int] = []
    real = session_mod._rich_club_segment_with_budget

    def _spy(store, rich_club, *, budget):
        seen_budgets.append(budget)
        return real(store, rich_club, budget=budget)

    monkeypatch.setattr(session_mod, "_rich_club_segment_with_budget", _spy)

    def _run(compact: bool):
        seen_budgets.clear()
        monkeypatch.setenv(_RICH_CLUB_ENV, "1" if compact else "0")
        state = {**default_state(), "wake_depth": "deep"}
        payload = session_mod._compose_session_start_payload(
            store, CommunityAssignment(), rich_club_ids,
            session_id="deep-budget-identity", profile_state=state,
        )
        return seen_budgets[-1], payload

    legacy_budget, payload_off = _run(False)
    compact_budget, payload_on = _run(True)

    assert legacy_budget == compact_budget == session_mod._RICH_CLUB_DEEP_BUDGET_TOKENS, (
        "deep-branch selection must use ONE legacy-cost budget regardless "
        f"of the label toggle: off={legacy_budget} on={compact_budget}"
    )

    md_off = format_payload_as_markdown(payload_off)
    md_on = format_payload_as_markdown(payload_on)
    lines_off = _rich_club_lines(md_off)
    lines_on = _rich_club_lines(md_on)
    assert lines_off, "OFF branch produced no deep rich_club lines"
    assert lines_on, "ON branch produced no deep rich_club lines"
    assert len(lines_off) < len(records), "fixture must genuinely saturate the deep budget"

    ids_off = rich_club_ids[: len(lines_off)]
    ids_on = rich_club_ids[: len(lines_on)]
    assert ids_off == ids_on, (
        "deep-branch admitted-id-set is not IDENTICAL OFF vs ON "
        f"(off={len(ids_off)}, on={len(ids_on)})"
    )


# ---------------------------------------------------------------------------
# Generational cache invariants: no independent counter beyond the
# allow-listed pair, and no live-graph mutation outside graph.py. A future
# edit that adds a mutator or drops a bump line must fail a test here, not
# silently serve a stale corpus.
# ---------------------------------------------------------------------------

_ALLOWED_COUNTER_ATTRS = frozenset({"_pool_content_version", "_dirty_since_centrality"})

# The incremental per-node/edge mutators -- the methods a hypothetical new
# "generational counter" would actually be added alongside. clear_and_rebuild
# is a bulk reset, not an incremental mutator: it legitimately also resets
# `_centrality_resolved`, an older, unrelated centrality-cache flag that
# predates this generational cache and is intentionally out of scope here.
_GRAPH_COUNTER_MUTATOR_METHODS = frozenset({
    "add_node", "set_node_payload", "remove_node", "add_edge",
})


def _is_self_attr(node: ast.AST) -> "ast.Attribute | None":
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node
    return None


class _VersionCounterVisitor(ast.NodeVisitor):
    """Collects every `self.X += 1` / `self.X = True|False` target found
    inside the named incremental-mutator method bodies. Method-scoped (not
    whole-file) on purpose: `centrality()` and `clear_and_rebuild` touch
    `_dirty_since_centrality` / `_centrality_resolved` for the pre-existing,
    unrelated betweenness-centrality cache, which must not trip this guard.
    """

    def __init__(self) -> None:
        self.found: set[str] = set()
        self._in_scope = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        was_in_scope = self._in_scope
        if node.name in _GRAPH_COUNTER_MUTATOR_METHODS:
            self._in_scope = True
        self.generic_visit(node)
        self._in_scope = was_in_scope

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if (
            self._in_scope
            and isinstance(node.op, ast.Add)
            and _is_self_attr(node.target)
            and isinstance(node.value, ast.Constant)
            and node.value.value == 1
            and not isinstance(node.value.value, bool)
        ):
            self.found.add(node.target.attr)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if (
            self._in_scope
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, bool)
        ):
            for target in node.targets:
                attr_node = _is_self_attr(target)
                if attr_node is not None:
                    self.found.add(attr_node.attr)
        self.generic_visit(node)


def test_no_new_version_counter_or_dirty_flag_in_graph_mutators():
    """No new independent version counter (`self.X += 1`) or boolean
    dirty-flag (`self.X = True/False`) may be introduced inside graph.py's
    incremental mutators beyond the allow-listed pair -- a third counter
    desyncs a generational cache keyed on `_pool_content_version` from the
    mutation history it is meant to mirror."""
    tree = ast.parse((SRC / "graph.py").read_text(), filename="graph.py")
    visitor = _VersionCounterVisitor()
    visitor.visit(tree)
    assert visitor.found == _ALLOWED_COUNTER_ATTRS, (
        f"graph.py mutators' incremented/dirty-flag attribute set is "
        f"{sorted(visitor.found)}, expected exactly "
        f"{sorted(_ALLOWED_COUNTER_ATTRS)} -- either a new counter was "
        "added, or the scan pattern stopped matching one of the "
        "allow-listed pair (both are failures, not passes)"
    )


def test_version_counter_visitor_flags_planted_third_counter():
    """Positive control: prove the visitor actually flags a planted third
    counter, not just passes vacuously because no real file trips it."""
    source = (
        "class MemoryGraph:\n"
        "    def add_node(self, node_id, community_id, embedding):\n"
        "        self._pool_content_version += 1\n"
        "        self._dirty_since_centrality = True\n"
        "        self._shadow_generation += 1\n"
    )
    tree = ast.parse(source, filename="<synthetic>")
    visitor = _VersionCounterVisitor()
    visitor.visit(tree)
    assert visitor.found == _ALLOWED_COUNTER_ATTRS | {"_shadow_generation"}, (
        f"expected the planted third counter to be flagged, got {sorted(visitor.found)}"
    )
    assert visitor.found != _ALLOWED_COUNTER_ATTRS, (
        "the planted-third-counter snippet must NOT pass the allow-list "
        "exact-match check -- the guard would be vacuous otherwise"
    )


_LIVE_GRAPH_ATTRS = frozenset({"_node_payload", "_adj"})
_LIVE_GRAPH_MUTATOR_METHODS = frozenset({"pop", "clear", "setdefault", "update"})


def _live_graph_root_attr(node: ast.AST) -> str | None:
    """Walk a (possibly nested, e.g. `self._adj[u][v]`) subscript/attribute
    chain down to its root and return the attribute name if it is one of
    the live-graph containers. A bare local name (e.g. the unrelated
    `encoded_node_payload` variable in runtime_graph_cache.py) is never an
    `ast.Attribute`, so it never matches."""
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute) and node.attr in _LIVE_GRAPH_ATTRS:
        return node.attr
    return None


class _LiveGraphMutationVisitor(ast.NodeVisitor):
    def __init__(self, file: Path) -> None:
        self.file = file
        self.violations: list[tuple[Path, str, int]] = []
        self.candidates = 0

    def _record(self, attr: str, detail: str, lineno: int) -> None:
        self.candidates += 1
        if self.file.name != "graph.py":
            self.violations.append((self.file, detail, lineno))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                attr = _live_graph_root_attr(target)
                if attr is not None:
                    self._record(
                        attr, f"assignment to .{attr}[...] outside graph.py", node.lineno,
                    )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            attr = _live_graph_root_attr(node.target)
            if attr is not None:
                self._record(
                    attr, f"augmented assignment to .{attr}[...] outside graph.py", node.lineno,
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _LIVE_GRAPH_MUTATOR_METHODS:
            attr = _live_graph_root_attr(func.value)
            if attr is not None:
                self._record(
                    attr,
                    f".{attr}.{func.attr}(...) call outside graph.py",
                    node.lineno,
                )
        self.generic_visit(node)


def test_live_graph_mutation_only_in_graph_py():
    """OQ1 machine-enforced: no src/iai_mcp module other than graph.py may
    write into the live graph's node-payload or adjacency containers
    directly -- every mutation must route through graph.py's own
    bump-carrying mutators, or a generational cache keyed on
    `_pool_content_version` can silently miss a real content change."""
    all_violations: list[tuple[Path, str, int]] = []
    graph_py_candidates = 0
    for f in _all_iai_mcp_files():
        tree = ast.parse(f.read_text(), filename=str(f))
        visitor = _LiveGraphMutationVisitor(f)
        visitor.visit(tree)
        all_violations.extend(visitor.violations)
        if f.name == "graph.py":
            graph_py_candidates += visitor.candidates

    assert not all_violations, (
        "live graph mutation outside graph.py:\n"
        + "\n".join(f"  {path}:{lineno} -- {detail}" for path, detail, lineno in all_violations)
    )
    assert graph_py_candidates >= 5, (
        f"only found {graph_py_candidates} in-graph.py _node_payload/_adj "
        "mutation sites -- expected at least 5; the scan is broken "
        "(renamed attribute, moved file, or a pattern that stopped "
        "matching), not clean"
    )


def test_live_graph_mutation_guard_scan_is_actually_populated():
    """The guard above fails silent (a scan resolving to zero files passes
    forever). Pin the scanned file list non-empty and confirm graph.py --
    the one file this guard exempts from violations -- is actually in it."""
    scanned = _all_iai_mcp_files()
    assert scanned, "live-graph-mutation guard scan resolved to an empty file list"
    assert (SRC / "graph.py") in scanned


def test_live_graph_mutation_visitor_flags_planted_violation():
    """Positive control: prove the visitor actually flags a planted
    out-of-graph.py `._node_payload[...] =` write, not just passes
    vacuously because no real file trips it."""
    source = (
        "def leaky_sync(graph, node_id, payload):\n"
        "    graph._node_payload[str(node_id)] = payload\n"
    )
    tree = ast.parse(source, filename="<synthetic>")
    visitor = _LiveGraphMutationVisitor(Path("not_graph.py"))
    visitor.visit(tree)
    assert len(visitor.violations) == 1, (
        f"expected exactly one violation for a planted out-of-graph.py "
        f"_node_payload write, got {visitor.violations}"
    )

    # Control: the identical snippet, scanned as if it WERE graph.py, must
    # not be flagged -- the file-name exemption only ever applies to the
    # one real graph.py.
    graph_py_visitor = _LiveGraphMutationVisitor(Path("graph.py"))
    graph_py_visitor.visit(tree)
    assert not graph_py_visitor.violations, (
        "the same write inside graph.py must not be a violation"
    )
    assert graph_py_visitor.candidates == 1
