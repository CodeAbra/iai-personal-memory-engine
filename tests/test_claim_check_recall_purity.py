"""Structural guard: claim_check and the assistant-tail counter-evidence
lane stay off the awake recall path. Anticipation belongs on capture
(inside memory_capture's refresh_pack); a memory_recall dispatch --
including claim_check's own internal recall -- must never run it."""
from __future__ import annotations

import ast
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp import core
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord

_SRC = Path(__file__).parent.parent / "src" / "iai_mcp"

_RECALL_DISPATCH_MODULES = [
    "retrieve.py",
    "pipeline.py",
    "semantic_recall.py",
    "core/__init__.py",
    "store/_store.py",
]

_CLAIM_CHECK_BANNED_PREFIXES = (
    "iai_mcp.foresight",
    "iai_mcp.store",
    "iai_mcp.embed",
    "iai_mcp.capture",
)


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _is_banned(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in _CLAIM_CHECK_BANNED_PREFIXES
    )


def test_recall_dispatch_modules_never_import_foresight():
    """Restates the existing recall-path-purity import scan locally so a
    regression introduced by this phase's capture-path work is caught here
    too, not only in test_recall_path_purity_foresight.py."""
    pattern = re.compile(
        r"^\s*(from|import)\s+iai_mcp\.foresight\b|"
        r"^\s*from\s+iai_mcp\s+import\s+.*\bforesight\b",
        re.M,
    )
    offenders = []
    for rel in _RECALL_DISPATCH_MODULES:
        text = (_SRC / rel).read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(rel)
    assert not offenders, (
        f"recall-dispatch modules import foresight: {offenders} -- "
        f"anticipation must never sit on the awake recall path"
    )


def test_claim_check_module_is_pure():
    tree = ast.parse((_SRC / "claim_check.py").read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_banned(alias.name):
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _is_banned(node.module):
                offenders.append(node.module)
    assert not offenders, (
        f"claim_check.py (verdict synthesis) imports from a forbidden "
        f"module: {offenders} -- its verdict logic must stay pure, reading "
        f"only the JSON fields a memory_recall response already carries"
    )


def _seed_probe_record(store) -> None:
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="alice reference content for the recall-purity probe",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "seed", "role": "user"}],
        created_at=now,
        updated_at=now,
        tags=["role:user"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
    )
    store.insert(rec)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_assistant_tail_lane_never_fires_on_recall_dispatch(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))

    import iai_mcp.capture as capture_mod

    real_read = capture_mod.read_pending_live_events
    calls: list[str] = []

    def _spy(*args, **kwargs):
        calls.append("called")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(capture_mod, "read_pending_live_events", _spy)

    store_path = tmp_path / f"driver-{driver}-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_path)
    try:
        _seed_probe_record(store)

        recall_resp = core.dispatch(store, "memory_recall", {
            "cue": "reference content",
            "session_id": f"sess-tail-recall-noop-{driver}",
            "cue_embedding": [0.1] * EMBED_DIM,
        })
        assert "error" not in recall_resp, (
            f"[driver={driver}] memory_recall dispatch errored: {recall_resp.get('error')}"
        )

        claim_resp = core.dispatch(store, "claim_check", {
            "cue": "reference content",
            "session_id": f"sess-tail-claim-noop-{driver}",
        })
        assert "error" not in claim_resp, (
            f"[driver={driver}] claim_check dispatch errored: {claim_resp.get('error')}"
        )
        assert "verdict" in claim_resp
    finally:
        store.close()

    assert not calls, (
        f"[driver={driver}] the assistant-tail lane fired during a recall "
        "dispatch -- read_pending_live_events lives only inside "
        "refresh_pack on the capture path, never on memory_recall or "
        "claim_check's internal recall"
    )


# ---------------------------------------------------------------------------
# Execution-time spy: enumerates claim_check's unguarded side effects by
# actually running a dispatch, not by reading the handler. The two negative
# dispatches below (L0, common) are the load-bearing proof that a suppression
# gate is both present and effective on a real claim_check call; the
# byte-identity assertion elsewhere in this file proves such a gate is safe
# for a normal caller. Neither proves the other.
# ---------------------------------------------------------------------------

_L0_SENTINEL_UUID = UUID("00000000-0000-0000-0000-000000000001")

# Policy set, not spy-discovered: these fire only on a genuine fault
# (rank deficiency / role saturation / codec-marker missing / embed /
# centrality) and must survive suppression -- a probe that trips a real
# fault must still emit-then-raise. Inlined locally so a drift between
# this list and iai_mcp.events._FAIL_LOUD_KINDS is caught by
# test_fail_loud_exemption_set_matches_policy rather than silently diverging.
_EXPECTED_FAIL_LOUD = frozenset({
    "rank_deficiency_warning",
    "role_saturation_warning",
    "codec_marker_missing",
    "embed_native_failure",
    "recall_centrality_failed",
})


def _seed_sentinel_record(store) -> None:
    """L0 identity record: gate.should_skip_retrieval trips on any cue under
    3 chars and pipeline._recall_core serves this record directly by id,
    with no similarity search -- so a dummy embedding is fine here."""
    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=_L0_SENTINEL_UUID,
        tier="episodic",
        literal_surface="L0 identity sentinel for the recall-purity probe",
        aaak_index="",
        embedding=[0.0] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[{"session_id": "seed", "role": "system"}],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
    )
    store.insert(rec)
    flush_record_buffer(store)


def _seed_multiple_probes(store, session_id: str) -> list[UUID]:
    """Three real-embedded, community-assigned records sharing the cue
    vocabulary at different specificity -- unlike a single-record fixture,
    this makes hit ORDER (not just membership) a discriminating signal."""
    from iai_mcp.capture import capture_turn

    texts = (
        "alice reference content for the recall-purity probe",
        "alice mentions a reference document with related content notes",
        "alice discusses content strategy without much reference material",
    )
    ids: list[UUID] = []
    for i, text in enumerate(texts):
        seed = capture_turn(
            store, cue="c", text=text,
            session_id=f"{session_id}-{i}", role="user", live_turn=True,
        )
        ids.append(UUID(seed["record_id"]))
    flush_record_buffer(store)
    return ids


def _seed_indexable_probe(store, session_id: str) -> UUID:
    """A real-embedded, community-assigned probe. Unlike _seed_probe_record's
    dummy vector, this one surfaces as a ranked hit under server-side cue
    embedding -- claim_check's recursive dispatch never forwards a caller
    cue_embedding, so the probe must be indexable on its own terms."""
    from iai_mcp.capture import capture_turn

    seed = capture_turn(
        store, cue="c",
        text="alice reference content for the recall-purity probe",
        session_id=session_id, role="user", live_turn=True,
    )
    flush_record_buffer(store)
    return UUID(seed["record_id"])


def _install_side_effect_spy(store, monkeypatch) -> dict[str, list]:
    """Wraps every store-mutation and telemetry surface reachable from a
    claim_check dispatch and records what actually fires -- a key appears
    in the returned dict only if that surface was actually invoked."""
    calls: dict[str, list] = {}

    def _bind_store_method(name: str) -> None:
        if not hasattr(store, name):
            return
        original = getattr(store, name)

        def _wrapped(*args, __original=original, __name=name, **kwargs):
            calls.setdefault(f"store.{__name}", []).append(None)
            return __original(*args, **kwargs)

        monkeypatch.setattr(store, name, _wrapped, raising=True)

    for name in (
        "queue_reinforce", "queue_coactivation", "queue_profile_modulate",
        "boost_edges", "append_provenance", "merge_insert", "add",
        "queue_provenance_batch",
    ):
        _bind_store_method(name)

    def _bind_module_function(module, name: str, label: str) -> None:
        if not hasattr(module, name):
            return
        original = getattr(module, name)

        def _wrapped(*args, __original=original, __label=label, **kwargs):
            calls.setdefault(__label, []).append(None)
            return __original(*args, **kwargs)

        monkeypatch.setattr(module, name, _wrapped, raising=True)

    import iai_mcp.daemon_state as daemon_state_mod
    import iai_mcp.provenance_buffer as provenance_buffer_mod
    import iai_mcp.retrieve as retrieve_mod

    _bind_module_function(
        retrieve_mod, "potentiate_coactivation", "retrieve.potentiate_coactivation",
    )
    _bind_module_function(
        daemon_state_mod, "consume_first_turn", "daemon_state.consume_first_turn",
    )
    _bind_module_function(
        daemon_state_mod, "get_pending_digest", "daemon_state.get_pending_digest",
    )
    _bind_module_function(
        daemon_state_mod, "update_state", "daemon_state.update_state",
    )
    # Deferred write-behind: appends to a sidecar jsonl file under
    # store.root, later flushed into store.append_provenance_batch --
    # invisible to a table row-count backstop. Found by execution, not by
    # reading any prior guard list.
    _bind_module_function(
        provenance_buffer_mod, "defer_provenance", "provenance_buffer.defer_provenance",
    )

    # write_event is imported BY VALUE at the top of several modules, so
    # patching the events.py source alone misses those pre-bound callers --
    # every module on the claim_check dispatch path needs its own patch.
    import iai_mcp.events as events_mod
    import iai_mcp.pipeline as pipeline_mod

    for module in (events_mod, pipeline_mod, retrieve_mod):
        if not hasattr(module, "write_event"):
            continue
        original = module.write_event

        def _wrapped_write_event(store_arg, kind, data, *args, __original=original, **kwargs):
            # The suppression choke point lives INSIDE write_event and still
            # returns a value on a no-op call -- observing the call itself
            # would flag a correctly-suppressed call as an offender. What
            # matters is whether the write actually persisted: buffered
            # kinds land in _event_buffer, unbuffered kinds land straight
            # in the events table.
            buf_before = len(events_mod._event_buffer.get(id(store_arg), []))
            try:
                table_before = len(store_arg.db.open_table("events").to_pandas())
            except Exception:  # noqa: BLE001 -- table read is a diagnostic aid only
                table_before = None
            result = __original(store_arg, kind, data, *args, **kwargs)
            buf_after = len(events_mod._event_buffer.get(id(store_arg), []))
            persisted = buf_after > buf_before
            if not persisted and table_before is not None:
                try:
                    table_after = len(store_arg.db.open_table("events").to_pandas())
                    persisted = table_after > table_before
                except Exception:  # noqa: BLE001 -- table read is a diagnostic aid only
                    pass
            if persisted:
                path = data.get("path") if isinstance(data, dict) else None
                calls.setdefault(f"telemetry:{kind}", []).append(path)
            return result

        monkeypatch.setattr(module, "write_event", _wrapped_write_event, raising=True)

    return calls


def _offenders(calls: dict[str, list]) -> dict[str, list]:
    """The non-exempt subset of everything the spy actually saw fire."""
    return {
        key: values for key, values in calls.items()
        if not (key.startswith("telemetry:") and key.split(":", 1)[1] in _EXPECTED_FAIL_LOUD)
    }


def _row_count_snapshot(store) -> dict[str, int]:
    import iai_mcp.events as events_mod

    tables = ("records", "edges", "events", "record_tags", "budget_ledger", "ratelimit_ledger")
    snapshot = {t: len(store.db.open_table(t).to_pandas()) for t in tables}
    # Buffered telemetry never touches the events TABLE until an explicit
    # flush -- a row-count backstop alone is blind to it (the exact species
    # the deferred-provenance file write also belongs to).
    snapshot["_event_buffer"] = len(events_mod._event_buffer.get(id(store), []))
    return snapshot


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_claim_check_spy_l0_negative_offender_set_empty(driver, tmp_path, monkeypatch):
    """L0 negative: a sub-3-char cue against a seeded sentinel record must
    trigger ZERO non-exempt side effects on a claim_check dispatch. The
    returned hit identity (the sentinel record) is the witness that this
    dispatch actually entered the L0 fast-path -- a witness that survives
    suppression, unlike retrieval_used/append_provenance which are exactly
    the surface under test here."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    try:
        _seed_sentinel_record(store)
        before = _row_count_snapshot(store)
        calls = _install_side_effect_spy(store, monkeypatch)

        resp = core.dispatch(store, "claim_check", {
            "cue": "hi", "session_id": f"sess-l0-neg-{driver}",
        })

        after = _row_count_snapshot(store)
        offenders = _offenders(calls)
        hits = resp.get("hits", [])
        assert hits and hits[0]["record_id"] == str(_L0_SENTINEL_UUID), (
            f"[driver={driver}] L0 fast-path not entered: hits={hits}"
        )
        assert not offenders, (
            f"[driver={driver}] claim_check L0 dispatch triggered side "
            f"effects: {offenders}"
        )
        assert before == after, (
            f"[driver={driver}] claim_check L0 dispatch changed table row "
            f"counts: before={before} after={after}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_claim_check_spy_common_negative_offender_set_empty(driver, tmp_path, monkeypatch):
    """Common negative: a >=3-char cue against a seeded, indexable probe must
    also trigger ZERO non-exempt side effects. Witnessed by the returned hit
    being the ranked probe record, not the sentinel -- proving the common
    path (not L0) actually ran."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    try:
        probe_id = _seed_indexable_probe(store, session_id=f"seed-{driver}")
        before_prov = len(store.get(probe_id).provenance or [])
        before_rows = _row_count_snapshot(store)
        calls = _install_side_effect_spy(store, monkeypatch)

        resp = core.dispatch(store, "claim_check", {
            "cue": "reference content", "session_id": f"sess-common-neg-{driver}",
        })

        after_rows = _row_count_snapshot(store)
        after_prov = len(store.get(probe_id).provenance or [])
        offenders = _offenders(calls)
        hits = resp.get("hits", [])
        assert (
            hits
            and hits[0]["record_id"] == str(probe_id)
            and hits[0]["record_id"] != str(_L0_SENTINEL_UUID)
        ), f"[driver={driver}] common path not entered on the ranked probe: hits={hits}"
        assert not offenders, (
            f"[driver={driver}] claim_check common dispatch triggered side "
            f"effects: {offenders}"
        )
        assert before_rows == after_rows, (
            f"[driver={driver}] claim_check common dispatch changed table "
            f"row counts: before={before_rows} after={after_rows}"
        )
        assert before_prov == after_prov, (
            f"[driver={driver}] claim_check common dispatch grew the "
            f"probe's provenance: before={before_prov} after={after_prov}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_claim_check_spy_zero_count_fastpath_negative_offender_set_empty(driver, tmp_path, monkeypatch):
    """Zero-count fast path: `retrieve.recall()` is a second, complete recall
    implementation dispatched unconditionally whenever the count-cache reads
    zero. Forcing that same count-cache read to 0 against a store that
    actually holds an indexable record proves this path is structurally
    incapable of leaking a store write: `query_similar` gates on the
    identical count-cache read, so a forced-zero count always yields empty
    hits here too, and an empty hit list can never populate the pending
    provenance batch that this dispatch would otherwise queue. Verified by
    running this path (not by reading it) -- zero side effects, zero hits,
    same forced-zero count on both call sites."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    try:
        probe_id = _seed_indexable_probe(store, session_id=f"seed-{driver}")
        before_prov = len(store.get(probe_id).provenance or [])
        before_rows = _row_count_snapshot(store)
        monkeypatch.setattr(store, "active_records_count", lambda: 0)
        calls = _install_side_effect_spy(store, monkeypatch)

        resp = core.dispatch(store, "claim_check", {
            "cue": "reference content", "session_id": f"sess-zerocount-neg-{driver}",
        })

        after_rows = _row_count_snapshot(store)
        after_prov = len(store.get(probe_id).provenance or [])
        offenders = _offenders(calls)
        assert "error" not in resp, (
            f"[driver={driver}] claim_check dispatch errored on the "
            f"zero-count fast path: {resp.get('error')}"
        )
        hits = resp.get("hits", [])
        assert hits == [], (
            f"[driver={driver}] zero-count fast path unexpectedly surfaced "
            f"a hit despite the forced-zero count-cache read: hits={hits}"
        )
        assert not offenders, (
            f"[driver={driver}] claim_check zero-count fast-path dispatch "
            f"triggered side effects: {offenders}"
        )
        assert before_rows == after_rows, (
            f"[driver={driver}] claim_check zero-count fast-path dispatch "
            f"changed table row counts: before={before_rows} after={after_rows}"
        )
        assert before_prov == after_prov, (
            f"[driver={driver}] claim_check zero-count fast-path dispatch "
            f"grew the probe's provenance: before={before_prov} after={after_prov}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_claim_check_spy_fallback_negative_offender_set_empty(driver, tmp_path, monkeypatch):
    """Degraded fallback: a forced primary-pipeline failure routes the
    handler through `retrieve.recall()` (the same second recall
    implementation as the zero-count fast path) instead of the ranked
    common path -- must still trigger ZERO non-exempt side effects on a
    claim_check dispatch."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    try:
        probe_id = _seed_indexable_probe(store, session_id=f"seed-{driver}")
        before_prov = len(store.get(probe_id).provenance or [])
        before_rows = _row_count_snapshot(store)

        import iai_mcp.pipeline as pipeline_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("forced failure to exercise the fallback dispatch")

        monkeypatch.setattr(pipeline_mod, "recall_for_response", _boom)
        calls = _install_side_effect_spy(store, monkeypatch)

        resp = core.dispatch(store, "claim_check", {
            "cue": "reference content", "session_id": f"sess-fallback-neg-{driver}",
        })

        after_rows = _row_count_snapshot(store)
        after_prov = len(store.get(probe_id).provenance or [])
        offenders = _offenders(calls)
        hits = resp.get("hits", [])
        assert hits and hits[0]["record_id"] == str(probe_id), (
            f"[driver={driver}] fallback path not entered on the ranked "
            f"probe: hits={hits}"
        )
        assert not offenders, (
            f"[driver={driver}] claim_check fallback dispatch triggered "
            f"side effects: {offenders}"
        )
        assert before_rows == after_rows, (
            f"[driver={driver}] claim_check fallback dispatch changed "
            f"table row counts: before={before_rows} after={after_rows}"
        )
        assert before_prov == after_prov, (
            f"[driver={driver}] claim_check fallback dispatch grew the "
            f"probe's provenance: before={before_prov} after={after_prov}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_claim_check_spy_l0_positive_control_fires(driver, tmp_path, monkeypatch):
    """Non-vacuity: a direct memory_recall (not claim_check) over the same
    L0-triggering cue DOES fire the surfaces claim_check suppresses --
    proving the guard is claim_check-scoped, never a blanket suppression."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    try:
        _seed_sentinel_record(store)
        calls = _install_side_effect_spy(store, monkeypatch)

        resp = core.dispatch(store, "memory_recall", {
            "cue": "hi", "session_id": f"sess-l0-pos-{driver}",
        })

        assert "error" not in resp, resp.get("error")
        assert calls, f"[driver={driver}] direct memory_recall fired nothing"
        assert "store.append_provenance" in calls, (
            f"[driver={driver}] L0 append_provenance did not fire on a "
            f"direct memory_recall: {calls}"
        )
        l0_paths = calls.get("telemetry:retrieval_used", [])
        assert "recall_core_l0_fastpath" in l0_paths, (
            f"[driver={driver}] L0 retrieval_used telemetry absent: {calls}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_claim_check_spy_common_positive_control_fires(driver, tmp_path, monkeypatch):
    """Non-vacuity for the common path: a direct memory_recall fires at
    least one of the common-path-exclusive telemetry kinds."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    try:
        _seed_indexable_probe(store, session_id=f"seed-{driver}")
        calls = _install_side_effect_spy(store, monkeypatch)

        resp = core.dispatch(store, "memory_recall", {
            "cue": "reference content", "session_id": f"sess-common-pos-{driver}",
        })

        assert "error" not in resp, resp.get("error")
        assert calls, f"[driver={driver}] direct memory_recall fired nothing"
        common_witnesses = (
            "telemetry:recall_timing",
            "telemetry:retrieval_arousal_ab",
            "telemetry:deferred_curiosity_input",
        )
        assert any(w in calls for w in common_witnesses), (
            f"[driver={driver}] no common-path telemetry witness fired: {calls}"
        )
    finally:
        store.close()


def test_fail_loud_exemption_set_matches_policy():
    """Locks the policy exemption set against silent drift -- RED until the
    fix defines iai_mcp.events._FAIL_LOUD_KINDS, GREEN after."""
    import iai_mcp.events as events_mod

    actual = getattr(events_mod, "_FAIL_LOUD_KINDS", None)
    assert actual == _EXPECTED_FAIL_LOUD, (
        f"events._FAIL_LOUD_KINDS drifted from policy: {actual} != {_EXPECTED_FAIL_LOUD}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fail_loud_kinds_persist_under_suppression(driver, tmp_path, monkeypatch):
    """The exemption set locks a NAME, not a behavior, unless something
    proves the exempt kinds actually survive the choke point -- a probe
    that hits a real fail-loud fault must still emit-then-raise even
    while claim_check's recursion guard is active."""
    _select_driver(driver, monkeypatch)
    import iai_mcp.events as events_mod
    import iai_mcp.recall_suppression as rs

    store = MemoryStore(path=tmp_path)
    try:
        for kind in sorted(_EXPECTED_FAIL_LOUD):
            before = _row_count_snapshot(store)
            token = rs.recall_suppressed.set(True)
            try:
                events_mod.write_event(store, kind, {"probe": True}, severity="error")
            finally:
                rs.recall_suppressed.reset(token)
            after = _row_count_snapshot(store)
            persisted = (
                after["events"] > before["events"]
                or after["_event_buffer"] > before["_event_buffer"]
            )
            assert persisted, (
                f"[driver={driver}] fail-loud kind {kind!r} was swallowed "
                f"under suppression: before={before} after={after}"
            )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_normal_recall_byte_identical_under_suppression_change(driver, tmp_path, monkeypatch):
    """The contextvar defaults False, so an ordinary caller is unaffected by
    construction -- prove it empirically on ONE store, suppressed dispatch
    first (already proven zero-persisted-effect by the negative tests
    above, so the store is untouched going into the second call) then an
    ordinary dispatch second: forcing suppression on must not change hit or
    anti_hit identity/order, only whether the gated side effects persist.
    Three records at varying relevance (not one) so hit ORDER, not just
    membership, is a real discriminating signal."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    try:
        _seed_multiple_probes(store, session_id=f"seed-byteid-{driver}")
        before_rows = _row_count_snapshot(store)

        import iai_mcp.recall_suppression as rs
        token = rs.recall_suppressed.set(True)
        try:
            suppressed = core.dispatch(store, "memory_recall", {
                "cue": "reference content", "session_id": f"sess-byteid-suppressed-{driver}",
            })
        finally:
            rs.recall_suppressed.reset(token)

        after_suppressed_rows = _row_count_snapshot(store)
        assert before_rows == after_suppressed_rows, (
            f"[driver={driver}] forced suppression still mutated table row "
            f"counts: before={before_rows} after={after_suppressed_rows}"
        )

        baseline = core.dispatch(store, "memory_recall", {
            "cue": "reference content", "session_id": f"sess-byteid-normal-{driver}",
        })

        assert "error" not in suppressed, suppressed.get("error")
        assert "error" not in baseline, baseline.get("error")
        baseline_ids = [h["record_id"] for h in baseline.get("hits", [])]
        suppressed_ids = [h["record_id"] for h in suppressed.get("hits", [])]
        baseline_anti_ids = [h["record_id"] for h in baseline.get("anti_hits", [])]
        suppressed_anti_ids = [h["record_id"] for h in suppressed.get("anti_hits", [])]
        assert len(baseline_ids) >= 2, (
            f"[driver={driver}] fixture too thin to discriminate order: {baseline_ids}"
        )
        assert baseline_ids == suppressed_ids, (
            f"[driver={driver}] suppression changed hit identity/order: "
            f"{baseline_ids} != {suppressed_ids}"
        )
        assert baseline_anti_ids == suppressed_anti_ids, (
            f"[driver={driver}] suppression changed anti_hit identity/order: "
            f"{baseline_anti_ids} != {suppressed_anti_ids}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fallback_recall_byte_identical_under_suppression_change(driver, tmp_path, monkeypatch):
    """Same byte-identity proof as the common path, but for the degraded
    fallback dispatch (a forced primary-pipeline failure routes through
    `retrieve.recall()`): forcing suppression on must not change which hits
    a fallback recall returns, only whether the gated provenance write
    persists."""
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path)
    try:
        _seed_multiple_probes(store, session_id=f"seed-fbid-{driver}")
        before_rows = _row_count_snapshot(store)

        import iai_mcp.pipeline as pipeline_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("forced failure to exercise the fallback dispatch")

        monkeypatch.setattr(pipeline_mod, "recall_for_response", _boom)

        import iai_mcp.recall_suppression as rs
        token = rs.recall_suppressed.set(True)
        try:
            suppressed = core.dispatch(store, "memory_recall", {
                "cue": "reference content", "session_id": f"sess-fbid-suppressed-{driver}",
            })
        finally:
            rs.recall_suppressed.reset(token)

        after_suppressed_rows = _row_count_snapshot(store)
        assert before_rows == after_suppressed_rows, (
            f"[driver={driver}] forced suppression still mutated table row "
            f"counts on the fallback dispatch: before={before_rows} "
            f"after={after_suppressed_rows}"
        )

        baseline = core.dispatch(store, "memory_recall", {
            "cue": "reference content", "session_id": f"sess-fbid-normal-{driver}",
        })

        assert "error" not in suppressed, suppressed.get("error")
        assert "error" not in baseline, baseline.get("error")
        baseline_ids = [h["record_id"] for h in baseline.get("hits", [])]
        suppressed_ids = [h["record_id"] for h in suppressed.get("hits", [])]
        assert len(baseline_ids) >= 2, (
            f"[driver={driver}] fixture too thin to discriminate order: {baseline_ids}"
        )
        assert baseline_ids == suppressed_ids, (
            f"[driver={driver}] suppression changed fallback hit "
            f"identity/order: {baseline_ids} != {suppressed_ids}"
        )
    finally:
        store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_profile_modulate_gate_fires_and_suppresses(driver, tmp_path, monkeypatch):
    """The profile-modulate suppression gate wraps BOTH branches of its
    kill-switch (the deferred `queue_profile_modulate` write and the
    synchronous `boost_edges` fallback) but no fixture in this suite ever
    seeded a non-empty profile gain, so neither branch had actually run
    either way. Drives `_apply_post_rank_pipeline` directly with an
    explicit gain, once per branch, unsuppressed (must fire) and suppressed
    (must not)."""
    _select_driver(driver, monkeypatch)
    from iai_mcp import pipeline as pipeline_mod
    from iai_mcp.graph import MemoryGraph
    import iai_mcp.recall_suppression as rs

    store = MemoryStore(path=tmp_path)
    try:
        probe_id = _seed_indexable_probe(store, session_id=f"seed-{driver}")
        rec = store.get(probe_id)
        hit = MemoryHit(
            record_id=probe_id,
            score=1.0,
            reason="test",
            literal_surface=rec.literal_surface,
            adjacent_suggestions=[],
        )
        gains = {probe_id: {"profile:test": 1.0}}
        common_kwargs = dict(
            store=store,
            graph=MemoryGraph(),
            records_cache={probe_id: rec},
            cue="reference content",
            profile_state={"active": True},
            turn=0,
            mode="verbatim",
            budget_used=0,
            path_label="test",
            profile_gains=gains,
        )

        calls = _install_side_effect_spy(store, monkeypatch)

        for method_name, env_value in (
            ("store.queue_profile_modulate", None),
            ("store.boost_edges", "1"),
        ):
            if env_value is None:
                monkeypatch.delenv("IAI_MCP_DEFER_PROFILE_BOOST_OFF", raising=False)
            else:
                monkeypatch.setenv("IAI_MCP_DEFER_PROFILE_BOOST_OFF", env_value)

            calls.clear()
            pipeline_mod._apply_post_rank_pipeline(
                [hit], session_id=f"sess-profmod-fire-{driver}", **common_kwargs,
            )
            assert method_name in calls, (
                f"[driver={driver}] {method_name} never fired on an "
                f"unsuppressed call: {calls}"
            )

            calls.clear()
            token = rs.recall_suppressed.set(True)
            try:
                pipeline_mod._apply_post_rank_pipeline(
                    [hit], session_id=f"sess-profmod-suppress-{driver}", **common_kwargs,
                )
            finally:
                rs.recall_suppressed.reset(token)
            assert method_name not in calls, (
                f"[driver={driver}] {method_name} fired while suppressed: {calls}"
            )
    finally:
        store.close()
