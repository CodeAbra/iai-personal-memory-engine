"""The nightly embedding-integrity self-heal: two cursor-bounded scan lanes.

Lane A flags active rows whose stored vector is all-zero (a write bug or a
crash between blob and flag), Lane B re-embeds a rotating sample of rows and
flags those whose stored vector has drifted from ``embed(literal_surface)`` (a
historical capture bug, or an embedder-model swap). Both feed a shared,
resident-set-bounded write-back that re-embeds each flagged row from its intact
text — the encryption/verbatim boundary is never touched.

Hermetic: a tmp store path, an in-process deterministic embedder, no daemon.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.hippo import HippoDB
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord
from iai_mcp.write import cosine


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "embed-integrity-test-pass")
    yield


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


class _DeterministicEmbedder:
    """Maps text to a stable unit vector by content (no model load)."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v.tolist()

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


@pytest.fixture
def emb(store: MemoryStore) -> _DeterministicEmbedder:
    return _DeterministicEmbedder(store.embed_dim)


@pytest.fixture
def pipeline(store: MemoryStore, tmp_path: Path):
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=log_dir),
    )


def _make_record(store, text, embedding, rid=None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid or uuid4(), tier="episodic",
        literal_surface=text, aaak_index="", embedding=list(embedding),
        community_id=None, centrality=0.0, detail_level=2, pinned=False,
        stability=0.0, difficulty=0.0, last_reviewed=None, never_decay=False,
        never_merge=False, provenance=[], created_at=now, updated_at=now,
        tags=["capture", "role:user"], language="en",
    )


def _insert(store, text, embedding, rid=None) -> UUID:
    rid = rid or uuid4()
    store.insert(_make_record(store, text, embedding, rid=rid))
    return rid


def _stored(store, rid) -> list[float]:
    rec = store.get(rid)
    assert rec is not None
    return list(rec.embedding)


def _pending_flag(store, rid) -> int:
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT embedding_pending FROM records WHERE id = ?", (str(rid),)
        ).fetchone()
    assert row is not None
    return int(row[0] or 0)


# ---------------------------------------------------------------------------
# Lane A: zero-vector autoheal
# ---------------------------------------------------------------------------

def test_zero_vector_active_row_autoheals(store, emb):
    text = "The daemon consolidates memories only during the nightly sleep cycle."
    rid = _insert(store, text, [0.0] * store.embed_dim)
    flush_record_buffer(store)

    # Seed-time proof: it actually landed as an all-zero, non-pending row.
    assert not any(_stored(store, rid)), "seed must be all-zero before heal"
    assert _pending_flag(store, rid) == 0, "seed row must be active (not pending)"
    count_before = store.active_records_count()

    result = store.db.selfheal_degraded_embeddings(emb, embedder_for_zero_lane=emb)

    assert result["zero_found"] == 1
    assert result["zero_healed"] == 1
    healed = _stored(store, rid)
    assert any(healed), "vector must no longer be all-zero"
    assert cosine(healed, list(emb.embed(text))) > 0.98
    # AES/verbatim boundary untouched, flag unchanged, count unchanged.
    assert store.get(rid).literal_surface == text
    assert _pending_flag(store, rid) == 0
    assert store.active_records_count() == count_before, (
        "count_transition=False: an already-active row must not shift the count"
    )


# ---------------------------------------------------------------------------
# Lane B: text/embedding-mismatch autoheal
# ---------------------------------------------------------------------------

def test_mismatched_active_row_autoheals(store, emb):
    correct = "Orbital mechanics govern how a satellite holds a stable trajectory."
    wrong = "Sourdough fermentation depends on wild yeast and ambient temperature."
    # A few matching rows so the flagged fraction stays below the breaker.
    for i in range(9):
        t = f"unrelated healthy record number {i} with its own distinct content"
        _insert(store, t, list(emb.embed(t)))
    rid = _insert(store, correct, list(emb.embed(wrong)))
    flush_record_buffer(store)

    assert cosine(_stored(store, rid), list(emb.embed(wrong))) > 0.999

    result = store.db.selfheal_degraded_embeddings(emb, embedder_for_zero_lane=emb)

    assert result["sample_flagged"] == 1
    assert result["sample_healed"] == 1
    assert result["mass_mismatch_detected"] is False
    post = _stored(store, rid)
    assert cosine(post, list(emb.embed(correct))) > 0.98
    assert cosine(post, list(emb.embed(wrong))) < 0.5, "the vector actually changed"


# ---------------------------------------------------------------------------
# Budget truncation retries the SAME window next cycle (not a full rotation)
# ---------------------------------------------------------------------------

def test_rss_truncated_write_back_retries_next_cycle_not_next_rotation(
    store, emb, monkeypatch,
):
    rids = [
        _insert(store, f"zero-vector row number {i} carrying real recoverable text", [0.0] * store.embed_dim)
        for i in range(3)
    ]
    flush_record_buffer(store)

    # First resident-set reading (the pass baseline) is small; every later
    # reading is huge, so the soft cap trips before the SECOND write window.
    calls = {"n": 0}

    def fake_rss():
        calls["n"] += 1
        return 1000 if calls["n"] == 1 else 10**12

    monkeypatch.setattr(HippoDB, "_reembed_rss_bytes", staticmethod(fake_rss))

    # zero_scan_batch=3 == corpus size, so cursor_wrapped_a is False and a
    # HELD cursor ("") is distinguishable from an ADVANCED cursor (a real id).
    r1 = store.db.selfheal_degraded_embeddings(
        emb, embedder_for_zero_lane=emb,
        zero_scan_batch=3, batch_size=2, rss_soft_cap_bytes=1,
    )
    assert r1["zero_found"] == 3
    assert r1["zero_healed"] == 2, "only the first write-back window fits the budget"
    cur1 = store.db._load_embed_integrity_cursor()
    assert cur1.get("zero_scan_last_id", "") == "", (
        "a budget-truncated window must NOT advance the cursor"
    )
    healed_count = sum(1 for r in rids if any(_stored(store, r)))
    assert healed_count == 2

    # Next cycle re-examines the SAME window and heals the remaining row.
    calls["n"] = 0
    r2 = store.db.selfheal_degraded_embeddings(
        emb, embedder_for_zero_lane=emb,
        zero_scan_batch=3, batch_size=2, rss_soft_cap_bytes=1,
    )
    assert r2["zero_found"] == 1, "the two healed rows are no longer zero"
    assert r2["zero_healed"] == 1
    cur2 = store.db._load_embed_integrity_cursor()
    assert cur2.get("zero_scan_last_id", "") != "", (
        "a fully-healed window advances the cursor"
    )
    assert all(any(_stored(store, r)) for r in rids), "all three healed within two cycles"


# ---------------------------------------------------------------------------
# The cursor rotates and re-sweeps the whole corpus
# ---------------------------------------------------------------------------

def test_cursor_wraps_and_resweeps(store, emb):
    wrong = "one shared wrong provenance label reused for every seeded row"
    rids = [
        _insert(store, f"drifted record number {i} with unique recoverable content", list(emb.embed(wrong)))
        for i in range(5)
    ]
    flush_record_buffer(store)

    total_healed = 0
    wrapped = False
    for _ in range(3):
        # Breaker disabled (fraction 1.1 is unreachable) so this isolates the
        # cursor-wrap behavior from the mass-mismatch path.
        r = store.db.selfheal_degraded_embeddings(
            emb, embedder_for_zero_lane=emb,
            sample_size=2, mass_mismatch_fraction=1.1,
        )
        total_healed += r["sample_healed"]
        wrapped = wrapped or r["cursor_wrapped_b"]

    assert total_healed == 5, "every row healed exactly once across the sweep"
    assert wrapped, "the sample cursor wrapped at end-of-corpus"
    for rid, i in zip(rids, range(5)):
        text = f"drifted record number {i} with unique recoverable content"
        assert cosine(_stored(store, rid), list(emb.embed(text))) > 0.98


# ---------------------------------------------------------------------------
# The mass-mismatch circuit breaker: heal NOTHING on a systemic signal
# ---------------------------------------------------------------------------

def test_mass_mismatch_trips_circuit_breaker(store, emb):
    wrong = "single shared wrong label across every degraded row in the sample"
    for i in range(20):
        t = f"healthy record number {i} with its own genuinely distinct content"
        _insert(store, t, list(emb.embed(t)))
    bad_ids = [
        _insert(store, f"drifted record number {i} with distinct recoverable text", list(emb.embed(wrong)))
        for i in range(10)
    ]
    flush_record_buffer(store)

    # 10 flagged / 30 compared = 0.33 > 0.20, and 30 >= the min-sample floor,
    # so this reads as SYSTEMIC drift, not sparse per-row corruption.
    result = store.db.selfheal_degraded_embeddings(emb, embedder_for_zero_lane=emb)

    assert result["mass_mismatch_detected"] is True
    assert result["low_sample"] is False
    assert result["sample_healed"] == 0, "a systemic signal heals nothing this cycle"
    for rid in bad_ids:
        assert cosine(_stored(store, rid), list(emb.embed(wrong))) > 0.999, (
            "no row's embedding may change when the breaker trips"
        )
    # The window was fully examined (intentionally held) so the cursor advances
    # — it re-reports every cycle rather than going silent.
    cur = store.db._load_embed_integrity_cursor()
    assert cur.get("sample_sweeps_completed", 0) >= 1


def test_low_sample_systemic_drift_defers_not_mass_rewrites(store, emb):
    """On a small store the fraction is not trustworthy as systemic, so the
    breaker must NOT rewrite the flagged rows into a possibly-drifted space: it
    defers, holds its cursor, and flags low_sample. Nothing is healed."""
    wrong = "one shared wrong label so every drifted small-store row reads low"
    # 5 healthy + 5 drifted = 10 comparable rows, below the 25 min-mass floor.
    # 5/10 = 0.5 > 0.20 would trip the breaker on a trusted-size sample, but 10
    # rows is too small to call systemic — the low_sample defer must engage.
    for i in range(5):
        t = f"healthy small-store record number {i} with its own distinct content"
        _insert(store, t, list(emb.embed(t)))
    bad_ids = [
        _insert(
            store,
            f"drifted small-store record number {i} with recoverable text",
            list(emb.embed(wrong)),
        )
        for i in range(5)
    ]
    flush_record_buffer(store)

    result = store.db.selfheal_degraded_embeddings(emb, embedder_for_zero_lane=emb)

    assert result["low_sample"] is True
    assert result["mass_mismatch_detected"] is False
    assert result["sample_flagged"] == 5
    assert result["sample_healed"] == 0, (
        "a low-sample systemic signal defers — it rewrites no vector"
    )
    for i, rid in enumerate(bad_ids):
        assert cosine(_stored(store, rid), list(emb.embed(wrong))) > 0.999, (
            "no flagged row's vector may change on a deferred low-sample cycle"
        )
        assert store.get(rid).literal_surface == (
            f"drifted small-store record number {i} with recoverable text"
        ), "literal_surface stays intact so --reembed-from-text can recover"
    # The cursor is held so the window is re-examined next cycle (once more rows
    # can be compared), rather than advanced past an undecided window.
    cur = store.db._load_embed_integrity_cursor()
    assert cur.get("sample_last_id", "") == "", "a deferred low-sample lane holds its cursor"


def test_permanent_writeback_fault_is_bounded_not_livelocked(store, emb, monkeypatch):
    """A method-level write-back fault holds the lane cursor to retry, but a
    permanent storage fault must not hold it forever. After a bounded number of
    consecutive faults at one window the lane advances past it and the retry
    budget resets, so the lane cannot livelock on one broken window."""
    from iai_mcp.hippo import _db as dbmod

    text = "a zero row whose write-back keeps failing at the storage layer"
    rid = _insert(store, text, [0.0] * store.embed_dim)
    flush_record_buffer(store)

    def boom(*a, **k):
        raise RuntimeError("permanent storage fault in the write-back")

    monkeypatch.setattr(store.db, "_reembed_write_ids", boom)

    # Cycles below the bound: the fault holds the window; the counter climbs.
    for expected in range(1, dbmod.EMBED_INTEGRITY_MAX_WRITEBACK_RETRIES):
        r = store.db.selfheal_degraded_embeddings(emb, embedder_for_zero_lane=emb)
        assert r["zero_found"] == 1
        assert r["zero_healed"] == 0, "a failing write-back heals nothing"
        assert not any(_stored(store, rid)), "the vector stays zero while the fault persists"
        cur = store.db._load_embed_integrity_cursor()
        assert cur.get("zero_writeback_faults", 0) == expected

    # The bound-reaching cycle force-advances past the window and resets the
    # budget — the lane does not hold the same window forever.
    store.db.selfheal_degraded_embeddings(emb, embedder_for_zero_lane=emb)
    cur = store.db._load_embed_integrity_cursor()
    assert cur.get("zero_writeback_faults", 0) == 0, (
        "reaching the retry bound resets the counter (the window is advanced past)"
    )
    # literal_surface was never touched, so the row stays recoverable.
    assert store.get(rid).literal_surface == text


# ---------------------------------------------------------------------------
# Identity mismatch: skip Lane B, still heal Lane A via the bypass
# ---------------------------------------------------------------------------

def test_identity_mismatch_skips_lane_b_still_heals_lane_a(
    store, emb, pipeline, monkeypatch,
):
    import iai_mcp.embed as embmod
    import iai_mcp.events as evmod
    from iai_mcp.embed import EmbedIdentityMismatch

    def fake_resolve(store_arg, *, allow_identity_mismatch=False):
        if not allow_identity_mismatch:
            raise EmbedIdentityMismatch("test: stamped identity does not match runtime")
        return emb

    monkeypatch.setattr(embmod, "embedder_for_store", fake_resolve)

    captured: list[tuple[str, str]] = []
    real_we = evmod.write_event

    def spy_we(store_arg, kind, data, **kw):
        captured.append((kind, kw.get("severity")))
        return real_we(store_arg, kind, data, **kw)

    monkeypatch.setattr(evmod, "write_event", spy_we)

    zero_text = "a zero vector is no answer at all and must be refillable"
    zero_rid = _insert(store, zero_text, [0.0] * store.embed_dim)
    wrong = "a wrong-space vector that Lane B alone would be allowed to touch"
    mismatch_rid = _insert(store, "correct anchor text for the drifted row", list(emb.embed(wrong)))
    flush_record_buffer(store)

    done, result = pipeline._step_embedding_integrity(None)

    assert done is True
    assert result["sample_lane_skipped"] is True
    assert result["zero_lane_embedder_unavailable"] is False
    # Lane A healed the zero row through the explicitly-bypassed embedder.
    assert result["zero_healed"] == 1
    assert cosine(_stored(store, zero_rid), list(emb.embed(zero_text))) > 0.98
    # Lane B never ran, so the mismatched-but-nonzero row is untouched.
    assert cosine(_stored(store, mismatch_rid), list(emb.embed(wrong))) > 0.999
    assert ("embed_integrity_selfheal", "warning") in captured


def test_identity_mismatch_bypass_also_fails_detects_only(
    store, emb, pipeline, monkeypatch,
):
    import iai_mcp.embed as embmod
    from iai_mcp.embed import EmbedderConfigError, EmbedIdentityMismatch

    def fake_resolve(store_arg, *, allow_identity_mismatch=False):
        if not allow_identity_mismatch:
            raise EmbedIdentityMismatch("test: identity mismatch")
        raise EmbedderConfigError("test: no embedder resolvable at all")

    monkeypatch.setattr(embmod, "embedder_for_store", fake_resolve)

    zero_rid = _insert(store, "text for a zero row that cannot be healed this cycle", [0.0] * store.embed_dim)
    flush_record_buffer(store)

    done, result = pipeline._step_embedding_integrity(None)

    assert done is True
    assert result["zero_found"] == 1
    assert result["zero_healed"] == 0
    assert result["zero_lane_embedder_unavailable"] is True
    assert not any(_stored(store, zero_rid)), "nothing written when no embedder resolves"


# ---------------------------------------------------------------------------
# End-to-end: recall reaches a healed zero-vector record
# ---------------------------------------------------------------------------

def test_recall_reaches_healed_zero_vector_record(store, emb):
    for i in range(4):
        t = f"an unrelated decoy record number {i} about assorted topics"
        _insert(store, t, list(emb.embed(t)))
    text = "the rich-club hubs integrate distributed brain-network communities"
    rid = _insert(store, text, [0.0] * store.embed_dim)
    flush_record_buffer(store)

    result = store.db.selfheal_degraded_embeddings(emb, embedder_for_zero_lane=emb)
    assert result["zero_healed"] == 1

    hits = store.query_similar(list(emb.embed(text)), k=5)
    hit_ids = {str(rec.id) for rec, _score in hits}
    assert str(rid) in hit_ids, "the healed record must be reachable by similarity recall"


# ---------------------------------------------------------------------------
# Never-raise contract
# ---------------------------------------------------------------------------

def test_step_never_raises_on_embed_failure(store, pipeline, monkeypatch):
    """An embedder that raises inside the write-back yields normal counts with
    nothing healed — the per-row guard swallows it, the step never raises."""
    import iai_mcp.embed as embmod
    import iai_mcp.events as evmod

    class _RaisingEmbedder:
        def embed(self, text):
            raise RuntimeError("boom: embed always fails")

        def embed_batch(self, texts):
            raise RuntimeError("boom: batch always fails")

    monkeypatch.setattr(embmod, "embedder_for_store", lambda s, **k: _RaisingEmbedder())

    captured: list[tuple[str, str]] = []
    real_we = evmod.write_event

    def spy_we(store_arg, kind, data, **kw):
        captured.append((kind, kw.get("severity")))
        return real_we(store_arg, kind, data, **kw)

    monkeypatch.setattr(evmod, "write_event", spy_we)

    _insert(store, "a real zero row whose heal will fail at embed time", [0.0] * store.embed_dim)
    flush_record_buffer(store)

    done, result = pipeline._step_embedding_integrity(None)

    assert done is True
    assert isinstance(result, dict)
    assert "error" not in result, "a per-row embed failure is not a step error"
    assert result["zero_found"] == 1
    assert result["zero_healed"] == 0
    assert ("embed_integrity_selfheal", "warning") in captured


def test_drift_lane_defers_to_a_waiting_foreground(store, emb):
    """When a foreground read arrives mid-sample, the drift lane stops, holds
    its cursor, and heals nothing — sleep never starves the awake path."""
    wrong = "one shared wrong label so every seeded row reads as drifted"
    for i in range(40):
        _insert(store, f"drifted record number {i} with its own recoverable text", list(emb.embed(wrong)))
    flush_record_buffer(store)

    # False on the caller's pre-lane check, True at the first in-loop yield
    # (row 32), so the lane is entered and then interrupted partway through.
    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        return calls["n"] >= 2

    result = store.db.selfheal_degraded_embeddings(
        emb, embedder_for_zero_lane=emb, interrupt_check=interrupt,
    )

    assert result["sample_lane_deferred"] is True
    assert result["sample_healed"] == 0
    assert result["sample_scanned"] == 32, "reports rows actually examined, not fetched"
    cur = store.db._load_embed_integrity_cursor()
    assert cur.get("sample_last_id", "") == "", "a deferred lane holds its cursor"


def test_foreground_deferred_lane_reports_warning(store, emb, pipeline, monkeypatch):
    """A drift lane deferred to a waiting foreground carried its whole window
    into the next cycle: the step is found-but-not-fully-healed and must emit
    warning, not info (the observability half of the defer path)."""
    import iai_mcp.embed as embmod
    import iai_mcp.events as evmod

    monkeypatch.setattr(embmod, "embedder_for_store", lambda s, **k: emb)

    captured: list[tuple[str, str]] = []
    real_we = evmod.write_event

    def spy_we(store_arg, kind, data, **kw):
        captured.append((kind, kw.get("severity")))
        return real_we(store_arg, kind, data, **kw)

    monkeypatch.setattr(evmod, "write_event", spy_we)

    # Healthy, nonzero rows: nothing is zero and nothing drifts, so the deferral
    # is the ONLY not-fully-healed signal — without it the step would read info.
    for i in range(5):
        t = f"a healthy nonzero record number {i} with its own content"
        _insert(store, t, list(emb.embed(t)))
    flush_record_buffer(store)

    # False on the step's top-level interrupt check, True at the drift lane's
    # pre-lane foreground check, so Lane B defers before it scans a single row.
    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        return calls["n"] >= 2

    done, result = pipeline._step_embedding_integrity(interrupt)

    assert done is True
    assert result["sample_lane_deferred"] is True
    assert result["zero_found"] == 0
    assert result["sample_flagged"] == 0
    assert ("embed_integrity_selfheal", "warning") in captured, (
        "a deferred lane carried its window forward and must not report info"
    )


def test_step_never_raises_when_selfheal_itself_raises(store, pipeline, monkeypatch):
    """If the whole self-heal call raises, the step's outer guard returns
    (True, {'error': ...}) rather than propagating and quarantining the cycle."""
    import iai_mcp.embed as embmod

    monkeypatch.setattr(
        embmod, "embedder_for_store",
        lambda s, **k: _DeterministicEmbedder(store.embed_dim),
    )

    def boom(*a, **k):
        raise RuntimeError("total self-heal failure")

    monkeypatch.setattr(store.db, "selfheal_degraded_embeddings", boom)

    done, result = pipeline._step_embedding_integrity(None)

    assert done is True
    assert "error" in result, "the mandatory outer never-raise guard must catch it"
