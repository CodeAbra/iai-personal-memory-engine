"""capture_turn applies the directive flag end-to-end: explicit values pass
through untouched, and directive-shaped phrasing with no explicit value and
no marker opt-in is never auto-flagged (the fuzzy auto-classify write
branch is retired)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.core import dispatch
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_directive_phrasing_no_longer_auto_classifies(driver, store, monkeypatch):
    """Post-deletion contract: directive-shaped phrasing with no explicit
    caller value and no marker opt-in (directive_marker_allowed defaults
    False) is never auto-flagged -- the fuzzy classify_is_directive write
    branch this test used to exercise no longer exists."""
    _select_driver(driver, monkeypatch)

    result = capture_turn(
        store=store, cue="c", text="from now on reply in English",
        session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_explicit_false_never_overridden_by_classifier(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    result = capture_turn(
        store=store, cue="c", text="always reply in English",
        directive=False, session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_chit_chat_leaves_directive_false(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    result = capture_turn(
        store=store, cue="c", text="alice attended the weekly standup meeting",
        session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_explicit_true_on_ordinary_text_sets_directive(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    result = capture_turn(
        store=store, cue="c", text="ship the release notes by Friday",
        directive=True, session_id="s1", role="user",
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is True


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_turn_never_reaches_the_retired_classifier(driver, store, monkeypatch):
    """capture_turn's directive resolution no longer imports or calls
    classify_is_directive at all -- a broken classifier is unreachable, not
    merely fail-safe. A poisoned classify_is_directive that raises on any
    call proves capture_turn never invokes it: the call is never made, and
    the capture still succeeds with directive=False."""
    _select_driver(driver, monkeypatch)

    import iai_mcp.directive_classify as _dc_mod

    calls: list[str] = []

    def _boom(text):
        calls.append(text)
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(_dc_mod, "classify_is_directive", _boom)

    result = capture_turn(
        store=store, cue="c", text="from now on reply in English",
        session_id="s1", role="user",
    )

    assert calls == [], "capture_turn must never call classify_is_directive"
    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_dispatch_memory_capture_rejects_non_bool_directive_param(
    driver, store, monkeypatch,
):
    """A non-conforming client sending 'directive' as a JSON string must not
    have it truthy-coerced (bool('false') is True) -- the RPC boundary
    coerces anything not a real bool to None and lets the classifier decide."""
    _select_driver(driver, monkeypatch)

    result = dispatch(
        store, "memory_capture",
        {
            "text": "alice attended the weekly standup meeting",
            "session_id": "s1",
            "role": "user",
            "directive": "false",
        },
    )

    assert result["status"] == "inserted", result
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is False


def _seed_near_duplicate_neighbor(
    store: MemoryStore, text: str, *, role_tag: str,
) -> MemoryRecord:
    """Insert a non-directive record whose embedding exactly matches what
    ``capture_turn`` will compute for ``text``, so the near-dup gate has a
    guaranteed >=threshold match without depending on embedder-specific
    paraphrase similarity."""
    from iai_mcp.embed import embedder_for_store

    embedding = list(embedder_for_store(store).embed(text))
    now = datetime.now(timezone.utc)
    neighbor = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="an unrelated conversational aside about lunch plans",
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
        tags=["capture", f"role:{role_tag}"],
    )
    store.insert(neighbor)
    return neighbor


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_near_dup_gate_never_folds_into_neighbor(driver, store, monkeypatch):
    """near_dup_gate=True on directive-shaped conversational text must still
    write its own row -- capture.py's cosine dedup (Layer 1) must not fold
    a budget-accepted directive into a pre-existing near-duplicate."""
    _select_driver(driver, monkeypatch)

    text = "always answer in a formal tone from now on"
    neighbor = _seed_near_duplicate_neighbor(store, text, role_tag="user")

    result = capture_turn(
        store=store, cue="c", text=text,
        directive=True, near_dup_gate=True,
        session_id="s2", role="user",
    )

    assert result["status"] == "inserted", result
    assert result["record_id"] != str(neighbor.id), (
        "a budget-accepted directive capture must never be silently folded "
        "into a near-duplicate neighbor's row"
    )
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is True


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_pattern_separation_gate_never_folds_into_neighbor(
    driver, store, monkeypatch,
):
    """An explicit directive=True on a non-conversational role/tier must
    still write its own row -- the store-level pattern-separation gate
    (Layer 2, cosine branch live since _is_conv is False here) must not
    rewrite the record id to a foreign near-duplicate."""
    _select_driver(driver, monkeypatch)

    text = "always deploy from the release branch only"
    neighbor = _seed_near_duplicate_neighbor(store, text, role_tag="system")

    result = capture_turn(
        store=store, cue="c", text=text,
        directive=True,
        session_id="s3", role="system",
    )

    assert result["status"] == "inserted", result
    assert result["record_id"] != str(neighbor.id), (
        "a budget-accepted directive capture must never be silently folded "
        "into a near-duplicate neighbor's row"
    )
    rec = store.get(UUID(result["record_id"]))
    assert rec is not None
    assert rec.directive is True
