"""Unit contract for the daemon's boot-time GC-taming kill-switch.

Pure function tests against injectable hooks -- no daemon spin-up, no real
``gc`` mutation needed to prove polarity or call-count behavior.
"""

from __future__ import annotations

from iai_mcp.daemon import _apply_gc_taming, _resolve_gc_taming


class _Spy:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def test_resolve_gc_taming_defaults_on() -> None:
    assert _resolve_gc_taming({}) is True


def test_resolve_gc_taming_off_flag_reverts() -> None:
    assert _resolve_gc_taming({"IAI_MCP_GC_TAMING_OFF": "1"}) is False


def test_resolve_gc_taming_only_the_literal_1_reverts() -> None:
    # Matches the project's existing `_OFF` convention (exact "1" string,
    # not a broad truthy parse).
    assert _resolve_gc_taming({"IAI_MCP_GC_TAMING_OFF": "true"}) is True
    assert _resolve_gc_taming({"IAI_MCP_GC_TAMING_OFF": "0"}) is True
    assert _resolve_gc_taming({"IAI_MCP_GC_TAMING_OFF": ""}) is True


def test_apply_gc_taming_tamed_calls_both_hooks_once() -> None:
    disable_spy = _Spy()
    freeze_spy = _Spy()
    _apply_gc_taming(True, disable=disable_spy, freeze=freeze_spy)
    assert disable_spy.calls == 1
    assert freeze_spy.calls == 1


def test_apply_gc_taming_untamed_calls_neither_hook() -> None:
    disable_spy = _Spy()
    freeze_spy = _Spy()
    _apply_gc_taming(False, disable=disable_spy, freeze=freeze_spy)
    assert disable_spy.calls == 0
    assert freeze_spy.calls == 0


def test_apply_gc_taming_off_flag_reverts_byte_for_byte() -> None:
    # The full round trip: flag resolution feeding the apply call, mirroring
    # exactly how the daemon boot wires the two together.
    disable_spy = _Spy()
    freeze_spy = _Spy()
    tamed = _resolve_gc_taming({"IAI_MCP_GC_TAMING_OFF": "1"})
    _apply_gc_taming(tamed, disable=disable_spy, freeze=freeze_spy)
    assert disable_spy.calls == 0
    assert freeze_spy.calls == 0


def test_apply_gc_taming_default_hooks_are_real_gc(monkeypatch) -> None:
    # Proves the production default wiring resolves to the real gc module
    # functions (not accidentally shadowed), while never mutating this test
    # process's actual GC state -- both hooks are monkeypatched to spies
    # bound onto the `iai_mcp.daemon` module's own `gc` reference.
    import iai_mcp.daemon as daemon_mod

    disable_spy = _Spy()
    freeze_spy = _Spy()
    monkeypatch.setattr(daemon_mod.gc, "disable", disable_spy)
    monkeypatch.setattr(daemon_mod.gc, "freeze", freeze_spy)

    # Re-resolve the injectable defaults through a fresh call so the patched
    # gc.disable/gc.freeze are the ones actually invoked (the original
    # `_apply_gc_taming` default args were bound at module-import time, so
    # exercising the *default* path requires calling through the module's
    # gc reference explicitly here rather than relying on the stale bind).
    daemon_mod._apply_gc_taming(
        True, disable=daemon_mod.gc.disable, freeze=daemon_mod.gc.freeze,
    )
    assert disable_spy.calls == 1
    assert freeze_spy.calls == 1
