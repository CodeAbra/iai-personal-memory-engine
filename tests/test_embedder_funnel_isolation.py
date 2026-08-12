"""The module-level embedder funnel must be pristine at every test's start.

The daemon boot path swaps ``embed.embedder_for_store`` for a held-instance
closure from a worker thread; without the conftest guard the swap outlives
the booting test and every later test resolves embedders it did not choose.
The tests below are ordered: an in-test leak (plain assignment, bypassing
monkeypatch) must be gone by the next test, and a leak landing BETWEEN two
tests (simulated by a class-scoped fixture finalizer) must be gone by the
next test's setup.  Under sequential collection this discriminates both
restore sides of the guard.  When the arming side is deselected (xdist
split, -k, --lf), the between-tests check skips with an explicit reason,
while the in-test pair still passes vacuously — the sequential gate is the
authority.  The sentinel raises on resolution, so a guard regression fails
loudly here and cascades into later embedder-resolving tests by design.
"""

from __future__ import annotations

import pytest


def _leaked_funnel(_store, **_kwargs):
    raise AssertionError("a leaked embedder funnel must never be resolved")


_LEAK_ARMED = False


def test_a_deliberately_leaks_the_funnel() -> None:
    from iai_mcp import embed as embed_mod

    embed_mod.embedder_for_store = _leaked_funnel
    assert embed_mod.embedder_for_store is _leaked_funnel


def test_b_sees_pristine_funnel_after_leak(_pristine_embedder_funnel) -> None:
    from iai_mcp import embed as embed_mod

    assert embed_mod.embedder_for_store is _pristine_embedder_funnel
    assert embed_mod.embedder_for_store is not _leaked_funnel


class TestLeakBetweenTests:
    @pytest.fixture(scope="class", autouse=True)
    @staticmethod
    def _leak_after_class():
        yield
        # Runs after the last function-scoped teardown of this class and
        # before the next test's setup — the same window a daemon worker
        # thread lands its swap in.
        global _LEAK_ARMED
        from iai_mcp import embed as embed_mod

        embed_mod.embedder_for_store = _leaked_funnel
        _LEAK_ARMED = True

    def test_class_body_is_clean(self, _pristine_embedder_funnel) -> None:
        from iai_mcp import embed as embed_mod

        assert embed_mod.embedder_for_store is _pristine_embedder_funnel


class TestSetupSideRestore:
    def test_between_tests_leak_is_gone_at_setup(
        self, _pristine_embedder_funnel
    ) -> None:
        if not _LEAK_ARMED:
            pytest.skip(
                "arming class not run in this worker/selection; "
                "setup-side pin not exercised"
            )
        from iai_mcp import embed as embed_mod

        assert embed_mod.embedder_for_store is _pristine_embedder_funnel
        assert embed_mod.embedder_for_store is not _leaked_funnel
