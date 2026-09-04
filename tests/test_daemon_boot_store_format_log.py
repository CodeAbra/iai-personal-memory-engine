"""Boot-time visibility: the daemon names the resolved store format once.

A diagnostic, not a precondition -- the helper never aborts boot even when
the store object it receives is malformed.
"""

from __future__ import annotations

import logging
import re

import pytest

from iai_mcp.daemon import _log_store_format


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", "32")


def test_log_store_format_native_store(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    try:
        with caplog.at_level(logging.INFO, logger="iai_mcp.daemon"):
            _log_store_format(store)

        records = [r for r in caplog.records if r.name == "iai_mcp.daemon"]
        assert len(records) == 1
        message = records[0].getMessage()
        assert "lilli" in message
        assert str(store.root) in message
        assert not re.search(r"key|passphrase|token|secret", message, re.IGNORECASE)
    finally:
        store.close()


def test_log_store_format_legacy_store(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    store = MemoryStore(path=tmp_path)
    try:
        assert store.db.storage_driver == "stdlib"

        with caplog.at_level(logging.INFO, logger="iai_mcp.daemon"):
            _log_store_format(store)

        records = [r for r in caplog.records if r.name == "iai_mcp.daemon"]
        assert len(records) == 1
        message = records[0].getMessage()
        assert "stdlib" in message
        assert str(store.root) in message
    finally:
        store.close()


def test_log_store_format_missing_attribute_is_swallowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Stub:
        pass

    with caplog.at_level(logging.INFO, logger="iai_mcp.daemon"):
        _log_store_format(_Stub())  # must not raise

    records = [r for r in caplog.records if r.name == "iai_mcp.daemon"]
    assert records == []


def test_call_site_follows_store_open_in_main() -> None:
    import inspect

    import iai_mcp.daemon as daemon_module

    source = inspect.getsource(daemon_module)
    main_idx = source.index("async def main() -> int:")
    body = source[main_idx:]

    open_idx = body.index("store = await _open_exclusive_store_with_backoff(")
    call_idx = body.index("_log_store_format(store)")

    assert open_idx < call_idx, (
        "_log_store_format(store) must be called after the store-open "
        "assignment inside main()"
    )
