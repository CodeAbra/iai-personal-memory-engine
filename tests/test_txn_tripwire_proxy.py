"""The cross-thread double-writer tripwire must key transaction owners on
the UNWRAPPED connection: two HippoDB proxies over one raw connection would
otherwise each get their own owner slot and the tripwire silently yields
instead of raising."""
from __future__ import annotations

import sqlite3
import threading

import pytest


def _proxy(conn):
    from iai_mcp.hippo._db import _MutationSignallingConn

    return _MutationSignallingConn(conn, lambda: None)


def test_second_proxy_same_thread_reenters(tmp_path):
    from iai_mcp.hippo._db import _txn

    raw = sqlite3.connect(str(tmp_path / "t.db"), isolation_level=None)
    raw.execute("CREATE TABLE t (x)")
    p1, p2 = _proxy(raw), _proxy(raw)

    with _txn(p1):
        p1.execute("INSERT INTO t VALUES (1)")
        # Same thread, different proxy over the SAME raw connection: the
        # owner check must recognize it and re-enter, not raise.
        with _txn(p2):
            p2.execute("INSERT INTO t VALUES (2)")
    assert raw.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
    raw.close()


def test_cross_thread_double_writer_raises(tmp_path):
    from iai_mcp.hippo import HippoIntegrityError
    from iai_mcp.hippo._db import _txn

    raw = sqlite3.connect(
        str(tmp_path / "t.db"), isolation_level=None, check_same_thread=False,
    )
    raw.execute("CREATE TABLE t (x)")
    p1, p2 = _proxy(raw), _proxy(raw)

    entered = threading.Event()
    release = threading.Event()
    holder_err: list[str] = []

    def holder():
        try:
            with _txn(p1):
                p1.execute("INSERT INTO t VALUES (1)")
                entered.set()
                assert release.wait(10)
        except Exception as exc:  # noqa: BLE001
            holder_err.append(f"{type(exc).__name__}: {exc}")
            entered.set()

    t = threading.Thread(target=holder)
    t.start()
    assert entered.wait(10)

    # A DIFFERENT thread observing the open transaction through ANOTHER
    # proxy over the same raw connection must trip the integrity error.
    with pytest.raises(HippoIntegrityError):
        with _txn(p2):
            pass

    release.set()
    t.join(10)
    assert not holder_err, holder_err
    raw.close()
