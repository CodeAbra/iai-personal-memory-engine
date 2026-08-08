
from setproctitle import getproctitle


def test_set_process_title_sets_iai_lilli():
    from iai_mcp import daemon as _daemon

    original = getproctitle()
    try:
        _daemon._set_process_title()
        title = getproctitle()
        from iai_mcp.lifecycle_lock import DAEMON_PROCESS_TITLE_BASE
        # The title carries the resolved store path so the orphan sweep can
        # match daemons of the same store by argv[0] equality.
        assert title.startswith(DAEMON_PROCESS_TITLE_BASE + " store=")
        assert "iai_mcp.daemon" in title
        assert title.startswith("iai lilli")
    finally:
        from setproctitle import setproctitle as _spt
        _spt(original)


def test_set_process_title_fail_soft(monkeypatch):
    import setproctitle as _spt_mod
    from iai_mcp import daemon as _daemon

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated setproctitle breakage")

    monkeypatch.setattr(_spt_mod, "setproctitle", _raise)

    _daemon._set_process_title()
