from __future__ import annotations

from unittest.mock import MagicMock


def test_check_q_pass_when_iai_in_path(monkeypatch):
    from iai_mcp.doctor import check_q_iai_cli_reachable

    monkeypatch.setattr("shutil.which", lambda _: "/Users/test/.venv/bin/iai")

    def _fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "iai 0.1.0\n"
        result.stderr = ""
        return result

    monkeypatch.setattr("subprocess.run", _fake_run)

    r = check_q_iai_cli_reachable()
    assert r.status == "PASS"
    assert r.passed is True
    assert "iai 0.1.0" in r.detail
    assert "/Users/test/.venv/bin/iai" in r.detail


def test_check_q_warn_when_iai_not_in_path(monkeypatch):
    from iai_mcp.doctor import check_q_iai_cli_reachable

    monkeypatch.setattr("shutil.which", lambda _: None)

    r = check_q_iai_cli_reachable()
    assert r.status == "WARN"
    assert r.passed is True
    assert "pip install" in r.detail.lower()


def test_check_q_warn_on_nonzero_exit(monkeypatch):
    from iai_mcp.doctor import check_q_iai_cli_reachable

    monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/iai")

    def _fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 2
        result.stdout = ""
        result.stderr = "simulated crash"
        return result

    monkeypatch.setattr("subprocess.run", _fake_run)

    r = check_q_iai_cli_reachable()
    assert r.status == "WARN"
    assert "simulated crash" in r.detail


def test_check_q_warn_on_subprocess_error(monkeypatch):
    import subprocess

    from iai_mcp.doctor import check_q_iai_cli_reachable

    monkeypatch.setattr("shutil.which", lambda _: "/some/path/iai")

    def _raise(*a, **k):
        raise subprocess.SubprocessError("simulated")

    monkeypatch.setattr("subprocess.run", _raise)

    r = check_q_iai_cli_reachable()
    assert r.status == "WARN"
    assert "simulated" in r.detail


def test_check_q_in_run_diagnosis():
    from iai_mcp.doctor import run_diagnosis

    results = run_diagnosis()
    names = [r.name for r in results]

    q_rows = [n for n in names if n.startswith("(q)")]
    assert len(q_rows) == 1, f"expected one (q) row, got {q_rows}"

    p_idx = next((i for i, n in enumerate(names) if n.startswith("(p)")), -1)
    q_idx = names.index(q_rows[0])
    z_idx = next((i for i, n in enumerate(names) if n.startswith("(z)")), -1)

    assert p_idx < q_idx, "expected (p) < (q) ordering"
    if z_idx >= 0:
        assert q_idx < z_idx, "(z) must remain the last row"
