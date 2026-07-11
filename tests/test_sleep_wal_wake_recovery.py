from __future__ import annotations

import pytest

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake

def test_wal_pending_entries_behavioral(tmp_path):
    from iai_mcp.sleep_wal import SleepWAL

    wal = SleepWAL(path=tmp_path / ".sleep-wal.jsonl")

    assert wal.pending_entries() == []

    wal.begin(operation="optimize_drop", target_ids=["test-id"])

    assert len(wal.pending_entries()) >= 1

def test_grep_production_callsite_exists():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    daemon_src = repo_root / "src" / "iai_mcp" / "daemon" / "__init__.py"
    if not daemon_src.exists():
        pytest.skip("daemon.py source not on disk (sdist/wheel install)")
    text = daemon_src.read_text()
    assert "sleep_wal_pending_recovered" in text, (
        "Regression: 'sleep_wal_pending_recovered' event emit not found in "
        "daemon.py. WAL startup recovery may have been removed."
    )
