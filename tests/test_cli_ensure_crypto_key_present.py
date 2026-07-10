from __future__ import annotations

import stat



def _fresh_store_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)
    (tmp_path / ".iai-mcp").mkdir(parents=True, exist_ok=True)


def test_ensure_crypto_key_generates_on_fresh_install(tmp_path, monkeypatch):
    _fresh_store_root(tmp_path, monkeypatch)
    from iai_mcp.cli import _ensure_crypto_key_present

    path = _ensure_crypto_key_present()
    assert path is not None
    assert path.exists()
    assert path.stat().st_size == 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_ensure_crypto_key_idempotent_when_file_exists(tmp_path, monkeypatch):
    _fresh_store_root(tmp_path, monkeypatch)
    from iai_mcp.cli import _ensure_crypto_key_present

    first_path = _ensure_crypto_key_present()
    assert first_path is not None
    original_bytes = first_path.read_bytes()

    second = _ensure_crypto_key_present()
    assert second is None
    assert first_path.read_bytes() == original_bytes


def test_ensure_crypto_key_skips_when_passphrase_env_set(tmp_path, monkeypatch):
    _fresh_store_root(tmp_path, monkeypatch)
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase")
    from iai_mcp.cli import _ensure_crypto_key_present

    assert _ensure_crypto_key_present() is None
    assert not (tmp_path / ".iai-mcp" / ".crypto.key").exists()
