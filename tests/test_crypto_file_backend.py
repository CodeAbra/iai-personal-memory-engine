from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

import pytest


def test_try_file_get_returns_bytes_on_valid_0o600_file(tmp_path: Path) -> None:
    from iai_mcp.crypto import CryptoKey

    key_bytes = secrets.token_bytes(32)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(key_bytes)
    os.chmod(key_path, 0o600)

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    got = ck._try_file_get()
    assert got == key_bytes
    assert isinstance(got, bytes)
    assert len(got) == 32


def test_try_file_get_rejects_world_or_group_bits(tmp_path: Path) -> None:
    from iai_mcp.crypto import CryptoKey, CryptoKeyError

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(b"\x00" * 32)
    os.chmod(key_path, 0o644)

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    with pytest.raises(CryptoKeyError) as exc_info:
        ck._try_file_get()
    assert "insecure mode" in str(exc_info.value).lower()


def test_try_file_get_rejects_wrong_length(tmp_path: Path) -> None:
    from iai_mcp.crypto import CryptoKey, CryptoKeyError

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(b"\x01" * 31)
    os.chmod(key_path, 0o600)

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    with pytest.raises(CryptoKeyError) as exc_info:
        ck._try_file_get()
    assert "wrong length" in str(exc_info.value).lower()


def test_try_file_get_rejects_foreign_uid(tmp_path: Path, monkeypatch) -> None:
    from iai_mcp.crypto import CryptoKey, CryptoKeyError

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(b"\x02" * 32)
    os.chmod(key_path, 0o600)

    real_stat = os.stat
    real_result = real_stat(key_path)
    foreign_uid = (os.geteuid() + 12345) & 0xFFFF

    forged = os.stat_result((
        real_result.st_mode,
        real_result.st_ino,
        real_result.st_dev,
        real_result.st_nlink,
        foreign_uid,
        real_result.st_gid,
        real_result.st_size,
        real_result.st_atime,
        real_result.st_mtime,
        real_result.st_ctime,
    ))

    target_str = str(key_path)

    def fake_stat(path, *args, **kwargs):
        try:
            path_str = str(path)
        except Exception:
            return real_stat(path, *args, **kwargs)
        if path_str == target_str:
            return forged
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fake_stat)

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    with pytest.raises(CryptoKeyError) as exc_info:
        ck._try_file_get()
    assert "uid" in str(exc_info.value).lower()


def test_try_file_set_writes_atomic_with_0o600(tmp_path: Path) -> None:
    from iai_mcp.crypto import CryptoKey

    payload = b"\x00" * 32
    ck = CryptoKey(user_id="t", store_root=tmp_path)
    ck._try_file_set(payload)

    key_path = tmp_path / ".crypto.key"
    assert key_path.exists()
    assert key_path.read_bytes() == payload
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600

    leftover_tmps = list(tmp_path.glob(".crypto.key.tmp.*"))
    assert leftover_tmps == [], f"leaked tmp files: {leftover_tmps}"


def test_try_file_set_cleans_stale_tmp(tmp_path: Path) -> None:
    from iai_mcp.crypto import CryptoKey

    stale_tmp = tmp_path / ".crypto.key.tmp.99999"
    stale_tmp.write_bytes(b"GARBAGE-FROM-CRASHED-PRIOR-RUN")

    payload = b"\x01" * 32
    ck = CryptoKey(user_id="t", store_root=tmp_path)
    ck._try_file_set(payload)

    assert not stale_tmp.exists(), "stale tmp must be cleaned up before the new write"
    key_path = tmp_path / ".crypto.key"
    assert key_path.exists()
    assert key_path.read_bytes() == payload


def test_get_or_create_prefers_file_over_passphrase(
    tmp_path: Path, monkeypatch
) -> None:
    from iai_mcp.crypto import CryptoKey

    key_a = secrets.token_bytes(32)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(key_a)
    os.chmod(key_path, 0o600)

    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "hunter2")

    ck = CryptoKey(user_id="t", store_root=tmp_path)
    got = ck.get_or_create()
    assert got == key_a, "file-backed key must win over passphrase fallback"


def test_cmd_crypto_migrate_to_file_happy_path(
    tmp_path: Path, monkeypatch
) -> None:
    import argparse
    import base64
    import keyring as _keyring

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    keyring_key = secrets.token_bytes(32)
    keyring_blob = base64.urlsafe_b64encode(keyring_key).decode("ascii")

    def fake_get(service: str, username: str) -> str | None:
        return keyring_blob

    def fake_delete(service: str, username: str) -> None:
        pass

    monkeypatch.setattr(_keyring, "get_password", fake_get)
    monkeypatch.setattr(_keyring, "delete_password", fake_delete)

    from iai_mcp.cli import cmd_crypto_migrate_to_file

    args = argparse.Namespace(
        user_id="default", keep_keychain=True, delete_keychain=False
    )
    exit_code = cmd_crypto_migrate_to_file(args)
    assert exit_code == 0

    key_path = tmp_path / ".crypto.key"
    assert key_path.exists()
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600
    assert key_path.read_bytes() == keyring_key, (
        "file contents must equal the round-tripped keyring key bytes"
    )


def test_cmd_crypto_migrate_to_file_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    import argparse
    import keyring as _keyring

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    pre_existing = secrets.token_bytes(32)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(pre_existing)
    os.chmod(key_path, 0o600)

    def assert_not_called(*args, **kwargs):
        raise AssertionError(
            "keyring touched on idempotent path — migrate-to-file must "
            "skip keyring entirely when the file is already present"
        )

    monkeypatch.setattr(_keyring, "get_password", assert_not_called)
    monkeypatch.setattr(_keyring, "delete_password", assert_not_called)

    from iai_mcp.cli import cmd_crypto_migrate_to_file

    args = argparse.Namespace(
        user_id="default", keep_keychain=True, delete_keychain=False
    )
    exit_code = cmd_crypto_migrate_to_file(args)
    assert exit_code == 0
    assert key_path.read_bytes() == pre_existing
