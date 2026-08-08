"""Deferred-capture spool encryption: lines encrypt with the store's key
FILE (never keychain/passphrase — writers run inside editor hooks), keyless
stores fall back to plaintext, readers accept both wire formats, tampered
ciphertext degrades to a skipped line, and the AAD constant is pinned."""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

import pytest


@pytest.fixture()
def spool_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)
    import iai_mcp.capture as capture_mod

    monkeypatch.setattr(capture_mod, "_spool_key_cache", {})
    monkeypatch.setattr(capture_mod, "_spool_plaintext_warned", False)
    return store_root


def _plant_key(store_root: Path) -> bytes:
    key = secrets.token_bytes(32)
    key_path = store_root / ".crypto.key"
    key_path.write_bytes(key)
    os.chmod(key_path, 0o600)
    return key


def test_keyless_spool_stays_plaintext_and_readable(spool_env) -> None:
    from iai_mcp.capture import read_pending_live_events, write_deferred_event

    text = "keyless spool event text long enough to matter"
    path = write_deferred_event("sess-keyless", "user", text)
    raw = path.read_text(encoding="utf-8")
    assert text in raw
    assert "iai:enc:v1:" not in raw

    events = read_pending_live_events("sess-keyless")
    assert [e["text"] for e in events] == [text]


def test_keyed_spool_is_ciphertext_on_disk(spool_env) -> None:
    _plant_key(spool_env)
    from iai_mcp.capture import read_pending_live_events, write_deferred_event

    text = "secret memo about the alpha project rollout"
    path = write_deferred_event("sess-enc", "user", text)
    raw = path.read_text(encoding="utf-8")
    assert text not in raw
    assert "sess-enc" not in raw, "the header (session id, cwd) must encrypt too"
    for line in raw.splitlines():
        assert line.startswith("iai:enc:v1:")

    events = read_pending_live_events("sess-enc")
    assert [e["text"] for e in events] == [text]


def test_mixed_legacy_plaintext_lines_still_read(spool_env) -> None:
    _plant_key(spool_env)
    from iai_mcp.capture import read_pending_live_events, write_deferred_event

    new_text = "fresh encrypted line beside legacy plaintext"
    path = write_deferred_event("sess-mixed", "user", new_text)
    legacy = {
        "text": "legacy plaintext line written before encryption",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-08-01T00:00:00+00:00",
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(legacy) + "\n")

    texts = {e["text"] for e in read_pending_live_events("sess-mixed")}
    assert new_text in texts
    assert legacy["text"] in texts


def test_tampered_line_is_skipped_not_fatal(spool_env) -> None:
    _plant_key(spool_env)
    from iai_mcp.capture import read_pending_live_events, write_deferred_event

    good = "surviving turn next to a tampered one"
    path = write_deferred_event("sess-tamper", "user", good)
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = lines[1][:-6] + "AAAAAA"
    path.write_text(
        "\n".join([lines[0], tampered, lines[1]]) + "\n", encoding="utf-8"
    )

    events = read_pending_live_events("sess-tamper")
    assert [e["text"] for e in events] == [good]


def test_deferred_captures_file_encrypts_and_roundtrips(spool_env, tmp_path) -> None:
    _plant_key(spool_env)
    from iai_mcp.capture import _decode_spool_line, write_deferred_captures

    transcript = tmp_path / "transcript.jsonl"
    turn_text = "transcript turn body long enough for the capture floor"
    transcript.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": turn_text},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = write_deferred_captures("sess-batch", transcript)
    raw = out.read_text(encoding="utf-8")
    assert turn_text not in raw
    decoded = [
        json.loads(_decode_spool_line(ln))
        for ln in raw.splitlines()
        if ln.strip()
    ]
    assert decoded[0]["session_id"] == "sess-batch"
    assert any(turn_text in (d.get("text") or "") for d in decoded[1:])


def test_decode_defers_ciphertext_without_key(spool_env) -> None:
    key = _plant_key(spool_env)
    from iai_mcp.capture import _SPOOL_AAD, SpoolKeyUnavailable, _decode_spool_line
    from iai_mcp.crypto import encrypt_field

    line = encrypt_field(json.dumps({"text": "x"}), key, _SPOOL_AAD)
    (spool_env / ".crypto.key").unlink()
    import iai_mcp.capture as capture_mod

    capture_mod._spool_key_cache.clear()
    # A missing key is NOT line corruption: the typed error keeps every
    # advancing reader from walking the file toward permanent failure.
    with pytest.raises(SpoolKeyUnavailable):
        _decode_spool_line(line)


def test_missing_key_defers_live_drain_without_advancing_offset(spool_env) -> None:
    _plant_key(spool_env)
    from iai_mcp.capture import drain_active_live_captures, write_deferred_event

    path = write_deferred_event("sess-defer", "user", "turn behind a lost key")
    assert path.exists()
    (spool_env / ".crypto.key").unlink()
    import iai_mcp.capture as capture_mod

    capture_mod._spool_key_cache.clear()

    class _Store:  # never reached: decode defers before any store use
        pass

    counts = drain_active_live_captures(_Store(), exclude_session_id="-")
    assert counts["events_inserted"] == 0
    assert path.exists(), "the file must survive a keyless pass untouched"
    offset_path = (
        spool_env.parent / ".iai-mcp" / ".capture-state" / "sess-defer.drain-offset"
    )
    if offset_path.exists():
        assert int(offset_path.read_text(encoding="utf-8").strip() or "0") == 0


def test_key_rotation_re_encrypts_spool_lines(spool_env) -> None:
    import secrets as _secrets

    from iai_mcp.capture import _SPOOL_AAD
    from iai_mcp.cli._crypto import _rotate_one_spool_file
    from iai_mcp.crypto import decrypt_field, encrypt_field

    old_key = _plant_key(spool_env)
    spool_dir = spool_env / ".deferred-captures"
    spool_dir.mkdir(parents=True, exist_ok=True)
    f = spool_dir / "sess-rot.live.jsonl"
    payload = json.dumps({"text": "queued turn that must survive rotation"})
    f.write_text(
        encrypt_field(payload, old_key, _SPOOL_AAD) + "\n"
        + "plain legacy line\n",
        encoding="utf-8",
    )

    new_key = _secrets.token_bytes(32)
    changed = _rotate_one_spool_file(f, [old_key], new_key)
    assert changed == 1
    lines = f.read_text(encoding="utf-8").splitlines()
    assert decrypt_field(lines[0], new_key, _SPOOL_AAD) == payload
    assert lines[1] == "plain legacy line", "plaintext lines pass through"


def test_rotation_recovers_lines_under_a_retained_older_generation(spool_env) -> None:
    import secrets as _secrets

    from iai_mcp.capture import _SPOOL_AAD
    from iai_mcp.cli._crypto import _rotate_one_spool_file
    from iai_mcp.crypto import decrypt_field, encrypt_field

    gen0 = _secrets.token_bytes(32)
    gen1 = _plant_key(spool_env)
    spool_dir = spool_env / ".deferred-captures"
    spool_dir.mkdir(parents=True, exist_ok=True)
    f = spool_dir / "sess-gen.live.jsonl"
    p0 = json.dumps({"text": "line stranded by an earlier partial rotation"})
    p1 = json.dumps({"text": "line under the current key"})
    f.write_text(
        encrypt_field(p0, gen0, _SPOOL_AAD) + "\n"
        + encrypt_field(p1, gen1, _SPOOL_AAD) + "\n",
        encoding="utf-8",
    )

    gen2 = _secrets.token_bytes(32)
    # A re-run passes the current key plus every retained prior generation.
    changed = _rotate_one_spool_file(f, [gen1, gen0], gen2)
    assert changed == 2
    lines = f.read_text(encoding="utf-8").splitlines()
    assert decrypt_field(lines[0], gen2, _SPOOL_AAD) == p0
    assert decrypt_field(lines[1], gen2, _SPOOL_AAD) == p1


def test_spool_key_resolves_from_home_not_env_store(
    spool_env, tmp_path, monkeypatch
) -> None:
    import secrets as _secrets

    from iai_mcp.capture import _spool_key

    home_key = _plant_key(spool_env)

    other_store = tmp_path / "other-store"
    other_store.mkdir(parents=True, exist_ok=True)
    other_key_path = other_store / ".crypto.key"
    other_key_path.write_bytes(_secrets.token_bytes(32))
    os.chmod(other_key_path, 0o600)
    monkeypatch.setenv("IAI_MCP_STORE", str(other_store))

    # The spool dir is home-fixed, so its key must be too — a hook without
    # the env and a daemon with it have to agree on one key.
    assert _spool_key() == home_key


def test_spool_key_cache_invalidates_on_key_file_change(spool_env) -> None:
    import secrets as _secrets

    from iai_mcp.capture import _spool_key

    _plant_key(spool_env)
    first = _spool_key()
    assert first is not None

    new_key = _secrets.token_bytes(32)
    key_path = spool_env / ".crypto.key"
    key_path.write_bytes(new_key)
    os.chmod(key_path, 0o600)
    # Force a distinct mtime even on coarse filesystems.
    st = key_path.stat()
    os.utime(key_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    assert _spool_key() == new_key, (
        "a rotated key file must invalidate the in-process cache"
    )


def test_daemon_reencrypts_hook_staged_plaintext(spool_env) -> None:
    from iai_mcp.capture import (
        read_pending_live_events,
        reencrypt_plaintext_spool_lines,
    )

    _plant_key(spool_env)
    spool_dir = spool_env / ".deferred-captures"
    spool_dir.mkdir(parents=True, exist_ok=True)
    f = spool_dir / "sess-hookstage.live.jsonl"
    text = "turn staged in plaintext by the stdlib-only hook"
    f.write_text(
        json.dumps({"version": 1, "session_id": "sess-hookstage", "cwd": "/x"})
        + "\n"
        + json.dumps({"text": text, "tier": "episodic", "role": "user",
                      "ts": "2026-08-07T00:00:00+00:00"})
        + "\n",
        encoding="utf-8",
    )

    counts = reencrypt_plaintext_spool_lines()
    assert counts["lines"] == 2
    raw = f.read_text(encoding="utf-8")
    assert text not in raw
    for line in raw.splitlines():
        assert line.startswith("iai:enc:v1:")
    events = read_pending_live_events("sess-hookstage")
    assert [e["text"] for e in events] == [text], (
        "line count and readability must survive the in-place re-encrypt"
    )


def test_metadata_read_text_call_matches_stdlib_signature() -> None:
    import inspect
    from importlib.metadata import Distribution

    assert "encoding" not in inspect.signature(Distribution.read_text).parameters
    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "iai_mcp" / "version_check.py"
    ).read_text(encoding="utf-8")
    assert 'read_text("direct_url.json")' in src, (
        "Distribution.read_text takes no encoding kwarg; passing one raises "
        "TypeError into a bare except and kills the editable-install probe"
    )


def test_doctor_warns_on_spool_soft_cap(spool_env, monkeypatch) -> None:
    from iai_mcp.doctor._storage_checks import check_w_no_permanent_failed

    deferred_dir = spool_env / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    (deferred_dir / "big.jsonl").write_bytes(b"x" * (2 * 1024 * 1024))

    monkeypatch.setenv("IAI_MCP_SPOOL_SOFT_CAP_MB", "1")
    result = check_w_no_permanent_failed()
    assert result.status == "WARN"
    assert "soft cap" in result.detail

    monkeypatch.setenv("IAI_MCP_SPOOL_SOFT_CAP_MB", "0")
    result = check_w_no_permanent_failed()
    assert result.status != "WARN", "0 disables the soft cap"


def test_spool_aad_is_pinned_constant() -> None:
    # A filename AAD would break the spool's in-place renames
    # (.processing/.partial/.permanent-failed) — the AAD must stay constant.
    from iai_mcp.capture import _SPOOL_AAD

    assert _SPOOL_AAD == b"iai-mcp:deferred-spool:v1"


def test_rotation_salvages_lines_beside_corrupt_bytes(spool_env) -> None:
    import secrets as _secrets

    from iai_mcp.capture import _SPOOL_AAD
    from iai_mcp.cli._crypto import _rotate_one_spool_file
    from iai_mcp.crypto import decrypt_field, encrypt_field

    old_key = _plant_key(spool_env)
    quarantine = spool_env / ".deferred-captures" / ".quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    f = quarantine / "sess.permanent-failed-1.jsonl"
    payload = json.dumps({"text": "readable line inside a quarantined file"})
    good = encrypt_field(payload, old_key, _SPOOL_AAD).encode("utf-8")
    corrupt = b"\xd0\xff\xfe garbage"
    f.write_bytes(good + b"\n" + corrupt + b"\n" + b"\xd0")

    new_key = _secrets.token_bytes(32)
    changed = _rotate_one_spool_file(f, [old_key], new_key)
    assert changed == 1
    parts = f.read_bytes().split(b"\n")
    assert (
        decrypt_field(parts[0].decode("utf-8"), new_key, _SPOOL_AAD) == payload
    )
    assert parts[1] == corrupt, "corrupt bytes preserved byte-identical"
    assert parts[2] == b"\xd0", "torn non-UTF-8 tail preserved"


def test_quarantine_failure_lands_in_nonblocking_bucket(
    spool_env, monkeypatch
) -> None:
    import secrets as _secrets

    from iai_mcp.capture import _SPOOL_AAD
    from iai_mcp.cli import _crypto as crypto_cli
    from iai_mcp.crypto import decrypt_field, encrypt_field

    old_key = _plant_key(spool_env)
    spool_dir = spool_env / ".deferred-captures"
    quarantine = spool_dir / ".quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    live = spool_dir / "sess.live.jsonl"
    payload = json.dumps({"text": "live line"})
    live.write_text(
        encrypt_field(payload, old_key, _SPOOL_AAD) + "\n", encoding="utf-8"
    )
    (quarantine / "sess.permanent-failed-1.jsonl").write_bytes(b"x\n")

    orig = crypto_cli._rotate_one_spool_file

    def _fail_quarantined(path, old_keys, new_key):
        if ".quarantine" in str(path):
            raise RuntimeError("file kept changing during rotation; re-run rotate")
        return orig(path, old_keys, new_key)

    monkeypatch.setattr(
        crypto_cli, "_rotate_one_spool_file", _fail_quarantined
    )

    class _Store:
        root = spool_env

    new_key = _secrets.token_bytes(32)
    report = crypto_cli._rotate_spool_lines([old_key], new_key, _Store())
    assert report["spool_files_skipped"] == [], (
        "a quarantine failure must never land in the live bucket"
    )
    assert len(report["quarantine_files_skipped"]) == 1
    assert report["quarantine_files_skipped"][0].startswith(".quarantine/")
    lines = live.read_text(encoding="utf-8").splitlines()
    assert decrypt_field(lines[0], new_key, _SPOOL_AAD) == payload


def test_torn_ciphertext_tail_blocks_clean_rotation(spool_env) -> None:
    import secrets as _secrets

    from iai_mcp.cli._crypto import _rotate_spool_lines

    old_key = _plant_key(spool_env)
    spool_dir = spool_env / ".deferred-captures"
    spool_dir.mkdir(parents=True, exist_ok=True)
    # A crashed writer left only the leading fragment of a ciphertext line —
    # once the append completes it needs the outgoing generation.
    (spool_dir / "sess.live.jsonl").write_bytes(b"iai:enc:v1:AAAA")

    class _Store:
        root = spool_env

    report = _rotate_spool_lines(
        [old_key], _secrets.token_bytes(32), _Store()
    )
    assert any(
        "torn tail" in entry for entry in report["spool_files_skipped"]
    ), "a torn ciphertext tail is a need the retention gate must see"


def test_torn_plaintext_tail_does_not_block_rotation(spool_env) -> None:
    import secrets as _secrets

    from iai_mcp.cli._crypto import _rotate_spool_lines

    old_key = _plant_key(spool_env)
    spool_dir = spool_env / ".deferred-captures"
    spool_dir.mkdir(parents=True, exist_ok=True)
    (spool_dir / "sess.live.jsonl").write_bytes(b"{\"text\": \"plain par")

    class _Store:
        root = spool_env

    report = _rotate_spool_lines(
        [old_key], _secrets.token_bytes(32), _Store()
    )
    assert report["spool_files_skipped"] == [], (
        "a plaintext tail needs no key generation"
    )
