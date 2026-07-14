"""Encryption key management commands for the iai-mcp operator CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_crypto_status(args: argparse.Namespace) -> int:
    import json as _json
    import os as _os

    from iai_mcp.crypto import CIPHERTEXT_PREFIX, CryptoKey, KEY_BYTES

    user_id = getattr(args, "user_id", None) or "default"
    ck = CryptoKey(user_id=user_id)
    path = ck._key_file_path()

    present = path.exists()
    status: dict[str, object] = {
        "user_id": user_id,
        "backend": "file",
        "path": str(path),
        "present": present,
        "algorithm": "AES-256-GCM",
        "format": CIPHERTEXT_PREFIX,
    }

    if present:
        st = path.stat()
        mode_octal = f"0o{st.st_mode & 0o777:03o}"
        length = st.st_size
        # POSIX mode bits and euid ownership are the security model on
        # macOS/Linux; on Windows os.geteuid() does not exist and st_mode/st_uid
        # are synthetic (access is governed by ACLs, which crypto.py sets via
        # icacls). Report the POSIX-specific assessments only where they mean
        # something, mirroring the os.name guard in crypto.py's read path. Key
        # insertion order (mode, mode_secure, uid, uid_matches_process) is kept
        # identical to the pre-Windows code so the POSIX JSON output is unchanged.
        status["mode"] = mode_octal
        status["mode_secure"] = (st.st_mode & 0o077 == 0) if _os.name != "nt" else None
        status["uid"] = st.st_uid
        status["uid_matches_process"] = (
            (st.st_uid == _os.geteuid()) if _os.name != "nt" else None
        )
        status["length_bytes"] = length
        status["length_valid"] = (length == KEY_BYTES)
        status["passphrase_fallback_set"] = bool(
            _os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
        )
    else:
        status["passphrase_fallback_set"] = bool(
            _os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE")
        )
        status["hint"] = (
            "no key file. Run `iai-mcp crypto migrate-to-file` "
            "(existing Keychain key) or `iai-mcp crypto init` "
            "(fresh install), or set IAI_MCP_CRYPTO_PASSPHRASE."
        )

    print(_json.dumps(status, indent=2))
    return 0


def cmd_crypto_rotate(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.crypto import encrypt_field
    from iai_mcp.store import (
        EVENTS_TABLE,
        MemoryStore,
        RECORDS_TABLE,
        _uuid_literal,
    )

    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)

    decrypted_records = store.all_records()

    events_tbl = store.db.open_table(EVENTS_TABLE)
    events_df = events_tbl.to_pandas()
    decrypted_events: list[dict] = []
    from iai_mcp.crypto import decrypt_field, is_encrypted
    for _, row in events_df.iterrows():
        raw = row.get("data_json") or "{}"
        eid = str(row["id"])
        if is_encrypted(raw):
            try:
                raw = decrypt_field(
                    raw, store._key(), associated_data=eid.encode("ascii")
                )
            except (OSError, ValueError, RuntimeError):
                raw = "{}"
        decrypted_events.append({"id": eid, "data_json": raw})

    new_key = store._crypto_key_wrapper.rotate()
    store._crypto_key = new_key

    import json as _json_inner

    tbl = store.db.open_table(RECORDS_TABLE)
    record_count = 0
    rotate_failures: list[tuple[str, str]] = []
    for rec in decrypted_records:
        # Re-encrypt the encrypted columns in-place via a targeted SQL UPDATE.
        # This avoids all side effects of store.insert (pattern separation,
        # HNSW index rebuild, edge flush) and is purely a ciphertext swap.
        # If the UPDATE fails, the original ciphertext under the old key
        # remains intact on disk — the record is never lost.  On success,
        # the row already holds the new-key ciphertext and there is nothing
        # to delete.
        try:
            rid = str(rec.id)
            ad = store._ad(rec.id)
            literal_ct = encrypt_field(
                rec.literal_surface, new_key, associated_data=ad
            )
            provenance_plain = _json_inner.dumps(rec.provenance)
            provenance_ct = encrypt_field(
                provenance_plain, new_key, associated_data=ad
            )
            gain_plain = _json_inner.dumps(rec.profile_modulation_gain or {})
            gain_ct = encrypt_field(
                gain_plain, new_key, associated_data=ad
            )
            tbl.update(
                where=f"id = '{_uuid_literal(rec.id)}'",
                values={
                    "literal_surface": literal_ct,
                    "provenance_json": provenance_ct,
                    "profile_modulation_gain_json": gain_ct,
                },
            )
            record_count += 1
        except (OSError, ValueError, RuntimeError) as exc:
            # Re-encrypt failed.  The original row is still on disk under
            # the old key — do NOT delete it.  Record the failure so the
            # operator can see which records could not be rotated.
            rotate_failures.append((str(rec.id), str(exc)))

    event_count = 0
    for ev in decrypted_events:
        ad = ev["id"].encode("ascii")
        new_ct = encrypt_field(ev["data_json"], new_key, associated_data=ad)
        try:
            events_tbl.update(
                where=f"id = '{ev['id']}'",
                values={"data_json": new_ct},
            )
            event_count += 1
        except (OSError, ValueError, RuntimeError):
            continue

    result: dict[str, object] = {
        "status": "rotated" if not rotate_failures else "partial",
        "user_id": user_id,
        "records_re_encrypted": record_count,
        "events_re_encrypted": event_count,
        "algorithm": "AES-256-GCM",
        "format": "iai:enc:v1:",
    }
    if rotate_failures:
        result["records_failed"] = len(rotate_failures)
        result["failed_record_ids"] = [r[0] for r in rotate_failures]
        result["failure_details"] = [
            {"id": r[0], "error": r[1]} for r in rotate_failures
        ]
    print(_json.dumps(result, indent=2))
    try:
        from iai_mcp.crypto_key_watch import sync_crypto_key_watcher_to_disk
        from iai_mcp.events import write_event

        write_event(
            store,
            kind="crypto_key_rotated",
            data={
                "source": "cli_rotate",
                "records_re_encrypted": record_count,
                "events_re_encrypted": event_count,
                "records_failed": len(rotate_failures),
            },
            severity="info" if not rotate_failures else "error",
        )
        sync_crypto_key_watcher_to_disk(store)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("crypto rotate audit event failed: %s", exc)
    return 0


def cmd_crypto_recover_prior_key(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import json as _json

    from iai_mcp.crypto import KEY_BYTES
    from iai_mcp.migrate import migrate_crypto_recover_prior_key
    from iai_mcp.store import MemoryStore

    path: Path = args.prior_key_file
    try:
        prior = path.read_bytes()
    except OSError as exc:
        print(f"cannot read prior key file: {exc}", file=_cli.sys.stderr)
        return 1
    if len(prior) != KEY_BYTES:
        print(
            f"prior key file must be exactly {KEY_BYTES} bytes, got {len(prior)}",
            file=_cli.sys.stderr,
        )
        return 1
    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)
    try:
        out = migrate_crypto_recover_prior_key(
            store, prior, dry_run=bool(getattr(args, "dry_run", False)),
        )
    except Exception as exc:
        logger.error("crypto recover-prior-key failed: %s", exc)
        print(str(exc), file=_cli.sys.stderr)
        return 1
    print(_json.dumps(out, indent=2, default=str))
    return 0


def cmd_crypto_redact_undecryptable(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import json as _json

    from iai_mcp.migrate import migrate_redact_undecryptable_records
    from iai_mcp.store import MemoryStore

    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)
    try:
        out = migrate_redact_undecryptable_records(store)
    except Exception as exc:
        logger.error("crypto redact-undecryptable failed: %s", exc)
        print(str(exc), file=_cli.sys.stderr)
        return 1
    print(_json.dumps(out, indent=2, default=str))
    return 0


def cmd_crypto_migrate_to_file(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import base64 as _b64
    import keyring as _keyring
    import keyring.errors as _keyring_errors

    from iai_mcp.crypto import (
        CryptoKey,
        CryptoKeyError,
        KEY_BYTES,
        SERVICE_NAME_DEFAULT,
    )

    user_id = getattr(args, "user_id", None) or "default"
    keep_keychain = getattr(args, "keep_keychain", True)

    ck = CryptoKey(user_id=user_id)

    try:
        existing = ck._try_file_get()
    except CryptoKeyError as exc:
        print(
            f"refusing: existing key file is malformed: {exc}",
            file=_cli.sys.stderr,
        )
        return 1
    if existing is not None:
        print(f"already migrated: {ck._key_file_path()}")
        return 0

    try:
        encoded = _keyring.get_password(SERVICE_NAME_DEFAULT, user_id)
    except _keyring_errors.NoKeyringError:
        print(
            "no keyring backend available; nothing to migrate. "
            "If this is a fresh install, run `iai-mcp crypto init` instead.",
            file=_cli.sys.stderr,
        )
        return 1
    except _keyring_errors.KeyringError as exc:
        print(f"keyring read failed: {exc}", file=_cli.sys.stderr)
        return 1
    if encoded is None:
        print(
            f"no key found in keyring for user_id={user_id!r}. "
            f"If this is a fresh install, run `iai-mcp crypto init` instead.",
            file=_cli.sys.stderr,
        )
        return 1

    try:
        source = _b64.urlsafe_b64decode(encoded.encode("ascii"))
    except (ValueError, TypeError) as exc:
        print(f"keyring entry is malformed: {exc}", file=_cli.sys.stderr)
        return 1
    if len(source) != KEY_BYTES:
        print(
            f"keyring entry has wrong length {len(source)} (expected {KEY_BYTES})",
            file=_cli.sys.stderr,
        )
        return 1

    try:
        ck._try_file_set(source)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"failed to write key file: {exc}", file=_cli.sys.stderr)
        return 1

    try:
        roundtrip = ck._try_file_get()
    except CryptoKeyError as exc:
        try:
            ck._key_file_path().unlink()
        except OSError:
            pass
        print(f"round-trip verification failed: {exc}", file=_cli.sys.stderr)
        return 1
    if roundtrip != source:
        try:
            ck._key_file_path().unlink()
        except OSError:
            pass
        print(
            "round-trip verification failed: bytes differ", file=_cli.sys.stderr
        )
        return 1

    path = ck._key_file_path()
    print(f"migrated: {path} (mode 0o600, {KEY_BYTES} bytes)")

    if not keep_keychain:
        try:
            _keyring.delete_password(SERVICE_NAME_DEFAULT, user_id)
            print(f"deleted keyring entry for user_id={user_id!r}")
        except _keyring_errors.PasswordDeleteError:
            pass
        except _keyring_errors.KeyringError as exc:
            print(
                f"warning: failed to delete keyring entry: {exc}",
                file=_cli.sys.stderr,
            )
    else:
        print(
            "keyring entry kept (default). "
            "To remove manually, run "
            "`iai-mcp crypto migrate-to-file --delete-keychain` "
            "or use macOS Keychain Access.app."
        )

    return 0


def cmd_crypto_init(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import secrets as _secrets

    from iai_mcp.crypto import CryptoKey, KEY_BYTES

    user_id = getattr(args, "user_id", None) or "default"
    ck = CryptoKey(user_id=user_id)
    path = ck._key_file_path()
    if path.exists():
        print(
            f"refusing: key file already exists at {path}. "
            f"To rotate, run `iai-mcp crypto rotate`. "
            f"To wipe and start over, remove the file manually first.",
            file=_cli.sys.stderr,
        )
        return 1
    fresh = _secrets.token_bytes(KEY_BYTES)
    ck._try_file_set(fresh)
    print(f"created: {path} (mode 0o600, {KEY_BYTES} bytes)")
    return 0
