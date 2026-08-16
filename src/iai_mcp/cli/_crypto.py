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
        status["mode"] = mode_octal
        status["mode_secure"] = (st.st_mode & 0o077 == 0)
        status["uid"] = st.st_uid
        status["uid_matches_process"] = (st.st_uid == _os.geteuid())
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

    sidecar = path.with_name(".crypto.key.pre-rotate")
    if sidecar.exists():
        st = sidecar.stat()
        status["pre_rotate_key_retained"] = {
            "path": str(sidecar),
            "generations": st.st_size // KEY_BYTES,
            "mode": f"0o{st.st_mode & 0o777:03o}",
            "hint": (
                "an earlier rotation ended partial; re-run "
                "`iai-mcp crypto rotate` (or `crypto recover-prior-key` "
                "for stranded records) to finish and drop this file"
            ),
        }

    print(_json.dumps(status, indent=2))
    return 0


def _rotate_one_spool_file(
    path: Path, old_keys: "list[bytes]", new_key: bytes
) -> int:
    import os as _os

    from iai_mcp.capture import _SPOOL_AAD
    from iai_mcp.crypto import decrypt_field, encrypt_field, is_encrypted

    def _decrypt_any(s: str) -> "str | None":
        for k in old_keys:
            try:
                return decrypt_field(s, k, _SPOOL_AAD)
            except Exception:  # noqa: BLE001 -- try the next retained generation
                continue
        return None

    for _attempt in range(3):
        pre = path.stat()
        raw = path.read_bytes()
        # Byte-level line handling: a corrupted (e.g. quarantined) line that
        # is not valid UTF-8 must not stop the readable lines around it from
        # re-keying — it is preserved byte-identical instead.
        parts = raw.split(b"\n")
        torn_tail = parts.pop()
        out: list[bytes] = []
        changed = 0
        for ln_b in parts:
            try:
                s = ln_b.decode("utf-8").strip()
            except UnicodeDecodeError:
                out.append(ln_b + b"\n")
                continue
            if s and is_encrypted(s):
                plain = _decrypt_any(s)
                if plain is None:
                    # Unreadable under every retained generation; keep as-is.
                    out.append(ln_b + b"\n")
                    continue
                out.append(
                    encrypt_field(plain, new_key, _SPOOL_AAD).encode("utf-8")
                    + b"\n"
                )
                changed += 1
            else:
                out.append(ln_b + b"\n")
        if torn_tail:
            # Bytes after the last newline — a concurrent append in flight;
            # preserved verbatim.
            out.append(torn_tail)
        if not changed:
            return 0
        tmp = path.with_name(path.name + ".rotate-tmp")
        flags = (
            _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC
            | getattr(_os, "O_BINARY", 0)
        )
        fd = _os.open(str(tmp), flags, 0o600)
        try:
            _os.write(fd, b"".join(out))
            _os.fsync(fd)
        finally:
            _os.close(fd)
        # Fence: a hook may append between the read and this swap; replacing
        # then would drop that turn. Re-stat and retry on any change.
        post = path.stat()
        if (
            post.st_size == pre.st_size
            and post.st_mtime_ns == pre.st_mtime_ns
        ):
            _os.replace(str(tmp), str(path))
            return changed
        try:
            tmp.unlink()
        except OSError:
            pass
    raise RuntimeError("file kept changing during rotation; re-run rotate")


def _torn_tail_may_need_old_key(path: Path) -> bool:
    from iai_mcp.crypto import CIPHERTEXT_PREFIX

    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > 65536:
                fh.seek(size - 65536)
            chunk = fh.read()
    except OSError:
        return True
    if not chunk:
        return False
    if b"\n" not in chunk:
        if size > len(chunk):
            # A single unterminated line longer than the window — cannot
            # classify, keep the generation.
            return True
        tail = chunk
    else:
        tail = chunk.rsplit(b"\n", 1)[-1]
    if not tail:
        return False
    try:
        s = tail.decode("utf-8")
    except UnicodeDecodeError:
        return True
    # A truncated ciphertext line is either prefixed by the marker already
    # or is a leading fragment of the marker itself.
    return s.startswith(CIPHERTEXT_PREFIX) or CIPHERTEXT_PREFIX.startswith(s)


def _rotate_spool_lines(old_keys: "list[bytes]", new_key: bytes, store) -> dict:
    from iai_mcp.capture import _spool_root, deferred_captures_dir

    report: dict = {
        "spool_files_re_encrypted": 0,
        "spool_lines_re_encrypted": 0,
        "spool_files_skipped": [],
        "quarantine_files_skipped": [],
    }
    try:
        store_root = Path(store.root).resolve()
    except Exception:  # noqa: BLE001 -- no resolvable root means no spool to rotate
        return report
    # Only the default home store carries the spool (and its key).
    if store_root != _spool_root().resolve():
        return report
    spool_dir = deferred_captures_dir()
    if not spool_dir.exists():
        return report

    def _iter_files(d: Path) -> "list[Path]":
        # The daemon's spool housekeeping may remove entries concurrently;
        # a vanished directory must not abort the rotation post-key-swap.
        try:
            return list(d.iterdir())
        except OSError:
            return []

    # Quarantined copies hold the same ciphertext generations as the live
    # spool — leaving them out would make them permanently unreadable once
    # a clean rotation drops the retained key. Their failures land in a
    # separate bucket: a broken file inside the broken-file directory must
    # never pin the key lifecycle.
    live = [(p, "spool_files_skipped") for p in _iter_files(spool_dir)]
    quarantine = spool_dir / ".quarantine"
    q = [(p, "quarantine_files_skipped") for p in _iter_files(quarantine)]
    for path, bucket in sorted(live + q):
        if not path.is_file():
            continue
        if path.name.endswith(".rotate-tmp") or path.suffix == ".tmp":
            continue
        try:
            rel = str(path.relative_to(spool_dir))
        except ValueError:
            rel = str(path)
        try:
            changed = _rotate_one_spool_file(path, old_keys, new_key)
        except Exception as exc:  # noqa: BLE001 -- per-file isolation; report, never abort
            report[bucket].append(f"{rel}: {exc}")
            continue
        if changed:
            report["spool_files_re_encrypted"] += 1
            report["spool_lines_re_encrypted"] += changed
        if _torn_tail_may_need_old_key(path):
            # Bytes after the last newline cannot be re-keyed until the
            # append completes — if they are (partial) ciphertext, dropping
            # the outgoing generation would strand that queued turn.
            report[bucket].append(
                f"{rel}: torn tail may hold ciphertext under the outgoing key"
            )
    return report


def cmd_crypto_rotate(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import json as _json

    from iai_mcp.cli._maintenance import (
        _maintenance_compact_preflight_daemon_alive,
    )
    from iai_mcp.crypto import encrypt_field
    from iai_mcp.store import (
        EVENTS_TABLE,
        MemoryStore,
        RECORDS_TABLE,
        _uuid_literal,
    )

    # Rows written by a live daemon between the snapshots below and the key
    # swap would carry the outgoing generation without being re-encrypted.
    # The daemon serves the HOME store only, so the check applies only there
    # — an env-selected store must not be refused by an unrelated daemon.
    from iai_mcp.capture import _spool_root

    import os as _os
    # An empty IAI_MCP_STORE is falsy to MemoryStore's resolver — the
    # preflight must classify it as the home store the same way.
    env_root = _os.environ.get("IAI_MCP_STORE")
    rotating_home_store = (
        not env_root
        or Path(env_root).resolve() == _spool_root().resolve()
    )
    if rotating_home_store:
        refusal = _maintenance_compact_preflight_daemon_alive()
        if refusal:
            print(f"refusing to rotate: {refusal}", file=_cli.sys.stderr)
            return 1

    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)

    # Rotation is defined for key-FILE stores only. A passphrase-derived
    # key has exactly one generation (the passphrase itself is the recovery
    # material), and rotate() would silently materialise a key file that
    # shadows the passphrase — an unrecoverable posture change.
    key_path = store._crypto_key_wrapper._key_file_path()
    if not key_path.exists():
        print(
            "refusing to rotate: no key file at "
            f"{key_path}. A passphrase-derived key does not rotate (change "
            "the passphrase means a new store); a keyring-backed store must "
            "run `iai-mcp crypto migrate-to-file` first; a fresh install "
            "creates its key with `iai-mcp crypto init`.",
            file=_cli.sys.stderr,
        )
        return 1

    # all_records() decrypts with the CURRENT key and raises a named
    # HippoIntegrityError on a foreign-generation record — the rotation
    # aborts here, before anything is written, with the recover-prior-key
    # runbook in the message.
    decrypted_records = store.all_records()

    old_key = store._key()

    # Retain the outgoing key at 0600 until every ciphertext store has
    # re-encrypted: any crash or unrotatable file mid-rotation leaves data
    # only this backup can still open. It is deleted only when no row still
    # depends on an old generation.
    from iai_mcp.crypto import KEY_BYTES as _KEY_BYTES
    from iai_mcp.crypto import write_key_material as _write_key_material

    backup_path = key_path.with_name(".crypto.key.pre-rotate")
    # A prior partial rotation may have retained older generations; keep
    # them (concatenated KEY_BYTES blocks) until a fully clean rotation.
    prior_keys: list[bytes] = []
    if backup_path.exists():
        raw_backup = backup_path.read_bytes()
        prior_keys = [
            raw_backup[i:i + _KEY_BYTES]
            for i in range(0, len(raw_backup) - len(raw_backup) % _KEY_BYTES,
                           _KEY_BYTES)
        ]
    old_keys = [old_key] + [k for k in prior_keys if k != old_key]
    retained = prior_keys + ([old_key] if old_key not in prior_keys else [])
    _write_key_material(backup_path, b"".join(retained))

    # Events decrypt trying every retained generation, so an event stranded
    # by an earlier partial rotation is healed here. One that no retained
    # generation opens is left under its original ciphertext — never
    # overwritten with a placeholder.
    events_tbl = store.db.open_table(EVENTS_TABLE)
    events_df = events_tbl.to_pandas()
    decrypted_events: list[dict] = []
    undecryptable_event_ids: set[str] = set()
    from iai_mcp.crypto import decrypt_field, is_encrypted
    for _, row in events_df.iterrows():
        raw = row.get("data_json") or "{}"
        eid = str(row["id"])
        if is_encrypted(raw):
            plain = None
            for k in old_keys:
                try:
                    plain = decrypt_field(
                        raw, k, associated_data=eid.encode("ascii")
                    )
                    break
                except Exception:  # noqa: BLE001 -- try the next retained generation
                    continue
            if plain is None:
                undecryptable_event_ids.add(eid)
                continue
            raw = plain
        decrypted_events.append({"id": eid, "data_json": raw})
    events_undecryptable = len(undecryptable_event_ids)

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
    event_update_failures = 0
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
            event_update_failures += 1
            continue

    # The deferred spool encrypts under the same home-store key file;
    # rotating it without re-encrypting the spool would strand every
    # queued turn as unreadable ciphertext.
    spool_report = _rotate_spool_lines(old_keys, new_key, store)

    from iai_mcp.lilli.profile.persistence import reencrypt_profile_blob

    profile_rotate = reencrypt_profile_blob(store, old_keys, new_key)

    # Retention is keyed to NEED: the sidecar stays only while some row the
    # retained generations can still OPEN remains un-rotated. Undecryptable
    # events open under no retained generation, so keeping key material for
    # them would retain it forever for nothing — their exit is
    # `crypto redact-undecryptable`.
    old_generation_still_needed = bool(
        rotate_failures
        or spool_report["spool_files_skipped"]
        or event_update_failures
        or profile_rotate == "stranded"
    )

    # The record/event snapshots above predate the key swap; a row written
    # in between carries the outgoing generation. Re-scan before dropping it.
    late_records = 0
    late_events = 0
    if not old_generation_still_needed:
        try:
            store.all_records()
        except Exception:  # noqa: BLE001 -- cannot prove the records are clean; keep the sidecar
            late_records = 1
        try:
            verify_df = events_tbl.to_pandas()
            for _, row in verify_df.iterrows():
                raw = row.get("data_json") or "{}"
                eid = str(row["id"])
                if not is_encrypted(raw) or eid in undecryptable_event_ids:
                    continue
                try:
                    decrypt_field(
                        raw, new_key, associated_data=eid.encode("ascii")
                    )
                except Exception:  # noqa: BLE001 -- any failure means the row is not under the new key
                    late_events += 1
        except Exception:  # noqa: BLE001 -- cannot prove the events are clean; keep the sidecar
            late_events += 1
    old_generation_still_needed = bool(
        old_generation_still_needed or late_records or late_events
    )

    if not old_generation_still_needed:
        try:
            backup_path.unlink()
        except OSError:
            pass

    fully_clean = not old_generation_still_needed and events_undecryptable == 0

    result: dict[str, object] = {
        "status": "rotated" if fully_clean else "partial",
        "user_id": user_id,
        "records_re_encrypted": record_count,
        "events_re_encrypted": event_count,
        "spool_files_re_encrypted": spool_report["spool_files_re_encrypted"],
        "spool_lines_re_encrypted": spool_report["spool_lines_re_encrypted"],
        "algorithm": "AES-256-GCM",
        "format": "iai:enc:v1:",
        "profile_state_rotated": profile_rotate,
    }
    if events_undecryptable:
        result["events_undecryptable"] = events_undecryptable
    if event_update_failures:
        result["events_update_failed"] = event_update_failures
    if late_records:
        result["late_records_detected"] = late_records
    if late_events:
        result["late_events_detected"] = late_events
    if not fully_clean:
        if old_generation_still_needed:
            # The un-rotated remainder is still openable with the retained key.
            result["pre_rotate_key_retained"] = str(backup_path)
        if rotate_failures or late_records or profile_rotate == "stranded":
            # A record under an old generation aborts the next rotate at
            # all_records(); the records must re-key FIRST.
            recovery = (
                "stop the daemon and active sessions, run `iai-mcp crypto "
                f"recover-prior-key --prior-key-file {backup_path}` to re-key "
                "the stranded records (the file may hold several retained "
                "generations; each is tried), then re-run "
                "`iai-mcp crypto rotate`"
            )
            if events_undecryptable:
                recovery += (
                    f"; {events_undecryptable} event row(s) additionally "
                    "decrypt under no available generation — clear them with "
                    "`iai-mcp crypto redact-undecryptable` once the records "
                    "are recovered"
                )
            result["recovery"] = recovery
        elif (
            spool_report["spool_files_skipped"]
            or event_update_failures
            or late_events
        ):
            recovery = (
                "stop the daemon and active sessions, then re-run "
                "`iai-mcp crypto rotate` — the retained generations open "
                "everything that did not rotate; the retained key file is "
                "removed on the next fully clean rotation"
            )
            if any(
                "torn tail" in s
                for s in spool_report["spool_files_skipped"]
            ):
                recovery += (
                    "; a torn-tail entry clears only when its file's "
                    "interrupted append completes or the file is removed — "
                    "re-running alone does not, and the retained key keeps "
                    "it readable until then"
                )
            result["recovery"] = recovery
        else:
            result["recovery"] = (
                f"{events_undecryptable} event row(s) decrypt under no "
                "available key generation; their payloads are unrecoverable. "
                "Run `iai-mcp crypto redact-undecryptable` to replace them "
                "with a redaction marker under the current key, after which "
                "rotate reports clean."
            )
    if spool_report["spool_files_skipped"]:
        result["spool_files_skipped"] = spool_report["spool_files_skipped"]
    if spool_report["quarantine_files_skipped"]:
        result["quarantine_files_skipped"] = (
            spool_report["quarantine_files_skipped"]
        )
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
                "status": result["status"],
                "records_re_encrypted": record_count,
                "events_re_encrypted": event_count,
                "records_failed": len(rotate_failures),
                "events_undecryptable": events_undecryptable,
                "events_update_failed": event_update_failures,
                "spool_files_re_encrypted": spool_report[
                    "spool_files_re_encrypted"
                ],
                "spool_files_skipped": len(
                    spool_report["spool_files_skipped"]
                ),
                "quarantine_files_skipped": len(
                    spool_report["quarantine_files_skipped"]
                ),
            },
            severity="info" if fully_clean else "error",
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
    # One key, or a `.crypto.key.pre-rotate` sidecar holding several
    # retained generations as concatenated KEY_BYTES blocks.
    if len(prior) == 0 or len(prior) % KEY_BYTES != 0:
        print(
            f"prior key file must be a multiple of {KEY_BYTES} bytes "
            f"(one block per retained generation), got {len(prior)}",
            file=_cli.sys.stderr,
        )
        return 1
    generations = [
        prior[i:i + KEY_BYTES] for i in range(0, len(prior), KEY_BYTES)
    ]
    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)
    try:
        out = migrate_crypto_recover_prior_key(
            store, generations, dry_run=bool(getattr(args, "dry_run", False)),
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

    from iai_mcp.migrate import (
        migrate_redact_undecryptable_events,
        migrate_redact_undecryptable_records,
    )
    from iai_mcp.store import MemoryStore

    user_id = getattr(args, "user_id", None) or "default"
    store = MemoryStore(user_id=user_id)
    try:
        out = migrate_redact_undecryptable_records(store)
        out.update(migrate_redact_undecryptable_events(store))
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
