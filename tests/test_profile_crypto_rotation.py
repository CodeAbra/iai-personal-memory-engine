"""Durability of the encrypted profile blob across a key-file rotation and
the recover-prior-key path.

Assertions read the decrypted blob or the raw ``_hippo_meta`` row directly --
never through ``profile_get``, which returns a registry default and would
make "the blob rotated" indistinguishable from "the blob never existed".
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from iai_mcp.crypto import decrypt_field, is_encrypted
from iai_mcp.lilli.profile.persistence import (
    PROFILE_META_KEY,
    load_profile_state,
    save_profile_state,
    reencrypt_profile_blob,
)
from iai_mcp.migrate import migrate_crypto_recover_prior_key
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT


def _read_raw_meta(store: MemoryStore) -> "str | None":
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?", (PROFILE_META_KEY,)
        ).fetchone()
    return row["value"] if row is not None else None


def _minimal_record(literal: str, embedding: list[float] | None = None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * 384,
        structure_hv=b"\x00" * 1250,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )


# ---------------------------------------------------------------------------
# 1. Rotation round-trip -- pins and knobs decrypt under the new key.
# ---------------------------------------------------------------------------


def test_reencrypt_profile_blob_round_trips_through_a_rotation(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    knobs = {"literal_preservation": "loose", "monotropism_depth": {"alice": 0.7}}
    posterior = {"literal_preservation": {"alpha": 2.0, "beta": 1.0}}
    pins = {"literal_preservation": "2026-08-01T10:00:00+00:00"}
    assert save_profile_state(store, knobs=knobs, posterior=posterior, pins=pins) is True

    old_key = store._key()
    new_key = store._crypto_key_wrapper.rotate()
    store._crypto_key = new_key

    result = reencrypt_profile_blob(store, [old_key], new_key)
    assert result == "rotated"

    raw = _read_raw_meta(store)
    assert is_encrypted(raw)
    # the row must open under the NEW key -- old_key must no longer work
    try:
        decrypt_field(raw, old_key, associated_data=b"profile_state")
        old_key_still_opens = True
    except Exception:  # noqa: BLE001 -- any failure proves the old key is dead
        old_key_still_opens = False
    assert not old_key_still_opens, "re-encrypted blob must not open under the retired key"

    blob = load_profile_state(store)
    assert blob is not None
    assert blob["knobs"] == knobs
    assert blob["posterior"] == posterior
    assert blob["pins"] == pins


# ---------------------------------------------------------------------------
# 2. A blob no supplied key opens is left byte-identical and reported
#    stranded -- the ciphertext must never be overwritten with a placeholder.
# ---------------------------------------------------------------------------


def test_reencrypt_profile_blob_leaves_unopenable_blob_untouched(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    save_profile_state(
        store, knobs={"masking_off": True}, posterior={}, pins={}
    )
    raw_before = _read_raw_meta(store)
    assert is_encrypted(raw_before)

    # simulate the pre-fix gap: the key file rotates but the blob is never
    # re-encrypted, so it strands under the retired key
    new_key = store._crypto_key_wrapper.rotate()
    store._crypto_key = new_key

    unrelated_key = secrets.token_bytes(32)
    result = reencrypt_profile_blob(store, [unrelated_key], store._key())
    assert result == "stranded"

    raw_after = _read_raw_meta(store)
    assert raw_after == raw_before, "an unopenable blob must be byte-identical after the call"


def test_reencrypt_profile_blob_reports_absent_for_missing_row(tmp_path) -> None:
    store = MemoryStore(path=tmp_path)
    result = reencrypt_profile_blob(
        store, [secrets.token_bytes(32)], secrets.token_bytes(32)
    )
    assert result == "absent"
    assert _read_raw_meta(store) is None


# ---------------------------------------------------------------------------
# 3. The save-guard sidecars an unreadable blob instead of destroying it --
#    exercised from the rotation angle: the blob strands under a retired
#    key exactly as it would after an unrotated `cmd_crypto_rotate`.
# ---------------------------------------------------------------------------


def test_save_guard_sidecars_a_blob_stranded_by_a_rotation_gap(tmp_path) -> None:
    from iai_mcp.events import query_events
    from iai_mcp.lilli.profile.persistence import PROFILE_META_ORPHAN_KEY

    store = MemoryStore(path=tmp_path)
    knobs = {"literal_preservation": "loose"}
    save_profile_state(store, knobs=knobs, posterior={}, pins={})
    stranded_ct = _read_raw_meta(store)

    new_key = store._crypto_key_wrapper.rotate()
    store._crypto_key = new_key
    # the blob is left under the retired key -- the profile_state row is
    # now unreadable under the current key, matching the pre-fix gap

    assert (
        save_profile_state(store, knobs={"masking_off": True}, posterior={}, pins={})
        is True
    )

    with store.db._conn_lock:
        orphan_row = store.db._conn.execute(
            "SELECT value FROM _hippo_meta WHERE key = ?",
            (PROFILE_META_ORPHAN_KEY,),
        ).fetchone()
    assert orphan_row is not None
    assert orphan_row["value"] == stranded_ct, "the stranded ciphertext must be preserved, not destroyed"

    events = query_events(store, kind="profile_state_unreadable", limit=10)
    assert events
    assert events[-1]["data"]["reason"] == "preserved_before_overwrite"

    blob = load_profile_state(store)
    assert blob is not None
    assert blob["knobs"]["masking_off"] is True


# ---------------------------------------------------------------------------
# 4. The recover-prior-key path re-keys the blob on the real
#    needs_prior == 0 scenario: records already decrypt under the current
#    key, only the profile blob strands under a retained generation.
# ---------------------------------------------------------------------------


def test_recover_prior_key_rekeys_profile_blob_on_needs_prior_zero(tmp_path: Path) -> None:
    root = tmp_path / "recover-profile"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)

    store_a = MemoryStore(path=root, user_id="default")
    knobs = {"literal_preservation": "loose"}
    pins = {"literal_preservation": "2026-08-01T10:00:00+00:00"}
    save_profile_state(store_a, knobs=knobs, posterior={}, pins=pins)
    del store_a

    key_b = secrets.token_bytes(32)
    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")

    # a record written under the NEW key file -- it is already current, so
    # the record-level pre-check finds nothing needing the prior key
    rec = _minimal_record("alice-recover-control")
    store_b.insert(rec)
    flush_record_buffer(store_b)

    with store_b.db.ro_conn() as conn:
        (row_count,) = conn.execute("SELECT COUNT(*) FROM records").fetchone()
    assert row_count == 1

    out = migrate_crypto_recover_prior_key(store_b, [key_a], dry_run=False)

    assert out.get("no_op") is True, (
        "this must hit the needs_prior == 0 early return -- the real "
        "recover scenario the profile re-key exists to serve"
    )
    assert out.get("reason") == "all_rows_decrypt_with_current_key"
    assert out.get("profile_state") == "rotated"

    blob = load_profile_state(store_b)
    assert blob is not None
    assert blob["knobs"] == knobs
    assert blob["pins"] == pins


# ---------------------------------------------------------------------------
# 5-6. Wiring: a full `cmd_crypto_rotate` run must actually call the helper
#    -- deleting the wiring (not just the helper) must fail these.
# ---------------------------------------------------------------------------


def test_cmd_crypto_rotate_carries_profile_blob_to_new_key(tmp_path, monkeypatch, capsys) -> None:
    import argparse
    import json as _json

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_rotate

    store = MemoryStore()
    knobs = {"literal_preservation": "loose"}
    pins = {"literal_preservation": "2026-08-01T10:00:00+00:00"}
    save_profile_state(store, knobs=knobs, posterior={}, pins=pins)
    store.insert(_minimal_record("alice-cli-rotate-control"))

    exit_code = cmd_crypto_rotate(argparse.Namespace(user_id="default"))
    out = capsys.readouterr().out
    assert exit_code == 0
    report = _json.loads(out)
    assert report["profile_state_rotated"] == "rotated"

    backup = tmp_path / ".crypto.key.pre-rotate"
    assert not backup.exists(), "a fully clean rotation must not keep the sidecar"

    blob = load_profile_state(MemoryStore())
    assert blob is not None
    assert blob["knobs"] == knobs
    assert blob["pins"] == pins


def test_cmd_crypto_rotate_reports_stranded_profile_and_keeps_sidecar(
    tmp_path, monkeypatch, capsys
) -> None:
    import argparse
    import json as _json

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    key_a = secrets.token_bytes(32)
    key_path.write_bytes(key_a)
    os.chmod(key_path, 0o600)

    store_a = MemoryStore()
    save_profile_state(store_a, knobs={"literal_preservation": "loose"}, posterior={}, pins={})
    raw_before = _read_raw_meta(store_a)
    del store_a

    # the key file rotates to a fresh generation WITHOUT re-encrypting the
    # blob -- an untracked prior generation the sidecar never retained,
    # reproducing a blob stranded from an earlier rotation
    key_b = secrets.token_bytes(32)
    key_path.write_bytes(key_b)
    os.chmod(key_path, 0o600)
    store_b = MemoryStore()
    store_b.insert(_minimal_record("alice-cli-rotate-stranded-control"))
    del store_b

    from iai_mcp.cli import cmd_crypto_rotate

    exit_code = cmd_crypto_rotate(argparse.Namespace(user_id="default"))
    out = capsys.readouterr().out
    assert exit_code == 0
    report = _json.loads(out)
    assert report["profile_state_rotated"] == "stranded"
    assert report["status"] == "partial"

    backup = tmp_path / ".crypto.key.pre-rotate"
    assert backup.exists(), "a stranded profile blob must keep the pre-rotate sidecar"

    raw_after = _read_raw_meta(MemoryStore())
    assert raw_after == raw_before, "the stranded blob must be byte-identical after the rotation"
