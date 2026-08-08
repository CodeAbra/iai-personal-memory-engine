
from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.migrate import migrate_crypto_recover_prior_key
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT


def _minimal_record(literal: str) -> MemoryRecord:
    rid = uuid4()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=literal,
        aaak_index="",
        embedding=[0.01] * 384,
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


def test_recover_prior_key_atomic_swap_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    key_b = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)

    store_a = MemoryStore(path=root, user_id="default")
    rec = _minimal_record("verbatim-prior-key-recover")
    store_a.insert(rec)
    rid = rec.id
    del store_a

    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")
    # A wrong-generation record surfaces as a NAMED integrity error carrying
    # the recover-prior-key runbook — never a bare empty-message InvalidTag.
    from iai_mcp.hippo import HippoIntegrityError

    with pytest.raises(HippoIntegrityError, match="recover-prior-key"):
        store_b.get(rid)

    out = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=False)
    assert out.get("no_op") is False
    assert out.get("records_staged") == 1
    assert out.get("rows_needed_prior_key") == 1

    got = store_b.get(rid)
    assert got is not None
    assert got.literal_surface == "verbatim-prior-key-recover"

    out2 = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=False)
    assert out2.get("no_op") is True
    assert out2.get("reason") == "all_rows_decrypt_with_current_key"


def test_recover_prior_key_dry_run_counts(tmp_path: Path) -> None:
    root = tmp_path / "store2"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    key_b = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)
    store_a = MemoryStore(path=root, user_id="default")
    store_a.insert(_minimal_record("dry-run-count"))
    del store_a
    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")
    out = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=True)
    assert out.get("dry_run") is True
    assert out.get("would_stage") == 1
    assert out.get("rows_needing_prior_key") == 1
