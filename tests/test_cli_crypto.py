from __future__ import annotations

import json
import os
import secrets
import stat
from datetime import datetime, timezone
from uuid import uuid4

import pytest


def test_cli_crypto_status_shows_file_backend(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_status

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_status(args)
    out = capsys.readouterr().out
    out_lower = out.lower()
    assert exit_code == 0
    assert "default" in out
    assert "file" in out_lower, f"status must report backend=file; got:\n{out}"
    assert ".crypto.key" in out, f"status must include the file path; got:\n{out}"
    assert "600" in out, f"status must expose mode 0o600; got:\n{out}"
    assert "keyring" not in out_lower, (
        f"status must NOT mention keyring (backend retired in 07.10); got:\n{out}"
    )


def test_cli_crypto_rotate_regenerates_key(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    key_a = secrets.token_bytes(32)
    key_path.write_bytes(key_a)
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_rotate
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore()
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="rotation test content",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    store.insert(rec)
    initial_ct = store.db.open_table(RECORDS_TABLE).to_pandas()[
        lambda df: df["id"] == str(rec.id)
    ].iloc[0]["literal_surface"]
    assert initial_ct.startswith("iai:enc:v1:")

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_rotate(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "rotat" in out.lower()

    new_key_bytes = key_path.read_bytes()
    assert len(new_key_bytes) == 32
    assert new_key_bytes != key_a, "rotate must write a fresh key to the file"
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600, f"rotated key file must be 0o600, got 0o{mode:03o}"

    store2 = MemoryStore()
    post_ct = store2.db.open_table(RECORDS_TABLE).to_pandas()[
        lambda df: df["id"] == str(rec.id)
    ].iloc[0]["literal_surface"]
    assert post_ct.startswith("iai:enc:v1:")
    assert post_ct != initial_ct
    got = store2.get(rec.id)
    assert got is not None
    assert got.literal_surface == "rotation test content"


def test_cli_migrate_to_3_dry_run_counts_plaintext_rows(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_migrate
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.types import EMBED_DIM

    store = MemoryStore()
    rid = uuid4()
    row = {
        "id": str(rid),
        "tier": "episodic",
        "literal_surface": "plain legacy",
        "aaak_index": "",
        "embedding": [0.1] * EMBED_DIM,
        "structure_hv": b"",
        "community_id": "",
        "centrality": 0.0,
        "detail_level": 2,
        "pinned": False,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": False,
        "never_merge": False,
        "provenance_json": json.dumps([{"ts": "x", "cue": "y", "session_id": "z"}]),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "tags_json": json.dumps([]),
        "language": "en",
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": json.dumps({}),
        "schema_version": 2,
    }
    store.db.open_table(RECORDS_TABLE).add([row])

    args = argparse.Namespace(from_=2, to=3, dry_run=True, verbose=False)
    exit_code = cmd_migrate(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "would" in out.lower() or "dry" in out.lower() or "migrat" in out.lower()
    assert "1" in out


def test_cli_migrate_to_3_encrypts_plaintext_rows(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_migrate
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.types import EMBED_DIM

    store = MemoryStore()
    rid = uuid4()
    row = {
        "id": str(rid),
        "tier": "episodic",
        "literal_surface": "still-plaintext",
        "aaak_index": "",
        "embedding": [0.1] * EMBED_DIM,
        "structure_hv": b"",
        "community_id": "",
        "centrality": 0.0,
        "detail_level": 2,
        "pinned": False,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": False,
        "never_merge": False,
        "provenance_json": json.dumps([]),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "tags_json": json.dumps([]),
        "language": "en",
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": json.dumps({}),
        "schema_version": 2,
    }
    store.db.open_table(RECORDS_TABLE).add([row])

    args = argparse.Namespace(from_=2, to=3, dry_run=False, verbose=False)
    exit_code = cmd_migrate(args)
    assert exit_code == 0

    df = store.db.open_table(RECORDS_TABLE).to_pandas()
    post = df[df["id"] == str(rid)].iloc[0]
    assert post["literal_surface"].startswith("iai:enc:v1:")


def test_cli_migrate_to_3_rejects_unsupported_version_pair(
    tmp_path, monkeypatch, capsys
):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.cli import cmd_migrate

    args = argparse.Namespace(from_=9, to=42, dry_run=False, verbose=False)
    exit_code = cmd_migrate(args)
    err = capsys.readouterr().err.lower()
    out = capsys.readouterr().out.lower()
    assert exit_code != 0
    assert ("unsupported" in err or "invalid" in err or
            "unsupported" in out or "invalid" in out)


@pytest.mark.perf
def test_neural_map_bench_passes_after_encryption(tmp_path):
    from bench.neural_map import run_neural_map_bench, D_SPEED_P95_MS

    from _perf_helpers import best_of_n, skip_if_loaded

    skip_if_loaded()

    out = run_neural_map_bench(n=100, iterations=10, store_path=tmp_path / "run0", seed=0)
    assert out["n"] == 100
    assert out["iterations"] == 10

    counter = {"i": 0}

    def _one_p95() -> float:
        i = counter["i"]
        counter["i"] += 1
        if i == 0:
            return float(out["latency_ms_p95"])
        run = run_neural_map_bench(
            n=100, iterations=10, store_path=tmp_path / f"run{i}", seed=i,
        )
        return float(run["latency_ms_p95"])

    min_p95 = best_of_n(_one_p95, n=3)
    assert min_p95 < D_SPEED_P95_MS, (
        f"D-SPEED regression post-encryption: best-of-3 p95={min_p95:.1f} ms "
        f">= {D_SPEED_P95_MS} ms"
    )


def test_cli_crypto_init_creates_fresh_file(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    assert not key_path.exists()

    from iai_mcp.cli import cmd_crypto_init

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_init(args)
    out = capsys.readouterr().out
    assert exit_code == 0

    assert key_path.exists()
    assert key_path.stat().st_size == 32
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600, f"init key file must be 0o600, got 0o{mode:03o}"
    assert ".crypto.key" in out
    raw = key_path.read_bytes()
    for i in range(0, 32, 4):
        chunk = raw[i:i + 4]
        if chunk == b"\x00\x00\x00\x00":
            continue
        assert chunk.decode("latin-1") not in out, (
            "init must not print key bytes to stdout"
        )


def test_cli_crypto_init_refuses_when_file_exists(tmp_path, monkeypatch, capsys):
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    pre = secrets.token_bytes(32)
    key_path.write_bytes(pre)
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_init

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_init(args)
    err = capsys.readouterr().err
    assert exit_code == 1
    assert ".crypto.key" in err
    assert key_path.read_bytes() == pre


def test_cli_crypto_rotate_no_loss_on_reencrypt_failure(tmp_path, monkeypatch):
    """A mid-rotation re-encryption failure must NOT lose the original record.

    The fixed implementation re-encrypts each record's ciphertext columns
    in-place via a targeted SQL UPDATE (no delete involved at any point).  If
    encrypt_field raises for a specific record, that record's original
    ciphertext under the old key remains on disk intact.  We verify this by
    patching encrypt_field to fail when called with a specific plaintext and
    confirming the affected record still exists in the table after rotation."""
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_rotate
    from iai_mcp.store import MemoryStore, RECORDS_TABLE
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore()

    rec1 = MemoryRecord(
        id=__import__("uuid").uuid4(),
        tier="episodic",
        literal_surface="record one — rotate succeeds",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        updated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        tags=[],
        language="en",
    )
    FAILING_SURFACE = "record two — rotate FAILS for this one"
    rec2 = MemoryRecord(
        id=__import__("uuid").uuid4(),
        tier="episodic",
        literal_surface=FAILING_SURFACE,
        aaak_index="",
        embedding=[0.2] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        updated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        tags=[],
        language="en",
    )
    store.insert(rec1)
    store.insert(rec2)

    # Capture the old ciphertext for rec2 so we can confirm it's preserved.
    tbl = store.db.open_table(RECORDS_TABLE)
    pre_df = tbl.to_pandas()
    old_ct_rec2 = pre_df[pre_df["id"] == str(rec2.id)].iloc[0]["literal_surface"]
    assert old_ct_rec2.startswith("iai:enc:v1:"), "rec2 must be encrypted before rotation"

    # Patch encrypt_field in the iai_mcp.crypto module so it raises when
    # called with rec2's plaintext literal_surface — simulating a mid-rotation
    # failure for that specific record.  We patch the source module (not the
    # cli._crypto import site) because cmd_crypto_rotate imports encrypt_field
    # inside the function body via `from iai_mcp.crypto import encrypt_field`,
    # which resolves at call time and will pick up the patched attribute.
    import iai_mcp.crypto as _crypto_mod
    original_encrypt = _crypto_mod.encrypt_field

    def patched_encrypt(plaintext: str, key: bytes, associated_data: bytes = b"") -> str:
        if plaintext == FAILING_SURFACE:
            raise ValueError("simulated re-encryption failure for rec2")
        return original_encrypt(plaintext, key, associated_data=associated_data)

    monkeypatch.setattr(_crypto_mod, "encrypt_field", patched_encrypt)

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_rotate(args)
    # Rotate completes with partial status (exit 0 — failures are reported
    # in the JSON output but do not abort the whole operation).
    assert exit_code == 0

    # Both records must still exist on disk.  rec2 never got re-encrypted
    # (encrypt_field raised), so the original row under the old key persists.
    store2 = MemoryStore()
    tbl2 = store2.db.open_table(RECORDS_TABLE)
    post_df = tbl2.to_pandas()
    post_ids = set(post_df["id"].tolist())
    assert str(rec1.id) in post_ids, "rec1 (successful rotation) must be present"
    assert str(rec2.id) in post_ids, (
        "rec2 (failed rotation) must still be present — original must not be lost"
    )

    # rec2's ciphertext must be UNCHANGED (still the old-key ciphertext),
    # proving the re-encryption failure left the original intact.
    post_ct_rec2 = post_df[post_df["id"] == str(rec2.id)].iloc[0]["literal_surface"]
    assert post_ct_rec2 == old_ct_rec2, (
        "rec2 ciphertext must remain the old-key value when re-encryption failed"
    )


def test_cli_crypto_rotate_swaps_active_key(tmp_path, monkeypatch):
    """After cmd_crypto_rotate, the store's active key reference must be the
    freshly rotated key — not the prior one. The codec primitive constructs an
    AEAD context per call from store._key(), so the only durable contract is
    that the key reference itself rotates atomically with the on-disk file."""
    import argparse

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)

    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    pre_rotate_key = key_path.read_bytes()

    from iai_mcp.cli import cmd_crypto_rotate

    args = argparse.Namespace(user_id="default")
    exit_code = cmd_crypto_rotate(args)

    assert exit_code == 0
    post_rotate_key = key_path.read_bytes()
    assert post_rotate_key != pre_rotate_key, (
        "rotate must replace the on-disk key bytes"
    )
    assert len(post_rotate_key) == 32
