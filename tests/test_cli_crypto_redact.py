
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


from iai_mcp.migrate import migrate_redact_undecryptable_records
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
        embedding=[0.02] * 384,
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
        tags=["t1"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )


def test_redact_makes_literal_decryptable_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "redact-store"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    key_b = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)
    store_a = MemoryStore(path=root, user_id="default")
    rec = _minimal_record("secret-surface")
    store_a.insert(rec)
    rid = rec.id
    del store_a

    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")
    out = migrate_redact_undecryptable_records(store_b)
    assert out["redacted"] == 1
    assert out["skipped_plain"] == 0

    got = store_b.get(rid)
    assert got is not None
    assert got.literal_surface.startswith("<REDACTED:")

    out2 = migrate_redact_undecryptable_records(store_b)
    assert out2["redacted"] == 0
    assert out2["skipped_ok"] >= 1


def test_cli_crypto_redact_undecryptable_smoke(tmp_path: Path) -> None:
    root = tmp_path / "cli-redact"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    key_b = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)
    store_a = MemoryStore(path=root, user_id="default")
    store_a.insert(_minimal_record("cli-redact-body"))
    store_a.close()
    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)

    env = {**os.environ, "IAI_MCP_STORE": str(root.resolve())}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.cli",
            "crypto",
            "redact-undecryptable",
            "--user-id",
            "default",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout.strip())
    assert payload.get("redacted") == 1
