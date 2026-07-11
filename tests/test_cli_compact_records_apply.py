from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from iai_mcp.store import MemoryStore
from test_capture_dedup_contract import _make_record


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


def test_apply_runs_to_completion_against_real_lancedb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    store_path = tmp_path
    hippo_dir = store_path / "hippo"

    store = MemoryStore(path=store_path)
    seeded_ids: set[str] = set()
    for i in range(5):
        rid = uuid4()
        store.insert(_make_record(rid, surface=f"compact-apply-fixture-{i}"))
        seeded_ids.add(str(rid))

    update_rid = uuid4()
    store.insert(_make_record(update_rid, surface="compact-apply-fixture-update"))
    seeded_ids.add(str(update_rid))
    store.delete(update_rid)
    seeded_ids.discard(str(update_rid))

    records_tbl = store.db.open_table("records")
    rows_before = records_tbl.count_rows()
    assert rows_before == len(seeded_ids), (
        f"row count before ({rows_before}) must match seeded id count "
        f"({len(seeded_ids)})"
    )

    from iai_mcp.cli import _maintenance_compact_apply

    rc = _maintenance_compact_apply(store_path, hippo_dir)

    assert rc == 0, f"apply exit code: {rc}"

    records_tbl_after = store.db.open_table("records")
    rows_after = records_tbl_after.count_rows()
    assert rows_after == rows_before, (
        f"row count changed after optimize: before={rows_before} after={rows_after}"
    )

    audit_dir = fake_home / ".iai-mcp"
    audit_files = sorted(audit_dir.glob(".maintenance-compact-*.json"))
    assert audit_files, (
        f"expected audit file under {audit_dir}; found {list(audit_dir.iterdir())}"
    )
    ok_files = [
        p for p in audit_files if "FAILED" not in p.name
    ]
    assert ok_files, (
        f"expected non-FAILED audit file; got {[p.name for p in audit_files]}"
    )
    payload = json.loads(ok_files[-1].read_text())
    assert payload["status"] == "ok", payload

    df_post = (
        records_tbl_after.search()
        .select(["id"])
        .to_pandas()
    )
    post_id_set = {str(x) for x in df_post["id"].tolist()}
    assert post_id_set == seeded_ids, (
        f"post id-set differs from seeded set; "
        f"missing={seeded_ids - post_id_set} extra={post_id_set - seeded_ids}"
    )
