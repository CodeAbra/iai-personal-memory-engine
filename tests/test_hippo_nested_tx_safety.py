from __future__ import annotations

import uuid
from pathlib import Path

import pyarrow as pa
import pytest

from iai_mcp.hippo import HippoDB


_EMBED_DIM = 384


def _make_db(tmp_path: Path) -> HippoDB:
    store_dir = tmp_path / "hippo"
    store_dir.mkdir(parents=True, exist_ok=True)
    return HippoDB(str(store_dir))


def _sample_edge_row(src_id: str, dst_id: str) -> dict:
    from datetime import datetime, timezone
    return {
        "src": src_id,
        "dst": dst_id,
        "edge_type": "hebbian",
        "weight": 0.1,
        "updated_at": datetime.now(timezone.utc),
    }


def _edge_arrow(rows: list[dict]) -> pa.Table:
    import pyarrow as pa
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("src", pa.string()),
                ("dst", pa.string()),
                ("edge_type", pa.string()),
                ("weight", pa.float32()),
                ("updated_at", pa.timestamp("us", tz="UTC")),
            ]
        ),
    )


def test_nested_merge_insert_under_outer_tx_succeeds(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())
    row = _sample_edge_row(src_id, dst_id)

    conn.execute("BEGIN")
    assert conn.in_transaction, "Expected conn to be in transaction after BEGIN"

    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(_edge_arrow([row]))
    )

    conn.execute("COMMIT")

    result = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src=? AND dst=?",
        (src_id, dst_id),
    ).fetchone()[0]
    assert result == 1, f"Expected 1 edge row, got {result}"

    db.close()


def test_nested_merge_insert_error_does_not_corrupt_outer_tx(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())

    conn.execute("BEGIN")

    bad_arrow = pa.table({"src": [src_id]})
    with pytest.raises(Exception):
        (
            tbl.merge_insert(["src", "dst", "edge_type"])
            .when_matched_update_all()
            .execute(bad_arrow)
        )

    count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert isinstance(count, int)

    conn.execute("ROLLBACK")
    db.close()


def test_non_nested_merge_insert_commits_cleanly(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())
    row = _sample_edge_row(src_id, dst_id)

    assert not conn.in_transaction, "Expected no active transaction before call"

    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(_edge_arrow([row]))
    )

    assert not conn.in_transaction, "Expected transaction closed after execute()"

    result = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src=? AND dst=?",
        (src_id, dst_id),
    ).fetchone()[0]
    assert result == 1, f"Expected 1 edge row, got {result}"

    db.close()


def test_update_under_outer_tx_succeeds(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())
    row = _sample_edge_row(src_id, dst_id)

    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(_edge_arrow([row]))
    )

    conn.execute("BEGIN")
    tbl.update(
        where=f"src = '{src_id}' AND dst = '{dst_id}' AND edge_type = 'hebbian'",
        values={"weight": 0.9, "updated_at": datetime.now(timezone.utc)},
    )
    conn.execute("COMMIT")

    result = conn.execute(
        "SELECT weight FROM edges WHERE src=? AND dst=?",
        (src_id, dst_id),
    ).fetchone()[0]
    assert abs(result - 0.9) < 0.01, f"Expected weight ~0.9, got {result}"

    db.close()


def test_delete_under_outer_tx_succeeds(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    conn = db._conn
    tbl = db.open_table("edges")

    src_id = str(uuid.uuid4())
    dst_id = str(uuid.uuid4())
    row = _sample_edge_row(src_id, dst_id)

    (
        tbl.merge_insert(["src", "dst", "edge_type"])
        .when_matched_update_all()
        .execute(_edge_arrow([row]))
    )

    conn.execute("BEGIN")
    tbl.delete(
        where=f"src = '{src_id}' AND dst = '{dst_id}'"
    )
    conn.execute("COMMIT")

    result = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE src=? AND dst=?",
        (src_id, dst_id),
    ).fetchone()[0]
    assert result == 0, f"Expected 0 edges after delete, got {result}"

    db.close()
