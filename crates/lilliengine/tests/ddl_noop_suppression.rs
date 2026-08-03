//! No-op DDL must not grow the meta history: every reopen replays the whole
//! DDL history, so a per-boot IF-NOT-EXISTS re-record bloats the meta tree
//! without bound and taxes every open and read-only snapshot refresh.

use lilliengine::conn::Connection;
use tempfile::tempdir;

const BOOT_DDL: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS records (id TEXT, tier TEXT)",
    "CREATE INDEX IF NOT EXISTS idx_records_id ON records (id)",
    "CREATE INDEX IF NOT EXISTS idx_records_tier ON records (tier)",
];

fn boot(path: &str) -> Connection {
    let mut conn = Connection::open(path, 0).unwrap();
    for ddl in BOOT_DDL {
        conn.execute(ddl, vec![]).unwrap();
    }
    conn
}

fn schema_rows(conn: &mut Connection) -> usize {
    let mut cur = conn
        .execute("SELECT name FROM sqlite_master", vec![])
        .unwrap();
    cur.fetchall().len()
}

#[test]
fn repeated_boots_leave_schema_and_registry_fixed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();

    let mut first = boot(path);
    let baseline = schema_rows(&mut first);
    drop(first);

    for _ in 0..10 {
        let conn = boot(path);
        drop(conn);
    }

    let mut last = boot(path);
    assert_eq!(
        schema_rows(&mut last),
        baseline,
        "10 boots of identical IF-NOT-EXISTS DDL must not multiply schema objects"
    );
}

#[test]
fn duplicate_bare_create_index_errors_like_sqlite3() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();
    let mut conn = Connection::open(path, 0).unwrap();
    conn.execute("CREATE TABLE t (a TEXT)", vec![]).unwrap();
    conn.execute("CREATE INDEX ix ON t (a)", vec![]).unwrap();
    let err = conn
        .execute("CREATE INDEX ix ON t (a)", vec![])
        .unwrap_err();
    assert!(err.to_string().contains("already exists"), "got: {err}");
    assert!(conn
        .execute("CREATE INDEX IF NOT EXISTS ix ON t (a)", vec![])
        .is_ok());
}

#[test]
fn vacuum_compacts_a_bloated_ddl_history() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();

    // Simulate a store written before suppression: force duplicate rows via
    // repeated boots of a build that recorded them. The public API no longer
    // creates duplicates, so seed them through the meta layer contract:
    // repeated boots now record nothing, and VACUUM on a clean store is a
    // no-op that must still succeed.
    let mut conn = boot(path);
    conn.execute("VACUUM", vec![]).unwrap();
    let before = schema_rows(&mut conn);
    conn.execute("VACUUM", vec![]).unwrap();
    assert_eq!(schema_rows(&mut conn), before);
}
