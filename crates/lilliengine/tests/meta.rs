//! Page-3 meta-table tests: DDL replay on reopen, root_map reconstruction, and
//! the per-table vec_label high-water-mark persistence.

use std::collections::BTreeMap;

use lilliengine::catalog::Catalog;
use lilliengine::meta::MetaTable;
use lillibrain::Store;
use tempfile::tempdir;

const DDL_RECORDS: &str =
    "CREATE TABLE IF NOT EXISTS records ( vec_label INTEGER PRIMARY KEY AUTOINCREMENT , \
     id TEXT NOT NULL UNIQUE , embedding BLOB NOT NULL )";
const DDL_EDGES: &str = "CREATE TABLE IF NOT EXISTS edges ( src TEXT NOT NULL , \
     dst TEXT NOT NULL , edge_type TEXT NOT NULL , PRIMARY KEY ( src , dst , edge_type ) )";

#[test]
fn create_tables_then_reopen_replays_catalog_and_root_map() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("m.lilli");

    let (records_root, edges_root);
    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();

        records_root = meta
            .create_table(&store, &mut cat, &mut root_map, DDL_RECORDS)
            .unwrap();
        edges_root = meta
            .create_table(&store, &mut cat, &mut root_map, DDL_EDGES)
            .unwrap();

        assert_ne!(records_root, edges_root);
        assert_eq!(root_map.get("records"), Some(&records_root));
        assert_eq!(root_map.get("edges"), Some(&edges_root));
    }

    // Reopen: a fresh catalog + root_map rebuilt purely from page 3.
    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.replay(&store, &mut cat, &mut root_map).unwrap();

        // Catalog rebuilt from replayed DDL.
        assert_eq!(
            cat.column_names("records").unwrap(),
            vec!["vec_label", "id", "embedding"]
        );
        assert_eq!(
            cat.primary_key_columns("edges").unwrap(),
            vec!["src", "dst", "edge_type"]
        );
        // root_map rebuilt with the same root pages.
        assert_eq!(root_map.get("records"), Some(&records_root));
        assert_eq!(root_map.get("edges"), Some(&edges_root));
    }
}

#[test]
fn vec_label_hwm_is_monotonic_and_survives_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hwm.lilli");

    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.create_table(&store, &mut cat, &mut root_map, DDL_RECORDS)
            .unwrap();

        assert_eq!(meta.next_vec_label(&store, "records").unwrap(), 1);
        assert_eq!(meta.next_vec_label(&store, "records").unwrap(), 2);
        assert_eq!(meta.next_vec_label(&store, "records").unwrap(), 3);
    }

    // Reopen: the HWM resumes from 3, so the next label is 4.
    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.replay(&store, &mut cat, &mut root_map).unwrap();

        assert_eq!(meta.next_vec_label(&store, "records").unwrap(), 4);
    }
}

#[test]
fn set_vec_label_hwm_persists_and_next_label_is_max_plus_one() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("setHwm.lilli");

    // Simulate a verbatim bulk copy that wrote explicit vec_labels up to 4096,
    // then set the mark to that max so the next auto-label cannot collide.
    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.create_table(&store, &mut cat, &mut root_map, DDL_RECORDS)
            .unwrap();

        meta.set_vec_label_hwm(&store, "records", 4096).unwrap();
        // The next auto-assigned label is max + 1, never a collision.
        assert_eq!(meta.next_vec_label(&store, "records").unwrap(), 4097);
    }

    // Reopen with a fresh handle: the mark survived, so labels resume past 4097.
    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.replay(&store, &mut cat, &mut root_map).unwrap();

        assert_eq!(meta.next_vec_label(&store, "records").unwrap(), 4098);
    }
}

#[test]
fn set_vec_label_hwm_is_monotonic_never_lowers_persisted_mark() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("monoHwm.lilli");

    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.create_table(&store, &mut cat, &mut root_map, DDL_RECORDS)
            .unwrap();
        meta.set_vec_label_hwm(&store, "records", 100).unwrap();
    }

    // A fresh handle that has not yet touched the table must read the on-disk
    // mark before setting, so a stale lower value cannot clobber it.
    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.replay(&store, &mut cat, &mut root_map).unwrap();

        meta.set_vec_label_hwm(&store, "records", 50).unwrap(); // no-op: 50 <= 100
        assert_eq!(meta.next_vec_label(&store, "records").unwrap(), 101);
    }
}

#[test]
fn create_table_is_idempotent_per_name() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("idem.lilli");
    let store = Store::open(&path).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut cat = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();

    let first = meta
        .create_table(&store, &mut cat, &mut root_map, DDL_RECORDS)
        .unwrap();
    let again = meta
        .create_table(&store, &mut cat, &mut root_map, DDL_RECORDS)
        .unwrap();
    assert_eq!(first, again, "re-registering a table returns the same root");
    assert_eq!(root_map.len(), 1, "no second tree was allocated");
}

#[test]
fn create_table_without_if_not_exists_errors_on_existing() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("dup.lilli");
    let store = Store::open(&path).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut cat = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();

    let bare = "CREATE TABLE t ( id TEXT , n INTEGER )";
    meta.create_table(&store, &mut cat, &mut root_map, bare)
        .unwrap();
    // A second bare CREATE TABLE of the same table errors (no silent no-op).
    let err = meta
        .create_table(&store, &mut cat, &mut root_map, bare)
        .unwrap_err();
    assert!(
        err.to_string().contains("already exists"),
        "a repeat bare CREATE TABLE must report table-already-exists, got {err}"
    );
    // IF NOT EXISTS makes it the existing-table no-op.
    let again = meta
        .create_table(
            &store,
            &mut cat,
            &mut root_map,
            "CREATE TABLE IF NOT EXISTS t ( id TEXT , n INTEGER )",
        )
        .unwrap();
    assert_eq!(root_map.get("t"), Some(&again), "IF NOT EXISTS is a no-op");
    assert_eq!(root_map.len(), 1, "no second tree allocated");
}

#[test]
fn replay_tolerates_repeated_add_column_meta_row() {
    use lilliengine::exec::TxnScope;

    let dir = tempdir().unwrap();
    let path = dir.path().join("dupcol.lilli");
    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.create_table(&store, &mut cat, &mut root_map, "CREATE TABLE t ( id TEXT )")
            .unwrap();
        // A store written without the live duplicate guard can carry the same
        // ADD COLUMN meta row twice; reopen must stay available.
        meta.record_ddl(&store, "ALTER TABLE t ADD COLUMN role TEXT", TxnScope::Own)
            .unwrap();
        meta.record_ddl(&store, "ALTER TABLE t ADD COLUMN role TEXT", TxnScope::Own)
            .unwrap();
    }
    {
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut cat = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.replay(&store, &mut cat, &mut root_map)
            .expect("replay treats the repeated ADD COLUMN as a no-op");
        let names = cat.column_names("t").expect("table present");
        assert_eq!(
            names.iter().filter(|n| n.as_str() == "role").count(),
            1,
            "replay must collapse the repeated ADD COLUMN to one column"
        );
    }
}
