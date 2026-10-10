//! Allocator PEAK-counting regression guard: a `SELECT id` scan over a
//! 41-column, 15k-row table must not scale its peak allocation with the
//! catalog's column count. This is its own test binary (its own crate, its
//! own `#[global_allocator]`) so the counting wrapper never affects any other
//! test binary or production build.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use lillibrain::{Store, Value};
use lilliengine::ast::Stmt;
use lilliengine::catalog::Catalog;
use lilliengine::exec::execute_select;
use lilliengine::meta::MetaTable;
use lilliengine::parser::parse;
use lilliengine::plan::plan_select;
use lilliengine::rowcodec::encode_row;
use tempfile::tempdir;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

// `realloc` is NOT overridden: the default `GlobalAlloc::realloc` delegates to
// alloc(new)+copy+dealloc(old), which is what preserves the both-buffers-live
// peak while a growing `Vec<Row>` reallocates. A direct `System.realloc`
// override would delegate around alloc/dealloc and lose the very peak this
// test exists to catch.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let cur = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(cur, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// ~41 columns: the 6 real `records` columns plus 35 filler columns, matching
/// the production catalog width the boot-residual decomposition measured
/// (`boot_perf.rs`'s `WideFixture` is only 6 columns wide and under-measures
/// the per-row column-name clone this guard exists to catch).
const DDL_RECORDS: &str = "CREATE TABLE IF NOT EXISTS records ( \
    vec_label        INTEGER PRIMARY KEY AUTOINCREMENT , \
    id               TEXT NOT NULL UNIQUE , \
    tier             TEXT , \
    tombstoned_at    TEXT , \
    embedding_pending INTEGER , \
    embedding        BLOB NOT NULL , \
    f01 TEXT , f02 TEXT , f03 TEXT , f04 TEXT , f05 TEXT , \
    f06 TEXT , f07 TEXT , f08 TEXT , f09 TEXT , f10 TEXT , \
    f11 INTEGER , f12 INTEGER , f13 INTEGER , f14 INTEGER , f15 INTEGER , \
    f16 INTEGER , f17 INTEGER , f18 INTEGER , f19 INTEGER , f20 INTEGER , \
    f21 TEXT , f22 TEXT , f23 TEXT , f24 TEXT , f25 TEXT , \
    f26 TEXT , f27 TEXT , f28 TEXT , f29 TEXT , f30 TEXT , \
    f31 INTEGER , f32 INTEGER , f33 INTEGER , f34 INTEGER , f35 INTEGER )";

const N_COLS: usize = 41;
const EMBEDDING_BYTES: usize = 4 * 384;
const N_ROWS: i64 = 15_000;

struct WideFixture {
    _dir: tempfile::TempDir,
    store: Store,
    catalog: Catalog,
    root_map: BTreeMap<String, u32>,
}

impl WideFixture {
    fn new() -> WideFixture {
        let dir = tempdir().unwrap();
        let path = dir.path().join("records.lilli");
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut catalog = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.create_table(&store, &mut catalog, &mut root_map, DDL_RECORDS)
            .unwrap();

        store.begin_write().unwrap();
        for k in 1..=N_ROWS {
            let byte = (k % 256) as u8;
            let blob = vec![byte; EMBEDDING_BYTES];
            let tombstoned_at = if k % 2 == 0 {
                Value::Null
            } else {
                Value::Text("2024-01-01 00:00:00".to_string())
            };
            let mut row = vec![
                Value::Int(k),
                Value::Text(format!("id-{k:06}")),
                Value::Text("episodic".to_string()),
                tombstoned_at,
                Value::Int(0), // embedding_pending = 0
                Value::Blob(blob),
            ];
            for i in 1..=10 {
                row.push(Value::Text(format!("filler-a-{i}-{k}")));
            }
            for i in 1..=10 {
                row.push(Value::Int(k + i));
            }
            for i in 1..=10 {
                row.push(Value::Text(format!("filler-b-{i}-{k}")));
            }
            for i in 1..=5 {
                row.push(Value::Int(k * i));
            }
            debug_assert_eq!(row.len(), N_COLS);
            store
                .tree(*root_map.get("records").unwrap())
                .insert(k, &encode_row(&row))
                .unwrap();
        }
        store.commit().unwrap();

        WideFixture {
            _dir: dir,
            store,
            catalog,
            root_map,
        }
    }

    fn run_sql(&self, sql: &str) -> lilliengine::exec::ResultSet {
        let stmt = parse(sql).unwrap();
        let select = match stmt {
            Stmt::Select(s) => s,
            _ => panic!("not a SELECT"),
        };
        let plan = plan_select(&select, &self.catalog, vec![], None).unwrap();
        execute_select(
            &plan,
            &self.store,
            &self.catalog,
            &self.root_map,
            None,
            None,
            None,
        )
        .unwrap()
    }
}

/// Pinned pre-fix PEAK-allocation floor for `SELECT id FROM records` on this
/// test's 41-column/15k-row fixture (own shape, not shared with other fixtures).
/// A regression that reintroduces the per-row column-name clone must clear it.
const PRE_FIX_PEAK_DELTA_BYTES: usize = 92_215_972;

#[test]
fn row_alloc_peak_delta_floor() {
    let f = WideFixture::new();

    // Warm the pager/page cache with a throwaway query before pinning the
    // baseline: a cold first query's page-cache fill would otherwise be
    // charged to the pinned per-query PEAK delta even though `SELECT id`
    // masks the embedding BLOB out of decode (this repo's warm/cold
    // measurement discipline applies to allocator peaks too).
    let warm = f.run_sql("SELECT id FROM records");
    assert_eq!(warm.rows.len(), N_ROWS as usize);

    let baseline = CURRENT.load(Ordering::SeqCst);
    PEAK.store(baseline, Ordering::SeqCst);
    let rs = f.run_sql("SELECT id FROM records");
    assert_eq!(rs.rows.len(), N_ROWS as usize);
    let peak_delta = PEAK.load(Ordering::SeqCst).saturating_sub(baseline);

    // Diagnostic only (not asserted): the predicted post-fix floor once Row
    // holds only Values (no per-row column-name Strings), so a later task's
    // ceiling can be checked against a computed expectation instead of a
    // guess.
    let predicted_value_only_floor = std::mem::size_of::<Value>() * N_COLS * (N_ROWS as usize);
    println!(
        "row_alloc_regression: peak_delta(SELECT id) = {peak_delta} bytes ({:.2} MiB)",
        peak_delta as f64 / (1024.0 * 1024.0)
    );
    println!(
        "row_alloc_regression: predicted post-fix Value-only floor = {predicted_value_only_floor} bytes ({:.2} MiB)",
        predicted_value_only_floor as f64 / (1024.0 * 1024.0)
    );

    // The decode-free COUNT(*) path's own non-Row overhead, for context only
    // (not asserted): compare against the final ceiling below when tuning it.
    let baseline_count = CURRENT.load(Ordering::SeqCst);
    PEAK.store(baseline_count, Ordering::SeqCst);
    let count_rs = f.run_sql("SELECT COUNT ( * ) FROM records");
    assert_eq!(count_rs.rows[0][0].1, Value::Int(N_ROWS));
    let count_peak_delta = PEAK.load(Ordering::SeqCst).saturating_sub(baseline_count);
    println!(
        "row_alloc_regression: peak_delta(SELECT COUNT(*)) = {count_peak_delta} bytes ({:.2} MiB)",
        count_peak_delta as f64 / (1024.0 * 1024.0)
    );

    // Final ceiling: a non-surviving row is never allocated, and a survivor
    // projects straight into its output OutRow inside the scan closure.
    assert!(
        peak_delta < PRE_FIX_PEAK_DELTA_BYTES / 10,
        "peak_delta {peak_delta} did not clear the final ceiling of {} \
         (PRE_FIX_PEAK_DELTA_BYTES / 10)",
        PRE_FIX_PEAK_DELTA_BYTES / 10
    );
}
