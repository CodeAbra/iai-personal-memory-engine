//! SELECT-path behavior tests: three-valued evaluation, datetime() normalization,
//! ORDER BY collation, the planner, the scan/filter/project/order/limit pipeline,
//! and the aggregate / GROUP BY / HAVING paths.
//!
//! The expected results encode sqlite3 (and the reference engine's) semantics:
//! NULL-propagating comparisons, numeric-affinity coercion, NULLs-first
//! collation, and datetime() second-resolution UTC normalization.

use lillibrain::Value;
use lilliengine::eval::{
    datetime_scalar, eval_predicate, eval_scalar, like_match, order_key, sql_and, sql_compare,
    sql_not, sql_or, Row,
};

// ---------------------------------------------------------------------------
// Three-valued evaluator
// ---------------------------------------------------------------------------

fn t(s: &str) -> Value {
    Value::Text(s.to_string())
}

#[test]
fn compare_int_equals_real() {
    // 1 == 1.0 is true under numeric coercion.
    assert_eq!(
        sql_compare("=", Value::Int(1), Value::Float(1.0)),
        Some(true)
    );
    assert_eq!(
        sql_compare("=", Value::Float(2.0), Value::Int(2)),
        Some(true)
    );
}

#[test]
fn compare_numeric_text_affinity() {
    // '5' == 5 is true (reference numeric affinity coercion of the text operand).
    assert_eq!(sql_compare("=", t("5"), Value::Int(5)), Some(true));
    assert_eq!(sql_compare("<", t("5"), Value::Int(6)), Some(true));
    // Non-numeric text never coerces; a number always sorts before text.
    assert_eq!(sql_compare("=", t("abc"), Value::Int(5)), Some(false));
    assert_eq!(sql_compare("<", Value::Int(5), t("abc")), Some(true));
}

#[test]
fn compare_null_propagates() {
    // Any NULL operand yields the SQL unknown (None), never false.
    assert_eq!(sql_compare("=", Value::Null, Value::Int(1)), None);
    assert_eq!(sql_compare("=", Value::Int(1), Value::Null), None);
    assert_eq!(sql_compare("!=", Value::Null, Value::Null), None);
    assert_eq!(sql_compare("<", Value::Null, t("x")), None);
}

#[test]
fn kleene_and() {
    assert_eq!(sql_and(Some(true), None), None); // true AND NULL -> NULL
    assert_eq!(sql_and(Some(false), None), Some(false)); // false AND NULL -> false
    assert_eq!(sql_and(None, Some(false)), Some(false));
    assert_eq!(sql_and(Some(true), Some(true)), Some(true));
    assert_eq!(sql_and(None, None), None);
}

#[test]
fn kleene_or() {
    assert_eq!(sql_or(None, Some(true)), Some(true)); // NULL OR true -> true
    assert_eq!(sql_or(None, Some(false)), None); // NULL OR false -> NULL
    assert_eq!(sql_or(Some(false), Some(false)), Some(false));
    assert_eq!(sql_or(None, None), None);
}

#[test]
fn kleene_not() {
    assert_eq!(sql_not(None), None); // NOT NULL -> NULL
    assert_eq!(sql_not(Some(true)), Some(false));
    assert_eq!(sql_not(Some(false)), Some(true));
}

#[test]
fn like_wildcards_case_insensitive() {
    assert_eq!(like_match(&t("Hello"), &t("h%")), Some(true));
    assert_eq!(like_match(&t("Hello"), &t("%LLO")), Some(true));
    assert_eq!(like_match(&t("Hello"), &t("h_llo")), Some(true));
    assert_eq!(like_match(&t("Hello"), &t("h_lo")), Some(false));
    assert_eq!(like_match(&t("abc"), &t("%")), Some(true));
    assert_eq!(like_match(&t(""), &t("%")), Some(true));
    assert_eq!(like_match(&t("a.b"), &t("a.b")), Some(true)); // '.' is literal
    assert_eq!(like_match(&Value::Null, &t("%")), None);
    assert_eq!(like_match(&t("x"), &Value::Null), None);
    // A numeric LHS is stringified before matching.
    assert_eq!(like_match(&Value::Int(123), &t("1%")), Some(true));
}

#[test]
fn in_list_null_semantics() {
    let row = Row::new();
    // 5 IN (1, NULL, 2) -> NULL (no match, NULL present => unknown).
    let items = vec![lit(Value::Int(1)), lit(Value::Null), lit(Value::Int(2))];
    let in_expr = in_list(lit(Value::Int(5)), items.clone());
    assert_eq!(eval_predicate(&in_expr, &row), None);
    // 1 IN (1, NULL, 2) -> true (match found).
    let in_expr = in_list(lit(Value::Int(1)), items);
    assert_eq!(eval_predicate(&in_expr, &row), Some(true));
    // 5 IN (1, 2) -> false (no NULL, no match).
    let in_expr = in_list(
        lit(Value::Int(5)),
        vec![lit(Value::Int(1)), lit(Value::Int(2))],
    );
    assert_eq!(eval_predicate(&in_expr, &row), Some(false));
    // NULL IN (1, 2) -> NULL.
    let in_expr = in_list(lit(Value::Null), vec![lit(Value::Int(1))]);
    assert_eq!(eval_predicate(&in_expr, &row), None);
}

#[test]
fn coalesce_first_non_null() {
    let row = Row::new();
    let c = coalesce(vec![lit(Value::Null), lit(Value::Null), lit(t("x"))]);
    assert_eq!(eval_scalar(&c, &row), t("x"));
    let c = coalesce(vec![lit(Value::Null), lit(Value::Null)]);
    assert_eq!(eval_scalar(&c, &row), Value::Null);
    let c = coalesce(vec![lit(Value::Int(7)), lit(t("y"))]);
    assert_eq!(eval_scalar(&c, &row), Value::Int(7));
}

#[test]
fn datetime_normalization_matches_sqlite3() {
    let d = |s: &str| datetime_scalar(&t(s));
    assert_eq!(d("2024-01-02T03:04:05"), t("2024-01-02 03:04:05"));
    assert_eq!(d("2024-01-02 03:04:05"), t("2024-01-02 03:04:05"));
    assert_eq!(d("2024-01-02T03:04:05Z"), t("2024-01-02 03:04:05"));
    // +02:00 offset converts to UTC (subtract two hours).
    assert_eq!(d("2024-01-02T03:04:05+02:00"), t("2024-01-02 01:04:05"));
    // Sub-second fraction is truncated.
    assert_eq!(d("2024-01-02T03:04:05.123456"), t("2024-01-02 03:04:05"));
    // Bare date -> midnight.
    assert_eq!(d("2024-01-02"), t("2024-01-02 00:00:00"));
    // Offset that rolls the clock across midnight, with date roll.
    assert_eq!(d("2024-01-02T23:30:00+02:00"), t("2024-01-02 21:30:00"));
    assert_eq!(d("2024-01-02T00:30:00-02:00"), t("2024-01-02 02:30:00"));
}

#[test]
fn datetime_unparseable_degrades_to_null() {
    assert_eq!(datetime_scalar(&t("garbage")), Value::Null);
    assert_eq!(datetime_scalar(&t("2024-13-99")), Value::Null);
    assert_eq!(datetime_scalar(&Value::Null), Value::Null);
    assert_eq!(datetime_scalar(&Value::Int(12345)), Value::Null);
    assert_eq!(datetime_scalar(&Value::Blob(vec![1, 2])), Value::Null);
}

#[test]
fn order_key_nulls_first_numeric_before_text() {
    // NULL < number < text < blob.
    let nul = order_key(&Value::Null);
    let num = order_key(&Value::Int(10));
    let txt = order_key(&t("a"));
    let blob = order_key(&Value::Blob(vec![0]));
    assert!(nul < num);
    assert!(num < txt);
    assert!(txt < blob);
    // Within the number class, value order holds and int↔real interleave.
    assert!(order_key(&Value::Int(2)) < order_key(&Value::Int(10)));
    assert!(order_key(&Value::Int(1)) < order_key(&Value::Float(1.5)));
}

// Expression-construction helpers shared across tests.
use lilliengine::ast::Expr;

fn lit(v: Value) -> Expr {
    Expr::Literal(v)
}
fn col(name: &str) -> Expr {
    Expr::Column(name.to_string())
}
fn binop(op: &str, l: Expr, r: Expr) -> Expr {
    Expr::BinOp {
        op: op.to_string(),
        left: Box::new(l),
        right: Box::new(r),
    }
}
fn in_list(operand: Expr, items: Vec<Expr>) -> Expr {
    Expr::InList {
        operand: Box::new(operand),
        items,
        text_set: None,
    }
}
fn coalesce(args: Vec<Expr>) -> Expr {
    Expr::Coalesce(args)
}

#[test]
fn where_x_equals_null_returns_nothing() {
    // WHERE x = NULL must reject every row (the predicate is always unknown).
    let pred = binop("=", col("x"), lit(Value::Null));
    let mut row = Row::new();
    row.set("x", Value::Int(1));
    assert_eq!(eval_predicate(&pred, &row), None);
    row.set("x", Value::Null);
    assert_eq!(eval_predicate(&pred, &row), None);
}

#[test]
fn is_null_predicate_on_column() {
    let pred = Expr::IsNull {
        operand: Box::new(col("x")),
        negated: false,
    };
    let mut row = Row::new();
    row.set("x", Value::Null);
    assert_eq!(eval_predicate(&pred, &row), Some(true));
    row.set("x", Value::Int(1));
    assert_eq!(eval_predicate(&pred, &row), Some(false));
}

// ---------------------------------------------------------------------------
// End-to-end pipeline: planner + scan/filter/project/order/limit, aggregates,
// GROUP BY, HAVING, the rowid resolution, and the COUNT(*) decode-free path.
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;

use lillibrain::Store;
use lilliengine::ast::Stmt;
use lilliengine::catalog::Catalog;
use lilliengine::exec::{execute_select, execute_select_instrumented, ResultSet};
use lilliengine::meta::MetaTable;
use lilliengine::parser::parse;
use lilliengine::plan::plan_select;
use lilliengine::rowcodec::encode_row;
use tempfile::tempdir;

const DDL_T: &str = "CREATE TABLE IF NOT EXISTS t ( vec_label INTEGER PRIMARY KEY AUTOINCREMENT , \
     id TEXT NOT NULL UNIQUE , n INTEGER , label TEXT , ts TEXT )";

/// A store fixture with table `t` populated by `rows` (each a full column vector
/// in declaration order). Returns everything the executor needs.
struct Fixture {
    _dir: tempfile::TempDir,
    store: Store,
    catalog: Catalog,
    root_map: BTreeMap<String, u32>,
}

impl Fixture {
    fn new(rows: &[Vec<Value>]) -> Fixture {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.lilli");
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut catalog = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        let root = meta
            .create_table(&store, &mut catalog, &mut root_map, DDL_T)
            .unwrap();
        let tree_key = |r: &Vec<Value>| match &r[0] {
            Value::Int(i) => *i,
            _ => panic!("vec_label must be int"),
        };
        store.begin_write().unwrap();
        for r in rows {
            let payload = encode_row(r);
            store.tree(root).insert(tree_key(r), &payload).unwrap();
        }
        store.commit().unwrap();
        Fixture {
            _dir: dir,
            store,
            catalog,
            root_map,
        }
    }

    fn run(&self, sql: &str, params: Vec<Value>) -> ResultSet {
        let stmt = parse(sql).unwrap();
        let select = match stmt {
            Stmt::Select(s) => s,
            _ => panic!("not a select"),
        };
        let plan = plan_select(&select, &self.catalog, params, None).unwrap();
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

    fn run_instrumented(&self, sql: &str) -> (ResultSet, bool) {
        let stmt = parse(sql).unwrap();
        let select = match stmt {
            Stmt::Select(s) => s,
            _ => panic!("not a select"),
        };
        let plan = plan_select(&select, &self.catalog, vec![], None).unwrap();
        let (rs, stats) = execute_select_instrumented(
            &plan,
            &self.store,
            &self.catalog,
            &self.root_map,
            None,
            None,
            None,
        )
        .unwrap();
        (rs, stats.decoded_rows)
    }
}

fn row(vl: i64, id: &str, n: Option<i64>, label: Option<&str>, ts: Option<&str>) -> Vec<Value> {
    vec![
        Value::Int(vl),
        Value::Text(id.to_string()),
        n.map(Value::Int).unwrap_or(Value::Null),
        label
            .map(|s| Value::Text(s.to_string()))
            .unwrap_or(Value::Null),
        ts.map(|s| Value::Text(s.to_string()))
            .unwrap_or(Value::Null),
    ]
}

fn col_vals(rs: &ResultSet, col: &str) -> Vec<Value> {
    let idx = rs.columns.iter().position(|c| c == col).unwrap();
    rs.rows.iter().map(|r| r[idx].1.clone()).collect()
}

#[test]
fn plan_rejects_unknown_table() {
    let f = Fixture::new(&[]);
    let stmt = parse("SELECT * FROM nope").unwrap();
    let select = match stmt {
        Stmt::Select(s) => s,
        _ => unreachable!(),
    };
    assert!(plan_select(&select, &f.catalog, vec![], None).is_err());
}

#[test]
fn select_star_returns_all_columns_in_order() {
    let f = Fixture::new(&[row(1, "a", Some(10), Some("x"), None)]);
    let rs = f.run("SELECT * FROM t", vec![]);
    assert_eq!(rs.columns, vec!["vec_label", "id", "n", "label", "ts"]);
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(rs.rows[0][1].1, Value::Text("a".to_string()));
}

#[test]
fn named_column_projection() {
    let f = Fixture::new(&[row(1, "a", Some(10), Some("x"), None)]);
    let rs = f.run("SELECT id , n FROM t", vec![]);
    assert_eq!(rs.columns, vec!["id", "n"]);
    assert_eq!(rs.rows[0][0].1, Value::Text("a".to_string()));
    assert_eq!(rs.rows[0][1].1, Value::Int(10));
}

#[test]
fn where_three_valued_filter() {
    let f = Fixture::new(&[
        row(1, "a", Some(1), None, None),
        row(2, "b", None, None, None),
        row(3, "c", Some(2), None, None),
    ]);
    // x = NULL returns nothing.
    let rs = f.run("SELECT id FROM t WHERE n = NULL", vec![]);
    assert_eq!(rs.rows.len(), 0);
    // IS NULL works.
    let rs = f.run("SELECT id FROM t WHERE n IS NULL", vec![]);
    assert_eq!(col_vals(&rs, "id"), vec![Value::Text("b".to_string())]);
    // n IN (1, NULL) matches 1 but not the NULL-only / non-member rows.
    let rs = f.run("SELECT id FROM t WHERE n IN ( 1 , NULL )", vec![]);
    assert_eq!(col_vals(&rs, "id"), vec![Value::Text("a".to_string())]);
}

#[test]
fn where_with_positional_param() {
    let f = Fixture::new(&[
        row(1, "a", Some(5), None, None),
        row(2, "b", Some(7), None, None),
    ]);
    let rs = f.run("SELECT id FROM t WHERE n = ?", vec![Value::Int(7)]);
    assert_eq!(col_vals(&rs, "id"), vec![Value::Text("b".to_string())]);
}

#[test]
fn bind_count_mismatch_fails_loud() {
    let f = Fixture::new(&[row(1, "a", Some(5), None, None)]);
    let stmt = parse("SELECT id FROM t WHERE n = ? AND vec_label = ?").unwrap();
    let select = match stmt {
        Stmt::Select(s) => s,
        _ => unreachable!(),
    };
    // Too few.
    let plan = plan_select(&select, &f.catalog, vec![Value::Int(5)], None).unwrap();
    let err =
        execute_select(&plan, &f.store, &f.catalog, &f.root_map, None, None, None).unwrap_err();
    assert!(format!("{err}").contains("Incorrect number of bindings supplied"));
    // Too many.
    let plan = plan_select(
        &select,
        &f.catalog,
        vec![Value::Int(5), Value::Int(1), Value::Int(9)],
        None,
    )
    .unwrap();
    let err =
        execute_select(&plan, &f.store, &f.catalog, &f.root_map, None, None, None).unwrap_err();
    assert!(format!("{err}").contains("Incorrect number of bindings supplied"));
}

#[test]
fn order_by_nulls_first_then_limit_offset() {
    let f = Fixture::new(&[
        row(1, "a", Some(3), None, None),
        row(2, "b", None, None, None),
        row(3, "c", Some(1), None, None),
        row(4, "d", Some(2), None, None),
    ]);
    let rs = f.run("SELECT id FROM t ORDER BY n", vec![]);
    // NULL first, then 1, 2, 3 -> b, c, d, a.
    assert_eq!(
        col_vals(&rs, "id"),
        vec![
            Value::Text("b".into()),
            Value::Text("c".into()),
            Value::Text("d".into()),
            Value::Text("a".into()),
        ]
    );
    // DESC: 3, 2, 1, then NULL last -> a, d, c, b.
    let rs = f.run("SELECT id FROM t ORDER BY n DESC", vec![]);
    assert_eq!(
        col_vals(&rs, "id"),
        vec![
            Value::Text("a".into()),
            Value::Text("d".into()),
            Value::Text("c".into()),
            Value::Text("b".into()),
        ]
    );
    // LIMIT / OFFSET over the ascending order.
    let rs = f.run("SELECT id FROM t ORDER BY n LIMIT 2 OFFSET 1", vec![]);
    assert_eq!(
        col_vals(&rs, "id"),
        vec![Value::Text("c".into()), Value::Text("d".into())]
    );
}

#[test]
fn order_by_datetime_normalizes() {
    let f = Fixture::new(&[
        row(1, "a", None, None, Some("2024-01-02T05:00:00+02:00")), // -> 03:00 UTC
        row(2, "b", None, None, Some("2024-01-02 02:00:00")),
        row(3, "c", None, None, Some("garbage")), // -> NULL, sorts first
    ]);
    let rs = f.run("SELECT id FROM t ORDER BY datetime ( ts )", vec![]);
    assert_eq!(
        col_vals(&rs, "id"),
        vec![
            Value::Text("c".into()),
            Value::Text("b".into()),
            Value::Text("a".into())
        ]
    );
}

#[test]
fn two_key_order_by() {
    let f = Fixture::new(&[
        row(1, "a", Some(1), Some("y"), None),
        row(2, "b", Some(1), Some("x"), None),
        row(3, "c", Some(2), Some("a"), None),
    ]);
    let rs = f.run("SELECT id FROM t ORDER BY n , label", vec![]);
    // n=1 group ordered by label (x<y): b, a; then n=2: c.
    assert_eq!(
        col_vals(&rs, "id"),
        vec![
            Value::Text("b".into()),
            Value::Text("a".into()),
            Value::Text("c".into())
        ]
    );
}

#[test]
fn expr_projection_honors_order_by_before_limit() {
    // A projection carrying an expression (COALESCE / alias) must sort by ORDER BY
    // before applying LIMIT, returning the correctly-sorted top-N rather than the
    // first rows in scan order.
    let f = Fixture::new(&[
        row(1, "a", Some(3), None, None),
        row(2, "b", Some(1), None, None),
        row(3, "c", None, None, None),
        row(4, "d", Some(2), None, None),
    ]);
    // ORDER BY n ASC LIMIT 2 over a COALESCE projection: NULL first, then 1
    // -> rows c, b.
    let rs = f.run(
        "SELECT id , COALESCE ( n , 0 ) AS p FROM t ORDER BY n LIMIT 2",
        vec![],
    );
    assert_eq!(
        col_vals(&rs, "id"),
        vec![Value::Text("c".into()), Value::Text("b".into())],
        "ORDER BY n ASC on an expr projection must return the sorted top-N"
    );
    // The COALESCE value travels with the sorted row (NULL -> 0 for c).
    assert_eq!(col_vals(&rs, "p"), vec![Value::Int(0), Value::Int(1)]);

    // DESC variant: 3, 2 -> a, d.
    let rs = f.run(
        "SELECT id , COALESCE ( n , 0 ) AS p FROM t ORDER BY n DESC LIMIT 2",
        vec![],
    );
    assert_eq!(
        col_vals(&rs, "id"),
        vec![Value::Text("a".into()), Value::Text("d".into())]
    );

    // ORDER BY a column NOT in the projection, with OFFSET — middle slice b, d.
    let rs = f.run(
        "SELECT COALESCE ( label , 'none' ) AS t FROM t ORDER BY n LIMIT 2 OFFSET 1",
        vec![],
    );
    assert_eq!(rs.rows.len(), 2);
}

#[test]
fn expr_projection_without_order_by_keeps_scan_order_fast_path() {
    // With NO ORDER BY the expression projection returns rows in scan order with
    // LIMIT applied early; this pins the fast path is preserved (it stops the scan
    // after enough survivors rather than materializing + sorting the whole table).
    let f = Fixture::new(&[
        row(1, "a", Some(3), None, None),
        row(2, "b", Some(1), None, None),
        row(3, "c", Some(2), None, None),
    ]);
    let (rs, decoded) = f.run_instrumented("SELECT id , COALESCE ( n , 0 ) AS p FROM t LIMIT 2");
    assert!(decoded, "the expr projection decodes rows");
    // First two rows in scan order (insertion order): a, b.
    assert_eq!(
        col_vals(&rs, "id"),
        vec![Value::Text("a".into()), Value::Text("b".into())]
    );
}

#[test]
fn rowid_resolves_to_vec_label() {
    let f = Fixture::new(&[row(7, "a", None, None, None)]);
    let rs = f.run("SELECT rowid FROM t", vec![]);
    assert_eq!(rs.columns, vec!["rowid"]);
    assert_eq!(rs.rows[0][0].1, Value::Int(7));
}

#[test]
fn id_point_lookup_is_a_single_match() {
    let f = Fixture::new(&[
        row(1, "alpha", Some(1), None, None),
        row(2, "beta", Some(2), None, None),
    ]);
    let rs = f.run("SELECT id , n FROM t WHERE id = 'beta' LIMIT 1", vec![]);
    assert_eq!(rs.rows.len(), 1);
    assert_eq!(rs.rows[0][0].1, Value::Text("beta".into()));
    assert_eq!(rs.rows[0][1].1, Value::Int(2));
    // A missing id yields an empty set.
    let rs = f.run("SELECT id FROM t WHERE id = 'missing' LIMIT 1", vec![]);
    assert_eq!(rs.rows.len(), 0);
}

#[test]
fn id_point_lookup_with_index_is_scan_free_and_matches_unindexed() {
    use lilliengine::exec::IdIndex;
    let f = Fixture::new(&[
        row(1, "alpha", Some(1), None, None),
        row(2, "beta", Some(2), None, None),
        row(3, "gamma", Some(3), None, None),
    ]);
    let root = *f.root_map.get("t").unwrap();
    let col_names = f.catalog.column_names("t").unwrap();
    let mut idx = IdIndex::new();
    idx.ensure_built(&f.store.tree(root), &col_names).unwrap();

    let run_idx = |sql: &str, idx: &mut IdIndex| {
        let select = match parse(sql).unwrap() {
            Stmt::Select(s) => s,
            _ => unreachable!(),
        };
        let plan = plan_select(&select, &f.catalog, vec![], None).unwrap();
        execute_select(
            &plan,
            &f.store,
            &f.catalog,
            &f.root_map,
            Some(idx),
            None,
            None,
        )
        .unwrap()
    };

    // The indexed point lookup returns the same row as the unindexed scan, and
    // does NOT scan the table (the index resolves the row-key directly).
    let sql = "SELECT id , n FROM t WHERE id = 'gamma' LIMIT 1";
    let unindexed = f.run(sql, vec![]);
    f.store.reset_full_scan_count();
    let indexed = run_idx(sql, &mut idx);
    assert_eq!(
        f.store.full_scan_count(),
        0,
        "an id point lookup with a built index must be scan-free"
    );
    assert_eq!(
        indexed.rows, unindexed.rows,
        "indexed and unindexed point lookups must return identical rows"
    );
    assert_eq!(indexed.columns, unindexed.columns);

    // A missing id is also scan-free and returns an empty set.
    f.store.reset_full_scan_count();
    let missing = run_idx("SELECT id FROM t WHERE id = 'nope' LIMIT 1", &mut idx);
    assert_eq!(
        f.store.full_scan_count(),
        0,
        "an absent id is scan-free with a built index"
    );
    assert_eq!(missing.rows.len(), 0);
}

#[test]
fn edges_in_lookup_with_col_index_is_scan_free_and_matches_unindexed() {
    use lilliengine::exec::ColIndex;
    use lilliengine::meta::MetaTable;
    use lilliengine::rowcodec::encode_row;
    use tempfile::tempdir;

    const DDL_E: &str = "CREATE TABLE IF NOT EXISTS edges ( src TEXT , dst TEXT , \
        edge_type TEXT , weight REAL , PRIMARY KEY ( src , dst , edge_type ) )";
    let dir = tempdir().unwrap();
    let store = Store::open(dir.path().join("e.lilli")).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    let root = meta
        .create_table(&store, &mut catalog, &mut root_map, DDL_E)
        .unwrap();

    // Seed edges: a..a9 each with a contradicts/relates edge. Row key = sequence.
    let mk = |src: &str, dst: &str, et: &str, w: f64| {
        vec![
            Value::Text(src.into()),
            Value::Text(dst.into()),
            Value::Text(et.into()),
            Value::Float(w),
        ]
    };
    store.begin_write().unwrap();
    let mut k = 1i64;
    for i in 0..50 {
        let src = format!("n{}", i);
        let dst = format!("n{}", (i + 1) % 50);
        store
            .tree(root)
            .insert(k, &encode_row(&mk(&src, &dst, "contradicts", 0.5)))
            .unwrap();
        k += 1;
        store
            .tree(root)
            .insert(k, &encode_row(&mk(&dst, &src, "relates", 0.2)))
            .unwrap();
        k += 1;
    }
    store.commit().unwrap();

    let mut cidx = ColIndex::new(vec!["src".to_string(), "dst".to_string()]);
    cidx.ensure_built(&store.tree(root), &catalog.column_names("edges").unwrap())
        .unwrap();

    let run = |sql: &str, cidx: Option<&mut ColIndex>| {
        let select = match parse(sql).unwrap() {
            Stmt::Select(s) => s,
            _ => unreachable!(),
        };
        let plan = plan_select(&select, &catalog, vec![], None).unwrap();
        execute_select(&plan, &store, &catalog, &root_map, None, cidx, None).unwrap()
    };

    // The recall adjacency shape: src IN (...) OR dst IN (...) AND edge_type IN (...).
    let sql = "SELECT src , dst , edge_type , weight FROM edges \
               WHERE ( src IN ( 'n3' , 'n7' ) OR dst IN ( 'n3' , 'n7' ) ) \
               AND edge_type IN ( 'contradicts' )";
    let unindexed = run(sql, None);
    store.reset_full_scan_count();
    let indexed = run(sql, Some(&mut cidx));
    assert_eq!(
        store.full_scan_count(),
        0,
        "an indexed IN lookup must be scan-free"
    );
    // Row order may differ (index union vs sequential scan) — compare as sets.
    let to_set = |rs: &ResultSet| {
        let mut v: Vec<Vec<(String, Value)>> = rs.rows.clone();
        v.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        v
    };
    assert_eq!(
        to_set(&indexed),
        to_set(&unindexed),
        "indexed IN lookup must return the same rows as the scan"
    );
    assert_eq!(indexed.columns, unindexed.columns);
    assert!(
        !indexed.rows.is_empty(),
        "the seed corpus has matching edges"
    );
}

#[test]
fn records_id_in_lookup_with_col_index_is_scan_free_and_matches_unindexed() {
    use lilliengine::exec::ColIndex;
    use lilliengine::meta::MetaTable;
    use lilliengine::rowcodec::encode_row;
    use tempfile::tempdir;

    // A records-shaped table: AUTOINCREMENT vec_label PK, a UNIQUE text id, and an
    // embedding BLOB — the batch-get filters on `id` and projects the wide row.
    const DDL_R: &str = "CREATE TABLE IF NOT EXISTS records ( \
        vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
        tier TEXT , literal_surface TEXT , embedding BLOB )";
    let dir = tempdir().unwrap();
    let store = Store::open(dir.path().join("r.lilli")).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    let root = meta
        .create_table(&store, &mut catalog, &mut root_map, DDL_R)
        .unwrap();

    store.begin_write().unwrap();
    for k in 1..=200i64 {
        let id = format!("id-{k:04}");
        let blob = vec![(k % 256) as u8; 32];
        let row = vec![
            Value::Int(k),
            Value::Text(id),
            Value::Text("episodic".into()),
            Value::Text(format!("surface {k}")),
            Value::Blob(blob),
        ];
        store.tree(root).insert(k, &encode_row(&row)).unwrap();
    }
    store.commit().unwrap();

    let mut cidx = ColIndex::new(vec!["id".to_string()]);
    cidx.ensure_built(&store.tree(root), &catalog.column_names("records").unwrap())
        .unwrap();

    let run = |sql: &str, cidx: Option<&mut ColIndex>| {
        let select = match parse(sql).unwrap() {
            Stmt::Select(s) => s,
            _ => unreachable!(),
        };
        let plan = plan_select(&select, &catalog, vec![], None).unwrap();
        execute_select(&plan, &store, &catalog, &root_map, None, cidx, None).unwrap()
    };

    // The batch-get shape: SELECT <wide cols> FROM records WHERE id IN (...).
    let sql = "SELECT id , tier , literal_surface , embedding FROM records \
               WHERE id IN ( 'id-0007' , 'id-0042' , 'id-0199' , 'id-9999' )";
    let unindexed = run(sql, None);
    store.reset_full_scan_count();
    let indexed = run(sql, Some(&mut cidx));
    assert_eq!(
        store.full_scan_count(),
        0,
        "an indexed id IN lookup must be scan-free"
    );
    let to_set = |rs: &ResultSet| {
        let mut v: Vec<Vec<(String, Value)>> = rs.rows.clone();
        v.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        v
    };
    assert_eq!(
        to_set(&indexed),
        to_set(&unindexed),
        "indexed id IN lookup must return the same rows as the scan"
    );
    assert_eq!(indexed.columns, unindexed.columns);
    // Three of four ids exist; the absent 'id-9999' contributes nothing.
    assert_eq!(indexed.rows.len(), 3, "exactly the present ids resolve");
}

#[test]
fn records_col_eq_literal_with_col_index_is_scan_free_and_matches_unindexed() {
    // The pending-marker probe filters `WHERE embedding_pending = 1`. With a
    // ColIndex over that column the `col = <literal>` equality resolves by a single
    // index probe instead of a whole-table scan; the result equals the scan.
    use lilliengine::exec::ColIndex;
    use lilliengine::meta::MetaTable;
    use lilliengine::rowcodec::encode_row;
    use tempfile::tempdir;

    const DDL_R: &str = "CREATE TABLE IF NOT EXISTS records ( \
        vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
        embedding BLOB , embedding_pending INTEGER )";
    let dir = tempdir().unwrap();
    let store = Store::open(dir.path().join("r.lilli")).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    let root = meta
        .create_table(&store, &mut catalog, &mut root_map, DDL_R)
        .unwrap();

    store.begin_write().unwrap();
    for k in 1..=300i64 {
        let pending = if k % 75 == 0 {
            Value::Int(1)
        } else {
            Value::Int(0)
        };
        let row = vec![
            Value::Int(k),
            Value::Text(format!("id-{k:04}")),
            Value::Blob(vec![0u8; 64]),
            pending,
        ];
        store.tree(root).insert(k, &encode_row(&row)).unwrap();
    }
    store.commit().unwrap();

    let mut cidx = ColIndex::new(vec!["id".to_string(), "embedding_pending".to_string()]);
    cidx.ensure_built(&store.tree(root), &catalog.column_names("records").unwrap())
        .unwrap();

    let run = |sql: &str, cidx: Option<&mut ColIndex>| {
        let select = match parse(sql).unwrap() {
            Stmt::Select(s) => s,
            _ => unreachable!(),
        };
        let plan = plan_select(&select, &catalog, vec![], None).unwrap();
        execute_select(&plan, &store, &catalog, &root_map, None, cidx, None).unwrap()
    };

    let sql = "SELECT id , COALESCE ( embedding_pending , 0 ) AS p FROM records \
               WHERE embedding_pending = 1 ORDER BY rowid DESC LIMIT 10";
    let unindexed = run(sql, None);
    store.reset_full_scan_count();
    let indexed = run(sql, Some(&mut cidx));
    assert_eq!(
        store.full_scan_count(),
        0,
        "an indexed col = literal must be scan-free"
    );
    let to_set = |rs: &ResultSet| {
        let mut v: Vec<Vec<(String, Value)>> = rs.rows.clone();
        v.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        v
    };
    assert_eq!(to_set(&indexed), to_set(&unindexed));
    // 75,150,225,300 -> 4 pending rows.
    assert_eq!(indexed.rows.len(), 4);
}

#[test]
fn select_star_limit_zero_is_scan_free_and_returns_columns() {
    // `SELECT * FROM <table> LIMIT 0` is a column-shape probe: it must return zero
    // rows and the full column list WITHOUT scanning the table. A merge-insert path
    // issues this once per batch to count the destination's columns; on a large
    // table a scan-per-probe is a per-write full-scan tax. The result is empty
    // regardless of contents, so the engine short-circuits before touching a row.
    use lilliengine::exec::execute_select;
    use lilliengine::meta::MetaTable;
    use lilliengine::rowcodec::encode_row;
    use tempfile::tempdir;

    const DDL_E: &str = "CREATE TABLE IF NOT EXISTS edges ( src TEXT , dst TEXT , \
        edge_type TEXT , weight REAL , PRIMARY KEY ( src , dst , edge_type ) )";
    let dir = tempdir().unwrap();
    let store = Store::open(dir.path().join("e.lilli")).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    let root = meta
        .create_table(&store, &mut catalog, &mut root_map, DDL_E)
        .unwrap();

    store.begin_write().unwrap();
    for i in 0..500i64 {
        let row = vec![
            Value::Text(format!("n{i}")),
            Value::Text(format!("n{}", (i + 1) % 500)),
            Value::Text("relates".into()),
            Value::Float(0.5),
        ];
        store.tree(root).insert(i + 1, &encode_row(&row)).unwrap();
    }
    store.commit().unwrap();

    let run = |sql: &str| {
        let select = match parse(sql).unwrap() {
            Stmt::Select(s) => s,
            _ => unreachable!(),
        };
        let plan = plan_select(&select, &catalog, vec![], None).unwrap();
        execute_select(&plan, &store, &catalog, &root_map, None, None, None).unwrap()
    };

    store.reset_full_scan_count();
    let rs = run("SELECT * FROM edges LIMIT 0");
    assert_eq!(
        store.full_scan_count(),
        0,
        "SELECT * ... LIMIT 0 must be scan-free (column-shape probe never touches a row)"
    );
    assert!(rs.rows.is_empty(), "LIMIT 0 yields zero rows");
    assert_eq!(
        rs.columns,
        vec!["src", "dst", "edge_type", "weight"],
        "LIMIT 0 still reports the full column list for the probe"
    );

    // A positive LIMIT on the same table still returns rows (the short-circuit is
    // strictly LIMIT 0, not any LIMIT).
    store.reset_full_scan_count();
    let rs2 = run("SELECT * FROM edges LIMIT 3");
    assert_eq!(rs2.rows.len(), 3, "LIMIT 3 returns three rows");
    assert_eq!(rs2.columns, vec!["src", "dst", "edge_type", "weight"]);
}

#[test]
fn aliased_scan_early_limit_matches_full_scan_first_n() {
    // Without an ORDER BY the aliased path applies LIMIT in scan order without
    // sorting: stopping the scan once LIMIT survivors are collected returns the same
    // rows as scanning the whole table and truncating. With an ORDER BY the path
    // sorts the survivors before the LIMIT, returning the correctly-sorted top-N.
    use lilliengine::meta::MetaTable;
    use lilliengine::rowcodec::encode_row;
    use tempfile::tempdir;

    const DDL_R: &str = "CREATE TABLE IF NOT EXISTS records ( \
        vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
        tier TEXT , embedding BLOB )";
    let dir = tempdir().unwrap();
    let store = Store::open(dir.path().join("r.lilli")).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    let root = meta
        .create_table(&store, &mut catalog, &mut root_map, DDL_R)
        .unwrap();

    store.begin_write().unwrap();
    for k in 1..=400i64 {
        let row = vec![
            Value::Int(k),
            Value::Text(format!("id-{k:04}")),
            Value::Text("episodic".into()),
            Value::Blob(vec![(k % 256) as u8; 64]),
        ];
        store.tree(root).insert(k, &encode_row(&row)).unwrap();
    }
    store.commit().unwrap();

    let run = |sql: &str| {
        let select = match parse(sql).unwrap() {
            Stmt::Select(s) => s,
            _ => unreachable!(),
        };
        let plan = plan_select(&select, &catalog, vec![], None).unwrap();
        execute_select(&plan, &store, &catalog, &root_map, None, None, None).unwrap()
    };

    let ids_of = |rs: &ResultSet| -> Vec<String> {
        rs.rows
            .iter()
            .map(|r| match &r.iter().find(|(c, _)| c == "id").unwrap().1 {
                Value::Text(s) => s.clone(),
                _ => unreachable!(),
            })
            .collect()
    };

    // No ORDER BY: the coalesce alias forces the aliased executor and the early
    // LIMIT keeps the first 5 episodic rows in scan order — vec_label 1..=5.
    let scan_order = run("SELECT id , COALESCE ( tier , 'x' ) AS tier FROM records \
         WHERE tier = 'episodic' LIMIT 5");
    assert_eq!(scan_order.rows.len(), 5);
    assert_eq!(
        ids_of(&scan_order),
        vec!["id-0001", "id-0002", "id-0003", "id-0004", "id-0005"],
        "without ORDER BY, the early LIMIT returns the first survivors in scan order"
    );

    // With ORDER BY rowid DESC: the aliased path now sorts the survivors before the
    // LIMIT, returning the correctly-sorted top-5 (vec_label 400..=396).
    let sorted = run("SELECT id , COALESCE ( tier , 'x' ) AS tier FROM records \
         WHERE tier = 'episodic' ORDER BY rowid DESC LIMIT 5");
    assert_eq!(sorted.rows.len(), 5);
    assert_eq!(
        ids_of(&sorted),
        vec!["id-0400", "id-0399", "id-0398", "id-0397", "id-0396"],
        "with ORDER BY rowid DESC, the aliased path returns the sorted top-N"
    );
}

#[test]
fn records_id_in_with_coalesce_alias_is_scan_free_and_matches_unindexed() {
    use lilliengine::exec::ColIndex;
    use lilliengine::meta::MetaTable;
    use lilliengine::rowcodec::encode_row;
    use tempfile::tempdir;

    // The batch-get the recall path issues projects a COALESCE alias
    // (`COALESCE(embedding_pending, 0) AS embedding_pending`), which routes through
    // the aliased-projection executor. That path must also resolve `id IN (...)`
    // through the ColIndex instead of decoding the whole table.
    const DDL_R: &str = "CREATE TABLE IF NOT EXISTS records ( \
        vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
        tier TEXT , embedding BLOB , embedding_pending INTEGER )";
    let dir = tempdir().unwrap();
    let store = Store::open(dir.path().join("r.lilli")).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    let root = meta
        .create_table(&store, &mut catalog, &mut root_map, DDL_R)
        .unwrap();

    store.begin_write().unwrap();
    for k in 1..=200i64 {
        let row = vec![
            Value::Int(k),
            Value::Text(format!("id-{k:04}")),
            Value::Text("episodic".into()),
            Value::Blob(vec![(k % 256) as u8; 32]),
            Value::Null,
        ];
        store.tree(root).insert(k, &encode_row(&row)).unwrap();
    }
    store.commit().unwrap();

    let mut cidx = ColIndex::new(vec!["id".to_string()]);
    cidx.ensure_built(&store.tree(root), &catalog.column_names("records").unwrap())
        .unwrap();

    let run = |sql: &str, cidx: Option<&mut ColIndex>| {
        let select = match parse(sql).unwrap() {
            Stmt::Select(s) => s,
            _ => unreachable!(),
        };
        let plan = plan_select(&select, &catalog, vec![], None).unwrap();
        execute_select(&plan, &store, &catalog, &root_map, None, cidx, None).unwrap()
    };

    let sql = "SELECT id , tier , COALESCE ( embedding_pending , 0 ) AS embedding_pending \
               FROM records WHERE id IN ( 'id-0007' , 'id-0042' , 'id-0199' )";
    let unindexed = run(sql, None);
    store.reset_full_scan_count();
    let indexed = run(sql, Some(&mut cidx));
    assert_eq!(
        store.full_scan_count(),
        0,
        "an indexed id IN aliased projection must be scan-free"
    );
    let to_set = |rs: &ResultSet| {
        let mut v: Vec<Vec<(String, Value)>> = rs.rows.clone();
        v.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        v
    };
    assert_eq!(
        to_set(&indexed),
        to_set(&unindexed),
        "aliased id IN lookup must match the scan"
    );
    assert_eq!(indexed.columns, unindexed.columns);
    assert_eq!(indexed.columns, vec!["id", "tier", "embedding_pending"]);
    assert_eq!(indexed.rows.len(), 3);
}

#[test]
fn aliased_scan_decode_skip_matches_full_decode_with_big_blob() {
    // A filtering scan over a wide row with a large embedding BLOB must return the
    // same rows whether or not the unused blob column is decode-skipped. The query
    // filters and projects without touching `embedding`, so the projected scan
    // skips that body — the result must be byte-identical to a full decode.
    use lilliengine::meta::MetaTable;
    use lilliengine::rowcodec::encode_row;
    use tempfile::tempdir;

    const DDL_R: &str = "CREATE TABLE IF NOT EXISTS records ( \
        vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
        tier TEXT , embedding BLOB , embedding_pending INTEGER )";
    let dir = tempdir().unwrap();
    let store = Store::open(dir.path().join("r.lilli")).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    let root = meta
        .create_table(&store, &mut catalog, &mut root_map, DDL_R)
        .unwrap();

    store.begin_write().unwrap();
    for k in 1..=300i64 {
        let pending = if k % 50 == 0 {
            Value::Int(1)
        } else {
            Value::Int(0)
        };
        let row = vec![
            Value::Int(k),
            Value::Text(format!("id-{k:04}")),
            Value::Text(if k % 2 == 0 {
                "episodic".into()
            } else {
                "semantic".into()
            }),
            Value::Blob(vec![(k % 256) as u8; 1500]),
            pending,
        ];
        store.tree(root).insert(k, &encode_row(&row)).unwrap();
    }
    store.commit().unwrap();

    let run = |sql: &str| {
        let select = match parse(sql).unwrap() {
            Stmt::Select(s) => s,
            _ => unreachable!(),
        };
        let plan = plan_select(&select, &catalog, vec![], None).unwrap();
        // No col-index passed: the aliased path falls to the decode-skip scan.
        execute_select(&plan, &store, &catalog, &root_map, None, None, None).unwrap()
    };

    // The pending-marker shape: filter on tier + a coalesced pending flag, project
    // wide cols but never the embedding.
    let sql = "SELECT id , tier , COALESCE ( embedding_pending , 0 ) AS embedding_pending \
               FROM records WHERE tier = 'episodic' AND embedding_pending = 1 LIMIT 10";
    let rs = run(sql);
    // k multiples of 50 that are even: 50,100,150,200,250,300 -> all even -> 6 rows.
    assert_eq!(rs.rows.len(), 6, "exactly the matching rows survive");
    assert_eq!(rs.columns, vec!["id", "tier", "embedding_pending"]);
    for row in &rs.rows {
        let tier = &row.iter().find(|(c, _)| c == "tier").unwrap().1;
        assert_eq!(tier, &Value::Text("episodic".into()));
        let pend = &row
            .iter()
            .find(|(c, _)| c == "embedding_pending")
            .unwrap()
            .1;
        assert_eq!(pend, &Value::Int(1));
    }
    // A wide projection that DOES include the blob still returns it intact via the
    // two-phase survivor decode.
    let sql2 = "SELECT id , embedding , COALESCE ( embedding_pending , 0 ) AS p \
                FROM records WHERE embedding_pending = 1 LIMIT 3";
    let rs2 = run(sql2);
    assert_eq!(rs2.rows.len(), 3);
    for row in &rs2.rows {
        let emb = &row.iter().find(|(c, _)| c == "embedding").unwrap().1;
        match emb {
            Value::Blob(b) => assert_eq!(b.len(), 1500, "survivor blob fully decoded"),
            other => panic!("embedding not a blob: {other:?}"),
        }
    }
}

#[test]
fn coalesce_projection_with_alias() {
    let f = Fixture::new(&[
        row(1, "a", Some(5), None, None),
        row(2, "b", None, None, None),
    ]);
    let rs = f.run("SELECT COALESCE ( n , 0 ) AS nz FROM t ORDER BY id", vec![]);
    assert_eq!(rs.columns, vec!["nz"]);
    assert_eq!(col_vals(&rs, "nz"), vec![Value::Int(5), Value::Int(0)]);
}

// ---------------------------------------------------------------------------
// Aggregates + GROUP BY + HAVING
// ---------------------------------------------------------------------------

#[test]
fn count_star_uses_decode_free_fast_path() {
    let f = Fixture::new(&[
        row(1, "a", Some(1), None, None),
        row(2, "b", Some(2), None, None),
        row(3, "c", Some(3), None, None),
    ]);
    let (rs, decoded) = f.run_instrumented("SELECT COUNT ( * ) FROM t");
    assert_eq!(rs.columns, vec!["COUNT(*)"]);
    assert_eq!(rs.rows[0][0].1, Value::Int(3));
    assert!(!decoded, "bare COUNT(*) must not decode row payloads");
}

#[test]
fn count_star_filtered_counts_matching_rows() {
    let f = Fixture::new(&[
        row(1, "a", Some(1), None, None),
        row(2, "b", Some(2), None, None),
        row(3, "c", None, None, None),
    ]);
    let rs = f.run("SELECT COUNT ( * ) FROM t WHERE n IS NOT NULL", vec![]);
    assert_eq!(rs.rows[0][0].1, Value::Int(2));
}

#[test]
fn sum_min_max_honor_null_semantics() {
    let f = Fixture::new(&[
        row(1, "a", Some(10), None, None),
        row(2, "b", None, None, None),
        row(3, "c", Some(4), None, None),
    ]);
    let rs = f.run("SELECT SUM ( n ) FROM t", vec![]);
    assert_eq!(rs.columns, vec!["SUM(n)"]);
    assert_eq!(rs.rows[0][0].1, Value::Int(14));
    let rs = f.run("SELECT MIN ( n ) FROM t", vec![]);
    assert_eq!(rs.rows[0][0].1, Value::Int(4));
    let rs = f.run("SELECT MAX ( n ) FROM t", vec![]);
    assert_eq!(rs.rows[0][0].1, Value::Int(10));
    // MIN over an all-NULL set -> NULL.
    let g = Fixture::new(&[row(1, "a", None, None, None)]);
    let rs = g.run("SELECT MIN ( n ) FROM t", vec![]);
    assert_eq!(rs.rows[0][0].1, Value::Null);
}

#[test]
fn group_by_with_count_and_having() {
    let f = Fixture::new(&[
        row(1, "a", Some(1), Some("x"), None),
        row(2, "b", Some(2), Some("x"), None),
        row(3, "c", Some(3), Some("y"), None),
        row(4, "d", Some(4), Some("x"), None),
        row(5, "e", Some(5), Some("z"), None),
    ]);
    // GROUP BY label, COUNT(*): x=3, y=1, z=1.
    let rs = f.run("SELECT label , COUNT ( * ) FROM t GROUP BY label", vec![]);
    assert_eq!(rs.columns, vec!["label", "COUNT(*)"]);
    let labels = col_vals(&rs, "label");
    let counts = col_vals(&rs, "COUNT(*)");
    assert_eq!(
        labels,
        vec![
            Value::Text("x".into()),
            Value::Text("y".into()),
            Value::Text("z".into())
        ]
    );
    assert_eq!(counts, vec![Value::Int(3), Value::Int(1), Value::Int(1)]);
    // HAVING COUNT(*) > 1 keeps only the x group.
    let rs = f.run(
        "SELECT label , COUNT ( * ) FROM t GROUP BY label HAVING COUNT ( * ) > 1",
        vec![],
    );
    assert_eq!(col_vals(&rs, "label"), vec![Value::Text("x".into())]);
    assert_eq!(col_vals(&rs, "COUNT(*)"), vec![Value::Int(3)]);
}

#[test]
fn dbstat_returns_a_constant_plan() {
    let f = Fixture::new(&[row(1, "a", None, None, None)]);
    let rs = f.run(
        "SELECT SUM ( pgsize ) FROM dbstat WHERE name = 'x'",
        vec![Value::Text("x".into())],
    );
    assert_eq!(rs.columns, vec!["pgsize"]);
    assert_eq!(rs.rows.len(), 1);
    match rs.rows[0][0].1 {
        Value::Int(n) => assert!(n >= 0),
        _ => panic!("dbstat estimate must be an integer"),
    }
}

#[test]
fn sum_over_real_promotes_and_compensates_like_sqlite3() {
    // SUM over a REAL column returns a REAL total that includes every float row,
    // accumulated with Kahan-Babuška-Neumaier compensation so the low bits match
    // stdlib sqlite3 (a naive running sum of 0.9+0.5+0.1+0.7+0.7 yields
    // 2.9000000000000004; the compensated sum yields 2.9, what sqlite3 returns).
    const DDL_R: &str = "CREATE TABLE r ( vec_label INTEGER PRIMARY KEY AUTOINCREMENT , \
        id TEXT NOT NULL UNIQUE , score REAL , n INTEGER )";
    let dir = tempdir().unwrap();
    let store = Store::open(dir.path().join("r.lilli")).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    let root = meta
        .create_table(&store, &mut catalog, &mut root_map, DDL_R)
        .unwrap();
    let scores = [0.9_f64, 0.5, 0.1, 0.7, 0.7];
    store.begin_write().unwrap();
    for (i, sc) in scores.iter().enumerate() {
        let key = (i + 1) as i64;
        let payload = encode_row(&[
            Value::Int(key),
            Value::Text(format!("id-{i}")),
            Value::Float(*sc),
            Value::Int(i as i64),
        ]);
        store.tree(root).insert(key, &payload).unwrap();
    }
    store.commit().unwrap();

    let run = |sql: &str| -> ResultSet {
        let stmt = parse(sql).unwrap();
        let select = match stmt {
            Stmt::Select(s) => s,
            _ => panic!("not a select"),
        };
        let plan = plan_select(&select, &catalog, vec![], None).unwrap();
        execute_select(&plan, &store, &catalog, &root_map, None, None, None).unwrap()
    };

    // SUM(score) promotes to REAL and equals the compensated total (== sqlite3).
    let rs = run("SELECT SUM ( score ) FROM r");
    assert_eq!(rs.rows[0][0].1, Value::Float(2.9));
    // SUM over an all-integer column stays an integer.
    let rs = run("SELECT SUM ( n ) FROM r");
    assert_eq!(rs.rows[0][0].1, Value::Int(1 + 2 + 3 + 4));

    // SUM over an empty / all-NULL column is NULL (sqlite3 semantics), not 0.
    const DDL_E: &str = "CREATE TABLE e ( vec_label INTEGER PRIMARY KEY AUTOINCREMENT , \
        id TEXT NOT NULL UNIQUE , score REAL )";
    let dir2 = tempdir().unwrap();
    let store2 = Store::open(dir2.path().join("e.lilli")).unwrap();
    let meta2 = MetaTable::open(&store2).unwrap();
    let mut catalog2 = Catalog::new();
    let mut root_map2: BTreeMap<String, u32> = BTreeMap::new();
    meta2
        .create_table(&store2, &mut catalog2, &mut root_map2, DDL_E)
        .unwrap();
    let stmt = parse("SELECT SUM ( score ) FROM e").unwrap();
    let select = match stmt {
        Stmt::Select(s) => s,
        _ => unreachable!(),
    };
    let plan = plan_select(&select, &catalog2, vec![], None).unwrap();
    let rs = execute_select(&plan, &store2, &catalog2, &root_map2, None, None, None).unwrap();
    assert_eq!(
        rs.rows[0][0].1,
        Value::Null,
        "empty SUM must be NULL, not 0"
    );
}
