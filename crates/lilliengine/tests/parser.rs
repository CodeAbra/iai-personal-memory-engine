//! Parser parity tests: the Rust recursive-descent parser must produce the
//! reference AST shape for every supported statement and fail loud — with the
//! reference's verbatim message text — on every shape outside the subset.

use lilliengine::ast::{Expr, Stmt};
use lilliengine::error::EngineError;
use lilliengine::parser::parse;
use lillibrain::Value;

fn select(stmt: Stmt) -> lilliengine::ast::SelectStmt {
    match stmt {
        Stmt::Select(s) => s,
        other => panic!("expected Select, got {other:?}"),
    }
}

fn insert(stmt: Stmt) -> lilliengine::ast::InsertStmt {
    match stmt {
        Stmt::Insert(s) => s,
        other => panic!("expected Insert, got {other:?}"),
    }
}

fn ddl(stmt: Stmt) -> lilliengine::ast::DdlStmt {
    match stmt {
        Stmt::Ddl(s) => s,
        other => panic!("expected Ddl, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// DDL
// ---------------------------------------------------------------------------

#[test]
fn create_table_with_column_and_table_constraints() {
    let d = ddl(
        parse("CREATE TABLE records (vec_label INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE)")
            .unwrap(),
    );
    assert_eq!(d.kind, "create_table");
    assert!(d.raw.starts_with("CREATE TABLE records ("));
}

#[test]
fn create_table_composite_primary_key() {
    let d = ddl(
        parse("CREATE TABLE edges (src INTEGER, dst INTEGER, edge_type TEXT, PRIMARY KEY (src,dst,edge_type))")
            .unwrap(),
    );
    assert_eq!(d.kind, "create_table");
}

#[test]
fn create_index_if_not_exists() {
    let d = ddl(parse("CREATE INDEX IF NOT EXISTS i ON records(id)").unwrap());
    assert_eq!(d.kind, "create_index");
}

#[test]
fn drop_table_if_exists() {
    let d = ddl(parse("DROP TABLE IF EXISTS foo").unwrap());
    assert_eq!(d.kind, "drop_table");
}

#[test]
fn alter_rename_add_drop() {
    assert_eq!(ddl(parse("ALTER TABLE r RENAME TO a").unwrap()).kind, "alter_rename_table");
    assert_eq!(ddl(parse("ALTER TABLE r ADD COLUMN x INTEGER").unwrap()).kind, "alter_add_column");
    assert_eq!(ddl(parse("ALTER TABLE r DROP COLUMN x").unwrap()).kind, "alter_drop_column");
}

#[test]
fn pragma_passthrough() {
    let d = ddl(parse("PRAGMA table_info(records)").unwrap());
    assert_eq!(d.kind, "pragma");
}

// ---------------------------------------------------------------------------
// SELECT
// ---------------------------------------------------------------------------

#[test]
fn select_plain_columns() {
    let s = select(parse("SELECT id, name FROM t").unwrap());
    assert_eq!(s.table, "t");
    assert_eq!(s.columns, vec!["id", "name"]);
    assert!(s.select_exprs.is_empty());
}

#[test]
fn select_star() {
    let s = select(parse("SELECT * FROM t").unwrap());
    assert_eq!(s.columns, vec!["*"]);
}

#[test]
fn select_aliased_column() {
    let s = select(parse("SELECT col AS alias FROM t").unwrap());
    assert!(s.columns.is_empty());
    assert_eq!(s.select_exprs.len(), 1);
    assert_eq!(s.select_exprs[0].output_name, "alias");
    assert_eq!(s.select_exprs[0].expr, Expr::Column("col".to_string()));
}

#[test]
fn select_coalesce_alias() {
    let s = select(parse("SELECT COALESCE(col, 'x') AS a FROM t").unwrap());
    assert_eq!(s.select_exprs.len(), 1);
    assert_eq!(s.select_exprs[0].output_name, "a");
    match &s.select_exprs[0].expr {
        Expr::Coalesce(args) => {
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expr::Column("col".to_string()));
            assert_eq!(args[1], Expr::Literal(Value::Text("x".to_string())));
        }
        other => panic!("expected Coalesce, got {other:?}"),
    }
}

#[test]
fn select_count_star_scalar() {
    let s = select(parse("SELECT COUNT(*) FROM t").unwrap());
    assert_eq!(s.aggregate.as_deref(), Some("count"));
    assert!(s.count_alias.is_none());
}

#[test]
fn select_count_star_aliased() {
    let s = select(parse("SELECT COUNT(*) AS n FROM t").unwrap());
    assert_eq!(s.aggregate.as_deref(), Some("count"));
    assert_eq!(s.count_alias.as_deref(), Some("n"));
}

#[test]
fn select_sum_min_max() {
    assert_eq!(select(parse("SELECT SUM(x) FROM t").unwrap()).aggregate.as_deref(), Some("sum"));
    assert_eq!(select(parse("SELECT MIN(x) FROM t").unwrap()).aggregate.as_deref(), Some("min"));
    let mx = select(parse("SELECT MAX(x) FROM t").unwrap());
    assert_eq!(mx.aggregate.as_deref(), Some("max"));
    assert_eq!(mx.aggregate_column.as_deref(), Some("x"));
}

#[test]
fn select_group_by_having() {
    let s = select(parse("SELECT c, COUNT(*) FROM t GROUP BY c HAVING COUNT(*) > 5").unwrap());
    assert_eq!(s.group_by_col.as_deref(), Some("c"));
    assert_eq!(s.group_count_alias.as_deref(), Some("COUNT(*)"));
    assert_eq!(s.group_select.len(), 2);
    assert_eq!(s.group_select[0].kind, "col");
    assert_eq!(s.group_select[1].kind, "count");
    assert_eq!(s.having_op.as_deref(), Some(">"));
    assert_eq!(s.having_value, Some(5));
}

#[test]
fn select_order_by_two_keys() {
    let s = select(parse("SELECT * FROM t ORDER BY a, b DESC").unwrap());
    let k1 = s.order_by.unwrap();
    assert_eq!(k1.col, "a");
    assert!(!k1.desc);
    let k2 = s.order_by2.unwrap();
    assert_eq!(k2.col, "b");
    assert!(k2.desc);
}

#[test]
fn select_order_by_datetime_limit_offset() {
    let s = select(parse("SELECT * FROM t ORDER BY datetime(ts) LIMIT 5 OFFSET 2").unwrap());
    let k = s.order_by.unwrap();
    assert_eq!(k.col, "ts");
    assert!(k.datetime);
    assert_eq!(s.limit, Some(5));
    assert_eq!(s.offset, Some(2));
}

#[test]
fn select_limit_offset_params() {
    let s = select(parse("SELECT * FROM t LIMIT ? OFFSET ?").unwrap());
    assert!(s.limit_is_param);
    assert!(s.offset_is_param);
}

#[test]
fn select_datetime_where_named_param() {
    let s = select(parse("SELECT * FROM t WHERE datetime(col) <= datetime(:bind)").unwrap());
    match s.where_clause.unwrap() {
        Expr::BinOp { op, left, right } => {
            assert_eq!(op, "<=");
            assert_eq!(
                *left,
                Expr::FuncCall {
                    name: "datetime".to_string(),
                    args: vec![Expr::Column("col".to_string())],
                }
            );
            assert_eq!(
                *right,
                Expr::FuncCall {
                    name: "datetime".to_string(),
                    args: vec![Expr::NamedParam("bind".to_string())],
                }
            );
        }
        other => panic!("expected BinOp, got {other:?}"),
    }
}

#[test]
fn select_where_in_list_and_like() {
    let s = select(parse("SELECT * FROM t WHERE a IN (1, 2, 3) AND b LIKE 'x%'").unwrap());
    match s.where_clause.unwrap() {
        Expr::BinOp { op, left, right } => {
            assert_eq!(op, "AND");
            assert!(matches!(*left, Expr::InList { .. }));
            assert!(matches!(*right, Expr::BinOp { ref op, .. } if op == "LIKE"));
        }
        other => panic!("expected AND BinOp, got {other:?}"),
    }
}

#[test]
fn select_where_is_not_null() {
    let s = select(parse("SELECT * FROM t WHERE x IS NOT NULL").unwrap());
    match s.where_clause.unwrap() {
        Expr::IsNull { operand, negated } => {
            assert_eq!(*operand, Expr::Column("x".to_string()));
            assert!(negated);
        }
        other => panic!("expected IsNull, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// INSERT / UPDATE / DELETE
// ---------------------------------------------------------------------------

#[test]
fn insert_on_conflict_do_update_excluded() {
    let i = insert(
        parse("INSERT INTO edges (src,dst,edge_type,weight,updated_at) VALUES (?,?,?,?,?) ON CONFLICT (src,dst,edge_type) DO UPDATE SET weight=excluded.weight, updated_at=excluded.updated_at")
            .unwrap(),
    );
    assert_eq!(i.table, "edges");
    assert_eq!(i.columns.len(), 5);
    assert!(i.values.iter().all(|v| *v == Expr::Placeholder));
    assert_eq!(i.conflict_keys, vec!["src", "dst", "edge_type"]);
    assert_eq!(i.conflict_action.as_deref(), Some("update"));
    assert_eq!(i.set_clauses.len(), 2);
    assert_eq!(i.set_clauses[0].0, "weight");
    assert_eq!(i.set_clauses[0].1, Expr::Excluded("weight".to_string()));
}

#[test]
fn insert_or_ignore() {
    let i = insert(parse("INSERT OR IGNORE INTO t (a) VALUES (1)").unwrap());
    assert!(i.or_ignore);
    assert_eq!(i.values, vec![Expr::Literal(Value::Int(1))]);
}

#[test]
fn insert_named_param_null_do_nothing() {
    let i = insert(parse("INSERT INTO t (a,b) VALUES (:x, NULL) ON CONFLICT (a) DO NOTHING").unwrap());
    assert_eq!(i.values[0], Expr::NamedParam("x".to_string()));
    assert_eq!(i.values[1], Expr::Literal(Value::Null));
    assert_eq!(i.conflict_action.as_deref(), Some("nothing"));
}

#[test]
fn insert_blob_literal_decoded() {
    let i = insert(parse("INSERT INTO t (a) VALUES (x'00ff')").unwrap());
    assert_eq!(i.values[0], Expr::Literal(Value::Blob(vec![0x00, 0xff])));
}

#[test]
fn update_set_where_id_eq() {
    let stmt = parse("UPDATE t SET a=1, b=:y WHERE id = 'foo'").unwrap();
    match stmt {
        Stmt::Update(u) => {
            assert_eq!(u.table, "t");
            assert_eq!(u.set_clauses.len(), 2);
            assert_eq!(u.set_clauses[0], ("a".to_string(), Expr::Literal(Value::Int(1))));
            assert_eq!(u.set_clauses[1], ("b".to_string(), Expr::NamedParam("y".to_string())));
            assert!(u.where_clause.is_some());
        }
        other => panic!("expected Update, got {other:?}"),
    }
}

#[test]
fn delete_where() {
    let stmt = parse("DELETE FROM t WHERE x IS NOT NULL").unwrap();
    match stmt {
        Stmt::Delete(d) => assert_eq!(d.table, "t"),
        other => panic!("expected Delete, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Fail-loud on unsupported shapes — verbatim reference message text.
// ---------------------------------------------------------------------------

fn parse_err_msg(sql: &str) -> String {
    match parse(sql) {
        Ok(s) => panic!("expected error for {sql:?}, got {s:?}"),
        Err(e) => format!("{e}"),
    }
}

#[test]
fn reject_join() {
    assert_eq!(
        parse_err_msg("SELECT a FROM t JOIN u ON t.x=u.x"),
        "JOIN (JOIN) is not supported; only single-table queries are allowed"
    );
}

#[test]
fn reject_subquery_in_from() {
    assert_eq!(
        parse_err_msg("SELECT * FROM (SELECT 1)"),
        "expected table name, got '('"
    );
}

#[test]
fn reject_cte() {
    assert_eq!(
        parse_err_msg("WITH x AS (SELECT 1) SELECT * FROM x"),
        "CTEs (WITH ... AS) are not supported in this SQL subset"
    );
}

#[test]
fn reject_window_function() {
    assert_eq!(
        parse_err_msg("SELECT row_number() OVER () FROM t"),
        "window functions (OVER) are not supported in this SQL subset"
    );
}

#[test]
fn reject_distinct() {
    assert_eq!(
        parse_err_msg("SELECT DISTINCT a FROM t"),
        "DISTINCT is not supported in this SQL subset"
    );
}

#[test]
fn reject_union() {
    assert_eq!(
        parse_err_msg("SELECT a FROM t UNION SELECT b FROM u"),
        "UNION is not supported; set operations are outside the supported grammar"
    );
}

#[test]
fn reject_three_order_by_keys() {
    assert_eq!(
        parse_err_msg("SELECT a, b, c FROM t ORDER BY a, b, c"),
        "unexpected trailing tokens after statement: ',' at position 13"
    );
}

#[test]
fn reject_subquery_in_in_list() {
    assert_eq!(
        parse_err_msg("SELECT a FROM t WHERE x IN (SELECT y FROM u)"),
        "subquery in IN (...) is not supported; use parameterized IN (?, ...) instead"
    );
}

#[test]
fn reject_unsupported_statement_type() {
    let msg = parse_err_msg("VACUUM");
    assert_eq!(msg, "unsupported statement type: 'VACUUM'");
}

#[test]
fn reject_empty_statement() {
    assert_eq!(parse_err_msg(""), "empty SQL statement");
}

#[test]
fn reject_left_join_names_the_matched_keyword() {
    assert_eq!(
        parse_err_msg("SELECT a FROM t LEFT JOIN u"),
        "JOIN (LEFT) is not supported; only single-table queries are allowed"
    );
}

#[test]
fn reject_function_call_in_select_list() {
    assert_eq!(
        parse_err_msg("SELECT foo(a) FROM t"),
        "function calls are not supported in the SELECT list; \
         use COUNT(*) or SUM(col) for aggregates, or COALESCE(col, lit)"
    );
}

#[test]
fn reject_having_without_group_by() {
    assert_eq!(
        parse_err_msg("SELECT a FROM t HAVING COUNT(*) > 1"),
        "HAVING without GROUP BY is not supported"
    );
}

#[test]
fn reject_bad_conflict_action() {
    assert_eq!(
        parse_err_msg("INSERT INTO t (a) VALUES (1) ON CONFLICT (a) DO MAYBE"),
        "expected NOTHING or UPDATE after DO, got 'MAYBE'"
    );
}

#[test]
fn reject_bad_or_action() {
    assert_eq!(
        parse_err_msg("INSERT OR FOO INTO t (a) VALUES (1)"),
        "expected IGNORE or REPLACE after OR, got 'FOO' at position 2"
    );
}

#[test]
fn reject_coalesce_wrong_arity() {
    assert_eq!(
        parse_err_msg("SELECT COALESCE(a) AS x FROM t"),
        "COALESCE expects exactly 2 arguments, got 1"
    );
}

#[test]
fn fail_loud_errors_are_parse_variant() {
    let err = parse("SELECT DISTINCT a FROM t").unwrap_err();
    assert!(matches!(err, EngineError::ParseError(_)));
}
