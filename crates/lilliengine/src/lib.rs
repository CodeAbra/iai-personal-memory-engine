//! A SQL-subset parser and executor over the durable storage layer.
//!
//! Parses and runs the subset of SQL the host relies on, executing against the
//! record store and returning results that match stdlib sqlite3 row-for-row for
//! every statement in the supported subset.
//!
//! The frontend is split across three modules: the [`lexer`] tokenizes a SQL
//! string into a flat [`lexer::Token`] stream, the [`ast`] module declares the
//! statement and expression node set, and the [`parser`] recursive-descent
//! parser turns the token stream into an [`ast::Stmt`]. Anything outside the
//! supported subset (joins, subqueries, CTEs, window functions, set operations,
//! `DISTINCT`, `GROUP BY` on more than one column) is rejected at parse time
//! with a typed [`EngineError`] — the rejection is part of result parity.
//!
//! Literal values reuse [`lillibrain::Value`]; no second value enum is defined.

pub mod ast;
pub mod catalog;
pub mod conn;
pub mod error;
pub mod eval;
pub mod exec;
pub mod lexer;
pub mod meta;
pub mod parser;
pub mod plan;
pub mod py;
pub mod rowcodec;

pub use error::{EngineError, Result};
pub use py::register;
