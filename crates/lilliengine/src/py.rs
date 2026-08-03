//! The pyo3 surface mounting the engine as `iai_mcp_native.engine`.
//!
//! Exposes a sqlite3-shaped `Connection` / `Cursor` / `Row` / `RawConn` so the
//! host's storage shim drives the Rust engine unchanged under
//! `LILLI_STORAGE_DRIVER=lilli`. The `Connection` wraps the pure-Rust
//! [`conn::Connection`] behind a `Mutex` (held across threads like the record
//! store surface), and `raw_conn(read_only=)` returns a `RawConn` adapter
//! sharing the same backing connection — the daemon-down direct-access path.
//!
//! Values cross the boundary by storage class: Null↔None, Int↔int, Float↔float,
//! Text↔str, Blob↔bytes (verbatim — ciphertext and literals survive the FFI
//! without a lossy str decode). The engine takes no secret-material parameter:
//! it stores ciphertext as opaque blobs and is blind to their meaning.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};
use pyo3::{create_exception, import_exception};

use lillibrain::Value;

use crate::conn::{self, ColumnDesc, CursorState, RawAction};
use crate::error::EngineError;

import_exception!(iai_mcp.errors, ProgrammingError);
import_exception!(iai_mcp.errors, OperationalError);

create_exception!(
    iai_mcp_native.engine,
    SnapshotFenceError,
    OperationalError,
    "A read-only snapshot fence tripped under a concurrent checkpoint; retry the read. \
     Subclasses iai_mcp.errors.OperationalError so a broad `except OperationalError` \
     still catches it, while new code matches this exact class structurally."
);

/// The shared, thread-held backing connection.
type SharedConn = Arc<Mutex<conn::Connection>>;

/// Map an engine error onto the matching `iai_mcp.errors` exception.
///
/// A bind-count fault maps to `ProgrammingError`; storage corruption maps to the
/// bare `DatabaseError` (the parent, not a subclass) so a caller's
/// `except OperationalError` does not swallow disk damage; a transient storage
/// I/O fault, read-only write rejection, and parse/unsupported errors map to
/// `OperationalError` so the host sees a SQL-shaped error.
fn to_pyerr(py: Python<'_>, e: EngineError) -> PyErr {
    let msg = e.to_string();
    // A snapshot fence is transient: raise the dedicated OperationalError
    // subclass so the reader's fence-retry loop matches it structurally (not by
    // message text), while a legacy `except OperationalError` still catches it.
    if matches!(e, EngineError::SnapshotFence(_)) {
        return SnapshotFenceError::new_err(msg);
    }
    let errors_mod = match py.import_bound("iai_mcp.errors") {
        Ok(m) => m,
        Err(_) => return PyRuntimeError::new_err(msg),
    };
    let exc_name = match &e {
        EngineError::ProgrammingError(_) => "ProgrammingError",
        EngineError::Corruption { .. } => "DatabaseError",
        _ => "OperationalError",
    };
    match errors_mod.getattr(exc_name) {
        Ok(exc) => match exc.call1((msg.clone(),)) {
            Ok(inst) => PyErr::from_value_bound(inst),
            Err(_) => PyRuntimeError::new_err(msg),
        },
        Err(_) => PyRuntimeError::new_err(msg),
    }
}

/// Convert a Python object into an engine [`Value`] by storage class.
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    // bool is a Python int subclass; sqlite3 stores it as 0/1 integers.
    if let Ok(b) = obj.extract::<bool>() {
        if obj.is_instance_of::<pyo3::types::PyBool>() {
            return Ok(Value::Int(i64::from(b)));
        }
    }
    // A Python float must bind as REAL even when it is whole-valued (`1.0`):
    // `extract::<i64>()` silently truncates a whole float to an integer, which
    // would store the wrong storage class (sqlite3 keeps the REAL, and a TEXT
    // column then renders it "1.0", not "1"). Match the float instance first so
    // the REAL class is preserved across the boundary; affinity coercion (if any)
    // is applied later by the executor against the target column.
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        if let Ok(f) = obj.extract::<f64>() {
            return Ok(Value::Float(f));
        }
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Float(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::Text(s));
    }
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Value::Blob(b.as_bytes().to_vec()));
    }
    // datetime / date bind: mirror the stdlib sqlite3 default adapters so a
    // timestamp round-trips byte-identically to the stdlib driver — a datetime
    // becomes isoformat with a space separator, a date its bare isoformat.
    if let Some(text) = datetime_to_iso(obj)? {
        return Ok(Value::Text(text));
    }
    Err(PyRuntimeError::new_err(format!(
        "unsupported bind parameter type: {}",
        obj.get_type().name()?
    )))
}

/// Convert a `datetime.datetime` / `datetime.date` into the text the stdlib
/// sqlite3 default adapters produce: a datetime renders as `isoformat(sep=" ")`,
/// a date as its bare `isoformat()`. Returns `None` for any non-datetime object.
fn datetime_to_iso(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let py = obj.py();
    let dt_mod = py.import_bound("datetime")?;
    let datetime_cls = dt_mod.getattr("datetime")?;
    let date_cls = dt_mod.getattr("date")?;
    // datetime is a subclass of date, so check datetime first.
    if obj.is_instance(&datetime_cls)? {
        let iso = obj.call_method1("isoformat", (" ",))?;
        return Ok(Some(iso.extract::<String>()?));
    }
    if obj.is_instance(&date_cls)? {
        let iso = obj.call_method0("isoformat")?;
        return Ok(Some(iso.extract::<String>()?));
    }
    Ok(None)
}

/// Convert an engine [`Value`] into a Python object by storage class. Blob
/// crosses as `bytes` verbatim.
fn value_to_py<'py>(py: Python<'py>, value: &Value) -> Bound<'py, PyAny> {
    match value {
        Value::Null => py.None().into_bound(py),
        Value::Int(i) => i.into_py(py).into_bound(py),
        Value::Float(f) => f.into_py(py).into_bound(py),
        Value::Text(s) => s.into_py(py).into_bound(py),
        Value::Blob(b) => PyBytes::new_bound(py, b).into_any(),
    }
}

/// Marshal a `params` argument into either positional binds or a `:name` map.
///
/// `None` / an empty sequence → no binds. A `Mapping` (a dict) → named binds.
/// Any other sequence (tuple / list) → positional binds in order.
enum Binds {
    Positional(Vec<Value>),
    Named(std::collections::HashMap<String, Value>),
}

fn extract_binds(params: Option<&Bound<'_, PyAny>>) -> PyResult<Binds> {
    let Some(p) = params else {
        return Ok(Binds::Positional(Vec::new()));
    };
    if p.is_none() {
        return Ok(Binds::Positional(Vec::new()));
    }
    // A dict is the :name channel.
    if let Ok(dict) = p.downcast::<pyo3::types::PyDict>() {
        let mut map = std::collections::HashMap::new();
        for (key, val) in dict.iter() {
            let key: String = key.extract()?;
            map.insert(key, py_to_value(&val)?);
        }
        return Ok(Binds::Named(map));
    }
    // Otherwise a positional sequence (tuple / list).
    let mut out = Vec::new();
    for item in p.iter()? {
        out.push(py_to_value(&item?)?);
    }
    Ok(Binds::Positional(out))
}

/// A sqlite3.Row-compatible result row: dict-like (by name) AND positional.
#[pyclass(name = "Row", module = "iai_mcp_native.engine")]
pub struct Row {
    inner: conn::Row,
}

#[pymethods]
impl Row {
    /// `row[i]` (positional) or `row["col"]` (by name).
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            let len = self.inner.values.len() as isize;
            let idx = if i < 0 { len + i } else { i };
            if idx < 0 || idx >= len {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "row index out of range",
                ));
            }
            return Ok(value_to_py(py, &self.inner.values[idx as usize]));
        }
        if let Ok(name) = key.extract::<String>() {
            return match self.inner.get_name(&name) {
                Some(v) => Ok(value_to_py(py, v)),
                None => Err(pyo3::exceptions::PyKeyError::new_err(name)),
            };
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "row index must be an int or a column name",
        ))
    }

    /// The number of columns.
    fn __len__(&self) -> usize {
        self.inner.values.len()
    }

    /// Iterate VALUES in column order (sqlite3.Row compatible).
    fn __iter__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let vals: Vec<Bound<'py, PyAny>> = slf
            .inner
            .values
            .iter()
            .map(|v| value_to_py(py, v))
            .collect();
        let list = pyo3::types::PyList::new_bound(py, &vals);
        Ok(list.as_any().iter()?.into())
    }

    /// The column names in order (sqlite3.Row.keys()).
    fn keys(&self) -> Vec<String> {
        self.inner.columns.as_ref().clone()
    }
}

/// Build the DB-API `description` tuple from column descriptions.
fn description_tuple<'py>(py: Python<'py>, desc: &[ColumnDesc]) -> Bound<'py, PyTuple> {
    let cols: Vec<Bound<'py, PyTuple>> = desc
        .iter()
        .map(|c| {
            let slots: Vec<Bound<'py, PyAny>> = vec![
                c.name.clone().into_py(py).into_bound(py),
                py.None().into_bound(py),
                py.None().into_bound(py),
                py.None().into_bound(py),
                py.None().into_bound(py),
                py.None().into_bound(py),
                py.None().into_bound(py),
            ];
            PyTuple::new_bound(py, &slots)
        })
        .collect();
    PyTuple::new_bound(py, &cols)
}

/// The error text raised when an op is issued against a closed cursor/connection,
/// mirroring stdlib `sqlite3`'s "Cannot operate on a closed ..." class.
const CLOSED_CURSOR_ERR: &str = "Cannot operate on a closed cursor.";

/// The error text raised when an op is issued against a closed connection,
/// mirroring stdlib `sqlite3`'s "Cannot operate on a closed database." class.
const CLOSED_CONN_ERR: &str = "Cannot operate on a closed database.";

/// A sqlite3.Cursor-compatible cursor over a materialized result set.
///
/// Holds an `Arc` clone of the shared backing connection in an `Option` so that
/// `close()` can deterministically drop the clone (decrementing the shared
/// strong-count) rather than waiting for pyo3 finalization.
#[pyclass(name = "Cursor", module = "iai_mcp_native.engine")]
pub struct Cursor {
    conn: Option<SharedConn>,
    state: CursorState,
}

#[pymethods]
impl Cursor {
    /// Run `sql` through the bound connection and mirror the result here.
    #[pyo3(signature = (sql, params=None))]
    fn execute<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        sql: &str,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let conn = match slf.conn.as_ref() {
            Some(c) => Arc::clone(c),
            None => return Err(ProgrammingError::new_err(CLOSED_CURSOR_ERR)),
        };
        let state = run_execute(py, &conn, sql, params)?;
        slf.state = state;
        Ok(slf)
    }

    /// Run `sql` once per parameter row and mirror the result here.
    fn executemany<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        sql: &str,
        seq_of_params: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let conn = match slf.conn.as_ref() {
            Some(c) => Arc::clone(c),
            None => return Err(ProgrammingError::new_err(CLOSED_CURSOR_ERR)),
        };
        let state = run_executemany(py, &conn, sql, seq_of_params)?;
        slf.state = state;
        Ok(slf)
    }

    /// Return the next row, or `None` when exhausted.
    ///
    /// A `Py::new` allocation failure surfaces as a typed `PyErr` rather than an
    /// `unwrap` panic unwinding across the C ABI.
    fn fetchone(&mut self, py: Python<'_>) -> PyResult<Option<Py<Row>>> {
        match self.state.fetchone() {
            Some(r) => Ok(Some(Py::new(py, Row { inner: r })?)),
            None => Ok(None),
        }
    }

    /// Return all remaining rows.
    fn fetchall(&mut self, py: Python<'_>) -> PyResult<Vec<Py<Row>>> {
        self.state
            .fetchall()
            .into_iter()
            .map(|r| Py::new(py, Row { inner: r }))
            .collect()
    }

    /// Return up to `size` rows.
    fn fetchmany(&mut self, py: Python<'_>, size: usize) -> PyResult<Vec<Py<Row>>> {
        self.state
            .fetchmany(size)
            .into_iter()
            .map(|r| Py::new(py, Row { inner: r }))
            .collect()
    }

    /// The DB-API description tuple.
    #[getter]
    fn description<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        description_tuple(py, &self.state.description())
    }

    /// The affected-row count (`-1` for a SELECT).
    #[getter]
    fn rowcount(&self) -> i64 {
        self.state.rowcount
    }

    /// The injected AUTOINCREMENT label, or `None`.
    #[getter]
    fn lastrowid(&self) -> Option<i64> {
        self.state.lastrowid
    }

    /// Drop this cursor's clone of the shared connection so the shared
    /// strong-count decreases deterministically at close() (not at pyo3
    /// finalization). When this cursor is the last live holder, the file/advisory
    /// lock is released here and now. Already-materialized rows in `state` remain
    /// fetchable; a further execute on the closed cursor raises the closed-cursor
    /// error.
    fn close(&mut self) {
        if let Some(arc) = self.conn.take() {
            release_if_last(arc);
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<Row>> {
        match self.state.fetchone() {
            Some(r) => Ok(Py::new(py, Row { inner: r })?),
            None => Err(PyStopIteration::new_err(())),
        }
    }
}

impl Drop for Cursor {
    /// Backstop the deterministic last-holder release for a cursor dropped
    /// without an explicit close() — when this cursor still holds the last clone
    /// of the shared connection, release the file/lock now rather than leaving it
    /// pinned until a later GC. A cursor whose clone was already surrendered by
    /// close() has nothing to release here.
    fn drop(&mut self) {
        if let Some(arc) = self.conn.take() {
            release_if_last(arc);
        }
    }
}

/// The auto-commit raw-access adapter (the daemon-down direct path).
///
/// Holds an `Arc` clone of the shared backing connection in an `Option` so that
/// `close()` can deterministically drop the clone (decrementing the shared
/// strong-count) rather than waiting for pyo3 finalization.
#[pyclass(name = "RawConn", module = "iai_mcp_native.engine")]
pub struct RawConn {
    conn: Option<SharedConn>,
    gate: conn::RawConn,
    #[pyo3(get, set)]
    row_factory: Option<Py<PyAny>>,
}

#[pymethods]
impl RawConn {
    /// Execute `sql` under auto-commit semantics. `PRAGMA query_only` flips this
    /// adapter's own flag; a write under read-only raises the stdlib readonly
    /// error; everything else delegates to the shared connection.
    #[pyo3(signature = (sql, params=None))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        sql: &str,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<Cursor>> {
        let conn = match self.conn.as_ref() {
            Some(c) => Arc::clone(c),
            None => return Err(ProgrammingError::new_err(CLOSED_CURSOR_ERR)),
        };
        let state = match self.gate.gate(sql) {
            RawAction::Empty => CursorState::empty(),
            RawAction::ReportQueryOnly(ro) => CursorState::query_only_report(ro),
            RawAction::Reject => {
                return Err(to_pyerr(
                    py,
                    EngineError::parse(conn::READONLY_WRITE_ERR.to_string()),
                ));
            }
            RawAction::Delegate(delegated) => run_execute(py, &conn, &delegated, params)?,
        };
        Py::new(
            py,
            Cursor {
                conn: Some(conn),
                state,
            },
        )
    }

    /// No-op: each execute already committed.
    fn commit(&self) {}

    /// Re-arm the backing read-only connection to the newest committed
    /// snapshot in place — the freshness path that replaces a close + reopen.
    /// Unchanged tables keep their adopted indexes; changed tables re-adopt
    /// the writer-published pair at the new generation. Returns True when the
    /// visible snapshot moved. Raises on a read-write backing connection.
    fn refresh(&mut self, py: Python<'_>) -> PyResult<bool> {
        let conn = match self.conn.as_ref() {
            Some(c) => Arc::clone(c),
            None => return Err(ProgrammingError::new_err(CLOSED_CURSOR_ERR)),
        };
        let result = py.allow_threads(|| {
            let mut guard = lock(&conn);
            guard.refresh_read_view()
        });
        result.map_err(|e| to_pyerr(py, e))
    }

    /// Whether the backing connection holds a built column index for `table`.
    fn col_index_ready(&self, table: &str) -> PyResult<bool> {
        let conn = match self.conn.as_ref() {
            Some(c) => Arc::clone(c),
            None => return Err(ProgrammingError::new_err(CLOSED_CURSOR_ERR)),
        };
        let guard = lock(&conn);
        Ok(guard.col_index_ready(table))
    }

    /// Whether the backing connection holds a built id index for `table`.
    fn id_index_ready(&self, table: &str) -> PyResult<bool> {
        let conn = match self.conn.as_ref() {
            Some(c) => Arc::clone(c),
            None => return Err(ProgrammingError::new_err(CLOSED_CURSOR_ERR)),
        };
        let guard = lock(&conn);
        Ok(guard.id_index_ready(table))
    }

    /// Leaf cells visited by this connection's store since the last reset.
    fn cells_visited_count(&self) -> PyResult<u64> {
        let conn = match self.conn.as_ref() {
            Some(c) => Arc::clone(c),
            None => return Err(ProgrammingError::new_err(CLOSED_CURSOR_ERR)),
        };
        let guard = lock(&conn);
        Ok(guard.cells_visited_count())
    }

    /// Writer-only: ensure-build and publish the read models for every
    /// col-indexed table so reader connections adopt them immediately.
    fn publish_read_models(&self, py: Python<'_>) -> PyResult<()> {
        let conn = match self.conn.as_ref() {
            Some(c) => Arc::clone(c),
            None => return Err(ProgrammingError::new_err(CLOSED_CURSOR_ERR)),
        };
        let result = py.allow_threads(|| {
            let mut guard = lock(&conn);
            guard.publish_read_models()
        });
        result.map_err(|e| to_pyerr(py, e))
    }

    /// Drop this adapter's clone of the shared connection so the shared
    /// strong-count decreases deterministically. When this adapter is the last
    /// live holder, the file/advisory lock is released here and now; otherwise a
    /// live holder remains and that holder's close/drop performs the release.
    fn close(&mut self) {
        if let Some(arc) = self.conn.take() {
            release_if_last(arc);
        }
    }
}

impl Drop for RawConn {
    /// Backstop the deterministic last-holder release for a raw adapter dropped
    /// without an explicit close() — when this adapter still holds the last clone
    /// of the shared connection, release the file/lock now rather than leaving it
    /// pinned until a later GC. An adapter whose clone was already surrendered by
    /// close() has nothing to release here.
    fn drop(&mut self) {
        if let Some(arc) = self.conn.take() {
            release_if_last(arc);
        }
    }
}

/// The sqlite3-shaped connection.
///
/// Holds the shared backing connection in an `Option` so that `close()` can
/// surrender this owner's own `Arc` clone (decrementing the shared strong-count)
/// deterministically, rather than pinning the store until pyo3 finalization.
/// Once `close()` has taken the option, every method raises the closed-connection
/// error — matching stdlib `sqlite3`'s closed-connection behavior.
#[pyclass(name = "Connection", module = "iai_mcp_native.engine")]
pub struct Connection {
    inner: Option<SharedConn>,
    #[pyo3(get, set)]
    row_factory: Option<Py<PyAny>>,
}

impl Connection {
    /// Borrow the live backing connection, or raise the closed-connection error
    /// when this handle has already been closed (its `inner` taken). Every method
    /// that touches the backing store routes through this so a post-close op
    /// surfaces a typed error instead of operating on a released store.
    fn arc(&self) -> PyResult<&SharedConn> {
        self.inner
            .as_ref()
            .ok_or_else(|| ProgrammingError::new_err(CLOSED_CONN_ERR))
    }
}

#[pymethods]
impl Connection {
    /// Open (or create) the engine database at `path`. `embed_dim` is accepted
    /// for parity with the reference open signature.
    ///
    /// Opening replays the persisted DDL and loads the on-disk column-index
    /// sidecar (a CBOR decode that scales with the corpus size -- tens of
    /// megabytes at production scale). Both `path`/`embed_dim` are plain,
    /// `Send` Rust values, so the whole open runs inside `py.allow_threads`:
    /// without it this call holds the GIL for the full sidecar decode,
    /// blocking every other Python thread in the process (including a
    /// foreground recall in a sibling worker thread) for as long as the open
    /// takes. Releasing the GIL here mirrors `run_execute`/`run_executemany`'s
    /// existing discipline for the pure-Rust I/O span.
    #[staticmethod]
    #[pyo3(signature = (path, embed_dim=0))]
    fn open(py: Python<'_>, path: &str, embed_dim: i64) -> PyResult<Connection> {
        let inner = py
            .allow_threads(|| conn::Connection::open(path, embed_dim))
            .map_err(|e| to_pyerr(py, e))?;
        Ok(Connection {
            inner: Some(Arc::new(Mutex::new(inner))),
            row_factory: None,
        })
    }

    /// Open an existing store for lock-free read-only access. Takes no advisory
    /// lock, so a separate reader process can read committed rows while a writer
    /// process holds the store. Writes through the returned connection are
    /// rejected by the read-only guard.
    ///
    /// See `open`'s doc comment: the GIL is released for the same reason (the
    /// column-index sidecar load dominates open cost on a large corpus, and
    /// this path is the one taken by every RO-pool slot and every short-lived
    /// snapshot reader the recall/graph-build paths open).
    #[staticmethod]
    #[pyo3(signature = (path, embed_dim=0))]
    fn open_read_only(py: Python<'_>, path: &str, embed_dim: i64) -> PyResult<Connection> {
        let inner = py
            .allow_threads(|| conn::Connection::open_read_only(path, embed_dim))
            .map_err(|e| to_pyerr(py, e))?;
        Ok(Connection {
            inner: Some(Arc::new(Mutex::new(inner))),
            row_factory: None,
        })
    }

    /// Execute one statement and return a Cursor.
    #[pyo3(signature = (sql, params=None))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        sql: &str,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<Cursor>> {
        let inner = self.arc()?;
        let state = run_execute(py, inner, sql, params)?;
        Py::new(
            py,
            Cursor {
                conn: Some(Arc::clone(inner)),
                state,
            },
        )
    }

    /// Execute one statement once per parameter row and return a Cursor.
    fn executemany<'py>(
        &self,
        py: Python<'py>,
        sql: &str,
        seq_of_params: &Bound<'py, PyAny>,
    ) -> PyResult<Py<Cursor>> {
        let inner = self.arc()?;
        let state = run_executemany(py, inner, sql, seq_of_params)?;
        Py::new(
            py,
            Cursor {
                conn: Some(Arc::clone(inner)),
                state,
            },
        )
    }

    /// A new connection-bound cursor (starts empty).
    fn cursor(&self, py: Python<'_>) -> PyResult<Py<Cursor>> {
        let inner = self.arc()?;
        Py::new(
            py,
            Cursor {
                conn: Some(Arc::clone(inner)),
                state: CursorState::empty(),
            },
        )
    }

    /// Export this connection's built index pairs as an opaque snapshot for
    /// [`Connection::adopt_built_indexes`] on a freshly-opened reader — Arc
    /// clones only, no postings copy. Empty while a transaction is open.
    /// The GIL is released while waiting for the connection mutex: a busy
    /// writer must never stall the whole interpreter through this call.
    fn export_built_indexes(&self, py: Python<'_>) -> PyResult<IndexSnapshot> {
        let arc = self.arc()?;
        let inner = py.allow_threads(move || {
            let guard = lock(&arc);
            guard.export_built_indexes()
        });
        Ok(IndexSnapshot { inner })
    }

    /// Non-blocking variant: returns None instead of waiting when the
    /// connection mutex is held (a busy writer). Adoption is an accelerator —
    /// a reader that skips it just pays the lazy index build.
    fn try_export_built_indexes(&self, _py: Python<'_>) -> PyResult<Option<IndexSnapshot>> {
        let arc = self.arc()?;
        match arc.try_lock() {
            Ok(guard) => Ok(Some(IndexSnapshot {
                inner: guard.export_built_indexes(),
            })),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(e)) => Ok(Some(IndexSnapshot {
                inner: e.into_inner().export_built_indexes(),
            })),
        }
    }

    /// Adopt built index pairs from `snapshot` for every table whose stamped
    /// write-generation and column set match this connection's committed
    /// state; mismatches are skipped (the lazy build remains the fallback).
    /// GIL released while waiting for the (freshly-opened, uncontended)
    /// connection mutex, for the same reason as export_built_indexes.
    fn adopt_built_indexes(&self, py: Python<'_>, snapshot: &IndexSnapshot) -> PyResult<()> {
        let arc = self.arc()?;
        let inner = snapshot.inner.clone();
        py.allow_threads(move || {
            let mut guard = lock(&arc);
            guard.adopt_built_indexes(&inner);
        });
        Ok(())
    }

    /// Return a new Connection sharing this one's backing store (a clone of the
    /// inner `Arc`), so a second in-process open at the same path can borrow the
    /// single live store without opening a second pager. The shared handle is a
    /// distinct holder of the backing connection — its existence keeps the
    /// shared strong-count above one, so an owner's close() does not release the
    /// file/lock while a borrower is still live; the file/lock is released only
    /// when the last shared handle closes (or drops).
    fn share(&self) -> PyResult<Connection> {
        let inner = self.arc()?;
        Ok(Connection {
            inner: Some(Arc::clone(inner)),
            row_factory: None,
        })
    }

    /// Return an auto-commit raw adapter backed by this connection, enforcing
    /// the `read_only` constraint at the adapter layer — the surface
    /// `get_lilli_raw_conn` calls for the daemon-down direct-access path.
    #[pyo3(signature = (read_only=false))]
    fn raw_conn(&self, py: Python<'_>, read_only: bool) -> PyResult<Py<RawConn>> {
        let inner = self.arc()?;
        Py::new(
            py,
            RawConn {
                conn: Some(Arc::clone(inner)),
                gate: conn::RawConn::new(read_only),
                row_factory: None,
            },
        )
    }

    /// Commit the open write transaction, if any.
    fn commit(&self, py: Python<'_>) -> PyResult<()> {
        let mut guard = lock(self.arc()?);
        guard.commit().map_err(|e| to_pyerr(py, e))
    }

    /// Roll back the open write transaction, if any.
    fn rollback(&self, py: Python<'_>) -> PyResult<()> {
        let mut guard = lock(self.arc()?);
        guard.rollback().map_err(|e| to_pyerr(py, e))
    }

    /// Close the connection (an uncommitted transaction is discarded).
    ///
    /// The flush/rollback half always runs so committed data is durable. The
    /// owner's own `Arc` clone is then surrendered (taken out of `inner`) — that
    /// is what makes the file/advisory-lock release happen at the TRUE last close
    /// rather than at garbage collection: if no other holder remains
    /// (`strong_count == 1` after the take), the file/lock is released here and
    /// now; if a borrower, cursor, or raw adapter is still live, surrendering this
    /// hold drops the count toward one so the LAST of those holders releases the
    /// file deterministically at ITS close/drop, never pinned by this handle.
    /// Idempotent: a second close() finds `inner` already taken and is a no-op.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(arc) = self.inner.take() else {
            return Ok(()); // already closed — no-op
        };
        {
            let mut guard = lock(&arc);
            guard.flush_only().map_err(|e| to_pyerr(py, e))?;
            // After the take, `arc` is this handle's surrendered clone. When it is
            // the sole remaining holder, release the file/lock deterministically
            // now. Otherwise a live holder remains, and that holder's own
            // close/drop performs the last-holder release.
            if Arc::strong_count(&arc) == 1 {
                guard.release_file().map_err(|e| to_pyerr(py, e))?;
            }
        }
        // `arc` (and its guard) drop here, decrementing the shared strong-count.
        Ok(())
    }

    /// True once this handle has been closed or its backing store file/lock has
    /// been released. The borrow registry reads this to avoid handing a released
    /// connection back out to a later open at the same path. A handle whose
    /// `inner` was taken by close() reports closed without touching the store.
    #[getter]
    fn is_closed(&self) -> bool {
        match self.inner.as_ref() {
            None => true,
            Some(inner) => !lock(inner).is_open(),
        }
    }

    /// Set `table`'s vec_label high-water mark to an absolute `value`, persisting
    /// it so the next auto-assigned label is `value + 1`. Called after a verbatim
    /// bulk migration that wrote explicit `vec_label` values, so a later
    /// auto-insert cannot collide with a copied row. Monotonic: a value not
    /// greater than the persisted mark is a no-op.
    fn set_vec_label_hwm(&self, py: Python<'_>, table: &str, value: i64) -> PyResult<()> {
        let guard = lock(self.arc()?);
        guard
            .set_vec_label_hwm(table, value)
            .map_err(|e| to_pyerr(py, e))
    }

    /// Build every declared-index table's column index and atomically persist
    /// the set to the on-disk sidecar, stamped with each table's committed
    /// write-generation. Called by the maintenance/SLEEP cadence; refuses
    /// inside an open transaction; a read-only connection no-ops.
    fn persist_col_indexes(&self, py: Python<'_>) -> PyResult<()> {
        let mut guard = lock(self.arc()?);
        guard.persist_col_indexes().map_err(|e| to_pyerr(py, e))
    }

    /// Writer-only: ensure-build and publish the read models (col/id index
    /// pair + committed row count) for every col-indexed table, so reader
    /// connections adopt them instead of paying per-query scans.
    fn publish_read_models(&self, py: Python<'_>) -> PyResult<()> {
        let inner = Arc::clone(self.arc()?);
        let result = py.allow_threads(|| {
            let mut guard = lock(&inner);
            guard.publish_read_models()
        });
        result.map_err(|e| to_pyerr(py, e))
    }

    /// Whether this connection holds a built column index for `table`.
    fn col_index_ready(&self, table: &str) -> PyResult<bool> {
        let guard = lock(self.arc()?);
        Ok(guard.col_index_ready(table))
    }

    /// Whether this connection holds a built id index for `table`.
    fn id_index_ready(&self, table: &str) -> PyResult<bool> {
        let guard = lock(self.arc()?);
        Ok(guard.id_index_ready(table))
    }

    /// True when an outer BEGIN is open and not yet committed/rolled back.
    #[getter]
    fn in_transaction(&self) -> bool {
        match self.inner.as_ref() {
            None => false,
            Some(inner) => lock(inner).in_transaction(),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Commit on success, roll back on error — leaving the connection open and
    /// reusable, matching stdlib `sqlite3.Connection.__exit__`.
    /// Return the number of whole-tree scans the backing store has issued since
    /// the last ``reset_full_scan_count`` call.  Used by tests that assert a
    /// COUNT or point-lookup did not degrade to a payload-copying full scan.
    fn full_scan_count(&self) -> PyResult<u64> {
        let guard = lock(self.arc()?);
        Ok(guard.full_scan_count())
    }

    /// Reset the whole-tree-scan counter to zero.  Call before the statement
    /// under test so the counter reflects only that statement's scans.
    fn reset_full_scan_count(&self) -> PyResult<()> {
        let guard = lock(self.arc()?);
        guard.reset_full_scan_count();
        Ok(())
    }

    /// Return the number of leaf cells the backing store's streaming scans have
    /// visited since the last ``reset_cells_visited_count`` call.  Used by tests
    /// that assert a query touches O(LIMIT) cells (index-served) rather than
    /// O(corpus) cells (a linear column-selective sweep).
    fn cells_visited_count(&self) -> PyResult<u64> {
        let guard = lock(self.arc()?);
        Ok(guard.cells_visited_count())
    }

    /// Reset the cells-visited counter to zero.  Call before the statement under
    /// test so the counter reflects only that statement's visited cells.
    fn reset_cells_visited_count(&self) -> PyResult<()> {
        let guard = lock(self.arc()?);
        guard.reset_cells_visited_count();
        Ok(())
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let mut guard = lock(self.arc()?);
        if exc_type.is_none() {
            guard.commit().map_err(|e| to_pyerr(py, e))?;
        } else {
            guard.rollback().map_err(|e| to_pyerr(py, e))?;
        }
        Ok(false)
    }
}

impl Drop for Connection {
    /// Backstop the deterministic file/lock release: if this handle still holds
    /// its `Arc` clone at drop (it was never explicitly closed) and is the last
    /// holder of the shared backing connection, release the file/lock now rather
    /// than leaving it held. A handle whose `inner` was already taken by close()
    /// has nothing to release here.
    fn drop(&mut self) {
        if let Some(arc) = self.inner.take() {
            release_if_last(arc);
        }
    }
}

/// Lock the shared connection, recovering a poisoned guard rather than turning
/// every later call into an FFI boundary panic.
fn lock(conn: &SharedConn) -> std::sync::MutexGuard<'_, conn::Connection> {
    conn.lock().unwrap_or_else(|e| e.into_inner())
}

/// Release the backing store file/advisory lock when the surrendered `arc` is the
/// last holder of the shared connection. The single place a Cursor / RawConn /
/// Connection routes its last-holder release through, so the gate is identical
/// across all three. `release_file` is idempotent, so a prior release does not
/// double-free; a non-last holder is a no-op (a live holder still needs the
/// store). The poisoned guard is recovered rather than skipped — leaking the
/// fd/lock on top of an already-degraded mutex would be the worse failure.
fn release_if_last(arc: SharedConn) {
    if Arc::strong_count(&arc) == 1 {
        let mut guard = arc.lock().unwrap_or_else(|e| e.into_inner());
        let _ = guard.release_file();
    }
    // `arc` (and its guard, if taken) drop here.
}

/// Run a single execute against the shared connection, dispatching positional vs
/// named binds, and return the resulting cursor state.
///
/// Bind extraction (Python object iteration) runs under the GIL. The
/// lock acquisition and the pure-Rust store I/O run with the GIL released
/// so concurrent Python threads are not starved during long scans. Error
/// mapping back to a Python exception re-acquires the GIL after the
/// store call returns.
fn run_execute(
    py: Python<'_>,
    conn: &SharedConn,
    sql: &str,
    params: Option<&Bound<'_, PyAny>>,
) -> PyResult<CursorState> {
    // Extract Python bind values under the GIL before releasing it.
    let binds = extract_binds(params)?;
    // Release the GIL for the pure-Rust lock + execute span. The closure
    // captures only Send data: binds by move, conn and sql by shared ref.
    let result = py.allow_threads(|| {
        let mut guard = lock(conn);
        match binds {
            Binds::Positional(p) => guard.execute(sql, p),
            Binds::Named(m) => guard.execute_named(sql, m),
        }
    });
    // Map any engine error to a Python exception; to_pyerr needs the GIL.
    result.map_err(|e| to_pyerr(py, e))
}

/// Run an executemany batch against the shared connection.
///
/// Parameter marshalling (Python object iteration) runs under the GIL.
/// The lock acquisition and the pure-Rust store I/O run with the GIL
/// released so concurrent Python threads are not starved during bulk
/// inserts. Error mapping back to a Python exception re-acquires the GIL.
fn run_executemany(
    py: Python<'_>,
    conn: &SharedConn,
    sql: &str,
    seq_of_params: &Bound<'_, PyAny>,
) -> PyResult<CursorState> {
    // Marshal the sequence of parameter rows under the GIL before releasing it.
    let mut seq: Vec<Vec<Value>> = Vec::new();
    for row in seq_of_params.iter()? {
        let row = row?;
        let mut params = Vec::new();
        for item in row.iter()? {
            params.push(py_to_value(&item?)?);
        }
        seq.push(params);
    }
    // Release the GIL for the pure-Rust lock + executemany span.
    let result = py.allow_threads(|| {
        let mut guard = lock(conn);
        guard.executemany(sql, seq)
    });
    // Map any engine error to a Python exception under the GIL.
    result.map_err(|e| to_pyerr(py, e))
}

/// Opaque transferable snapshot of built index pairs (see
/// `Connection.export_built_indexes` / `Connection.adopt_built_indexes`).
#[pyclass]
pub struct IndexSnapshot {
    inner: conn::IndexSnapshot,
}

/// Register the engine surface on a host module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Connection>()?;
    m.add_class::<Cursor>()?;
    m.add_class::<Row>()?;
    m.add_class::<RawConn>()?;
    m.add_class::<IndexSnapshot>()?;
    m.add(
        "SnapshotFenceError",
        m.py().get_type_bound::<SnapshotFenceError>(),
    )?;
    Ok(())
}
