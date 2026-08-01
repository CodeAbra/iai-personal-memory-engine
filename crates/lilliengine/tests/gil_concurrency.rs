/// Compile-time Send assertions for the types that cross the GIL boundary
/// when `py.allow_threads` releases the GIL during store I/O.
///
/// The `run_execute` / `run_executemany` wrappers extract Python-bound bind
/// values under the GIL, then release it for the pure-Rust lock+execute span.
/// The released-GIL closure captures only Send types; these assertions pin that
/// contract at compile time so a future change that makes any of these types
/// non-Send is a build error, not a runtime panic or data race.
fn assert_send<T: Send>() {}

#[test]
fn cursor_state_is_send() {
    assert_send::<lilliengine::conn::CursorState>();
}

#[test]
fn engine_error_is_send() {
    assert_send::<lilliengine::error::EngineError>();
}

#[test]
fn value_is_send() {
    assert_send::<lillibrain::Value>();
}

#[test]
fn vec_value_is_send() {
    assert_send::<Vec<lillibrain::Value>>();
}

#[test]
fn hashmap_value_is_send() {
    assert_send::<std::collections::HashMap<String, lillibrain::Value>>();
}

/// Assert that `conn::Connection` is Send, making the `Arc<Mutex<conn::Connection>>`
/// used as `SharedConn` Sync — confirming that `&SharedConn` can be captured in
/// a released-GIL closure without a data race.
#[test]
fn connection_is_send() {
    assert_send::<lilliengine::conn::Connection>();
}
