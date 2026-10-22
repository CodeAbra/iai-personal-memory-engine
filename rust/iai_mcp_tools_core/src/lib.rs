//! iai_mcp_tools_core -- brain-independent whole-file lexical search engine.
//!
//! Owns its own on-disk segment format, tokenizer, and BM25 ranking, copied
//! by shape (never by dependency) from iai_mcp_vec_core/iai_mcp_rank_core so
//! this crate carries zero link-time coupling to any brain crate. Ships as
//! its own cdylib wheel module, mounted independently of iai_mcp_native.

mod ast_compaction;
mod ast_index;
mod ast_segment;
mod bm25;
mod compaction;
pub mod daemon;
pub mod exec;
mod exact_scan;
mod fusion;
mod index;
mod postings;
mod segment;
mod segmentset;
pub mod session;
mod snippet;
mod symbol_compaction;
mod symbol_index;
mod symbol_scan;
mod symbol_segment;
mod termstats;
mod tokenize;
mod vec_index;
mod xref_compaction;
mod xref_index;
mod xref_scan;
mod xref_segment;

#[cfg(feature = "extension-module")]
use std::path::PathBuf;

#[cfg(feature = "extension-module")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

/// Tokenizes every given path's content into ONE new immutable segment
/// and publishes it under `store_root`, appending to (never overwriting)
/// whatever segments are already there.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn index_paths(store_root: String, paths: Vec<String>) -> PyResult<()> {
    let store_root = PathBuf::from(store_root);
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    index::append_batch(&store_root, &paths).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Loads every active segment under `store_root` and returns up to `k`
/// (path, score) hits ranked by summed BM25 read through the resident
/// global term-stat table, highest score first.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn search(store_root: String, query: String, k: usize) -> PyResult<Vec<(String, f64)>> {
    let store_root = PathBuf::from(store_root);
    index::search(&store_root, &query, k).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Returns `(n_files, generation)` for the whole-file manifest under
/// `store_root`, read fresh off disk on every call -- never cached, never
/// re-derived by counting segments. `n_files` is `file_index.len()`: files
/// ever indexed and not since tombstoned/superseded by a re-index -- NOT
/// necessarily files still present on disk (a path deleted without a
/// re-index still counts; neither this nor compaction checks disk state).
#[cfg(feature = "extension-module")]
#[pyfunction]
fn manifest_stats(store_root: String) -> PyResult<(u64, u64)> {
    let store_root = PathBuf::from(store_root);
    index::manifest_stats(&store_root).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Same ranking as `search`, plus `(n_files, generation)` read off the
/// SAME manifest load the hits came from -- callers that must pair a
/// scope claim with these exact hits (never a second, independently-timed
/// manifest read) use this instead of `search` + `manifest_stats`.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn search_with_scope(store_root: String, query: String, k: usize) -> PyResult<(Vec<(String, f64)>, u64, u64)> {
    let store_root = PathBuf::from(store_root);
    index::search_with_scope(&store_root, &query, k).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Scans every given path's Python def/class lines into ONE new immutable
/// symbol segment and publishes it under `store_root` (the symbol lane's
/// own subdir), appending to whatever symbol segments are already there.
/// The symbol lane reads each batch file once for its own build -- it does
/// not touch the frozen whole-file `Segment::build` signature. A single-
/// read fan-out (read each batch file's bytes once and feed BOTH the
/// whole-file tokenizer and the symbol scanner) is the durable follow-up
/// optimization, deferred here because it would require touching that
/// frozen signature.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn index_symbols(store_root: String, paths: Vec<String>) -> PyResult<()> {
    let store_root = PathBuf::from(store_root);
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    symbol_index::append_batch(&store_root, &paths)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Loads every active symbol segment under `store_root` and returns up to
/// `k` (path, score) hits ranked by the exact-name/kind tier, Def-owning
/// files ranked above Unknown-mention files.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn search_symbol(store_root: String, query: String, k: usize) -> PyResult<Vec<(String, f64)>> {
    let store_root = PathBuf::from(store_root);
    symbol_index::search(&store_root, &query, k)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Returns `(n_files, generation)` for the symbol manifest under
/// `store_root` -- the symbol lane's OWN manifest, independent of the
/// whole-file lane's.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn symbol_manifest_stats(store_root: String) -> PyResult<(u64, u64)> {
    let store_root = PathBuf::from(store_root);
    symbol_index::manifest_stats(&store_root).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Symbol-lane counterpart of `search_with_scope` -- same single-load
/// rationale, the symbol store's own manifest.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn search_symbol_with_scope(store_root: String, query: String, k: usize) -> PyResult<(Vec<(String, f64)>, u64, u64)> {
    let store_root = PathBuf::from(store_root);
    symbol_index::search_with_scope(&store_root, &query, k)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Indexes `paths` into BOTH lanes in the fixed lockstep write order:
/// the whole-file lane publishes first (its own atomic rename), then the
/// symbol lane (its own atomic rename) -- never the reverse. Each lane's
/// manifest carries its own generation counter, incremented on every
/// publish; the two lanes have INDEPENDENT doc-slot spaces (different
/// corpora -- whole files vs symbol names), so no consumer ever joins
/// postings across the two lanes and a torn cross-lane read is impossible
/// by construction. The generation stamp is therefore a skew DETECTOR (a
/// caller that reads both lanes can observe a batch skew after a crash
/// between the two publishes), never a cross-lane joiner. A crash after
/// the whole-file publish but before the symbol publish leaves the
/// symbol lane exactly one batch behind -- loadable and untorn (its own
/// publish is atomic) -- and the next call to this function converges
/// both lanes to the same batch.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn index_both(store_root: String, symbol_store_root: String, paths: Vec<String>) -> PyResult<()> {
    let store_root_path = PathBuf::from(store_root);
    let symbol_store_root_path = PathBuf::from(symbol_store_root);
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    index::append_batch(&store_root_path, &paths)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    symbol_index::append_batch(&symbol_store_root_path, &paths)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

/// Indexes `paths` into all THREE lanes in the fixed lockstep write
/// order: whole-file, then symbol, then xref -- extends `index_both`'s
/// skew-detector-not-joiner posture by one more publish, never reorders
/// the first two. Each lane's own atomic rename keeps a crash mid-
/// sequence loadable and untorn; a subsequent call converges all three
/// lanes to the same batch. `index_both` is kept as-is for callers that
/// only need the first two lanes.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn index_all(
    store_root: String,
    symbol_store_root: String,
    xref_store_root: String,
    paths: Vec<String>,
) -> PyResult<()> {
    let store_root_path = PathBuf::from(store_root);
    let symbol_store_root_path = PathBuf::from(symbol_store_root);
    let xref_store_root_path = PathBuf::from(xref_store_root);
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    index::append_batch(&store_root_path, &paths)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    symbol_index::append_batch(&symbol_store_root_path, &paths)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    xref_index::append_batch(&xref_store_root_path, &paths)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

/// Scans exactly the given paths for the one recognized cross-file
/// import shape and publishes ONE new immutable edge segment under
/// `store_root` (the xref lane's own subdir, never the whole-file or
/// symbol store roots).
#[cfg(feature = "extension-module")]
#[pyfunction]
fn index_xref(store_root: String, paths: Vec<String>) -> PyResult<()> {
    let store_root = PathBuf::from(store_root);
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    xref_index::append_batch(&store_root, &paths)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Runs the whole-file BM25 search over `store_root`, then looks up every
/// file the xref lane under `xref_store_root` links to a hit. A base hit
/// is never evicted or reordered by injection: the result holds up to `k`
/// (path, score) pairs when nothing was injected, and may hold more than
/// `k` when at least one file was injected, since injected rows are
/// appended after every base hit rather than displacing one.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn search_xref(
    store_root: String,
    xref_store_root: String,
    query: String,
    k: usize,
) -> PyResult<Vec<(String, f64)>> {
    let store_root = PathBuf::from(store_root);
    let xref_store_root = PathBuf::from(xref_store_root);
    xref_index::search(&store_root, &xref_store_root, &query, k)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Maps an edge's `XrefKind` to the lowercase label the Python glue and
/// every downstream response render -- the sole conversion point, so a
/// future `XrefKind` variant is caught here at compile time.
#[cfg(feature = "extension-module")]
fn xref_kind_label(kind: xref_scan::XrefKind) -> &'static str {
    match kind {
        xref_scan::XrefKind::Import => "import",
        xref_scan::XrefKind::Call => "call",
        xref_scan::XrefKind::Unresolved => "unresolved",
    }
}

/// From-symbol reverse lookup: every edge (Import or Call, resolved or
/// not) whose reference name is `name`, exhaustive over every currently
/// loaded xref segment under `xref_store_root` -- no top-k truncation.
/// Each row is `(from_path, to_path_or_none, kind_label)`.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn definitions_for_reference(
    xref_store_root: String,
    name: String,
) -> PyResult<Vec<(String, Option<String>, String)>> {
    let xref_store_root = PathBuf::from(xref_store_root);
    xref_index::definitions_for_reference(&xref_store_root, &name)
        .map(|rows| {
            rows.into_iter()
                .map(|(from_path, to_path, kind)| {
                    (from_path, to_path, xref_kind_label(kind).to_string())
                })
                .collect()
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Scans every file in the whole-file manifest under `store_root` for the
/// exact byte sequence `literal` -- no case-folding, no tokenizer, no
/// regex. Returns `(hits, files_scanned, files_unreadable, files_truncated,
/// n_files, generation)`; `hits` is (absolute path, 1-based first-match
/// line). See `exact_scan::scan`'s doc for the three-way per-file bucket
/// accounting this proof-of-absence depends on.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn scan_literal(
    store_root: String,
    literal: String,
) -> PyResult<(Vec<(String, u32)>, u64, u64, u64, u64, u64)> {
    let store_root = PathBuf::from(store_root);
    let result =
        exact_scan::scan(&store_root, &literal).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        result.hits,
        result.files_scanned,
        result.files_unreadable,
        result.files_truncated,
        result.n_files,
        result.generation,
    ))
}

/// The literal scanner's per-file bounded-read cap, in bytes -- the SAME
/// constant `snippet::extract_snippet` reuses for its own bounded re-read
/// (`snippet::MAX_SNIPPET_READ_BYTES`), never a second cap invented for
/// this lane. Exposed so Python can assert it stays >= the index-time
/// admission cap (`study.STUDY_DIR_MAX_FILE_BYTES`) as a live regression
/// guard, rather than mirroring the value as a second, driftable constant.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn literal_scan_read_cap_bytes() -> u64 {
    snippet::MAX_SNIPPET_READ_BYTES
}

/// One file's already-`ast.parse`d facts, deserialized from the JSON
/// `ast_facts.py` hands across this seam -- mirrors `session_capture`'s
/// own `hook_json: String` + `serde_json::from_str` pattern. Rust never
/// parses Python source; it only shapes and persists what Python already
/// walked.
#[cfg(feature = "extension-module")]
#[derive(serde::Deserialize)]
struct AstFactsInput {
    files: Vec<ast_segment::FileFacts>,
}

/// Publishes ONE new AST fact segment under `ast_store_root` (the AST
/// lane's own subdir, never the whole-file/symbol/xref store roots) from
/// `facts_json`. Called as a 4th step AFTER the Python-side
/// `iai_mcp.tools.index_all` wrapper's Rust `index_all` call returns --
/// Python must run `ast.parse` before this seam is ever reached, so this
/// publish cannot be folded into the 3-lane `index_all` above.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn index_ast_facts(store_root: String, facts_json: String) -> PyResult<()> {
    let root = PathBuf::from(store_root);
    let input: AstFactsInput =
        serde_json::from_str(&facts_json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    ast_index::append_batch(&root, input.files).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Returns the subset of `paths` whose current content hash is absent
/// from or differs from the AST manifest's own `file_index` -- the
/// pre-`ast.parse` staleness gate. A path is compared by exact string
/// equality against `file_index`'s keys (the same already-resolved
/// absolute-path form the Python glue always hands in), never re-derived
/// or normalized here.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn stale_paths(ast_store_root: String, paths: Vec<String>) -> PyResult<Vec<String>> {
    let root = PathBuf::from(ast_store_root);
    ast_index::stale_paths(&root, &paths).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// `(n_files, generation)` for the AST manifest under `ast_store_root`,
/// read fresh off disk on every call -- mirrors `manifest_stats` (the
/// whole-file lane's own accessor).
#[cfg(feature = "extension-module")]
#[pyfunction]
fn ast_manifest_stats(ast_store_root: String) -> PyResult<(u64, u64)> {
    let root = PathBuf::from(ast_store_root);
    ast_index::manifest_stats(&root).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Looks up `name` as referenced from `from_path` against the AST lane's
/// own manifest, read fresh from disk. Returns `(status, path, line)`:
/// `status` is one of `"resolved"`/`"unresolved"`/`"not_indexed"`/
/// `"parse_error"` -- never a silent empty. `path`/`line` are set only
/// when `status == "resolved"`. The Python glue additionally computes
/// `"unsupported_language"` before this is ever called, for a
/// non-Python `from_path`.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn goto_definition(
    ast_store_root: String,
    name: String,
    from_path: String,
) -> PyResult<(String, Option<String>, Option<u32>)> {
    let root = PathBuf::from(ast_store_root);
    let status = ast_index::goto_definition(&root, &name, &from_path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(match status {
        ast_index::GotoStatus::Resolved { path, line } => {
            ("resolved".to_string(), Some(path), Some(line))
        }
        ast_index::GotoStatus::Unresolved => ("unresolved".to_string(), None, None),
        ast_index::GotoStatus::NotIndexed => ("not_indexed".to_string(), None, None),
        ast_index::GotoStatus::ParseError => ("parse_error".to_string(), None, None),
    })
}

/// Reverse of `goto_definition`: every reference to `name` defined in
/// `def_path`, against the AST lane's own manifest, read fresh from
/// disk. Returns `(status, rows)`: `status` is one of `"hits"`/
/// `"no_references"`/`"not_indexed"`/`"parse_error"` -- never a silent
/// empty. `rows` holds `(path, row_status)` pairs, `row_status` one of
/// `"resolved"`/`"unresolved"`, populated only when `status == "hits"`.
/// The Python glue additionally computes `"unsupported_language"` before
/// this is ever called, for a non-Python `def_path`.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn find_references(
    ast_store_root: String,
    name: String,
    def_path: String,
) -> PyResult<(String, Vec<(String, String)>)> {
    let root = PathBuf::from(ast_store_root);
    let status = ast_index::find_references(&root, &name, &def_path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(match status {
        ast_index::FindReferencesStatus::Hits(rows) => (
            "hits".to_string(),
            rows.into_iter()
                .map(|r| {
                    let row_status = match r.status {
                        ast_index::RefRowStatus::Resolved => "resolved",
                        ast_index::RefRowStatus::Unresolved => "unresolved",
                    };
                    (r.path, row_status.to_string())
                })
                .collect(),
        ),
        ast_index::FindReferencesStatus::NoReferences => ("no_references".to_string(), Vec::new()),
        ast_index::FindReferencesStatus::NotIndexed => ("not_indexed".to_string(), Vec::new()),
        ast_index::FindReferencesStatus::ParseError => ("parse_error".to_string(), Vec::new()),
    })
}

/// Classifies `hook_json` and, if it maps to a category, appends the
/// resulting event to the session's own JSONL file under `store_root`. A
/// hook payload that classify rejects (unmatched or malformed) is a no-op,
/// never an error.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn session_capture(store_root: String, hook_json: String) -> PyResult<()> {
    let root = PathBuf::from(store_root);
    let input: session::extract::HookInput =
        serde_json::from_str(&hook_json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if let Some(event) = session::extract::classify(&input) {
        let session_id = event.session_id.clone();
        session::store::append_event(&root, &session_id, &event)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }
    Ok(())
}

/// Returns every matching event in `session_id`'s file as its raw JSONL
/// line, for the Python side to `json.loads`. An invalid `session_id`
/// yields an empty list, never an error.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (store_root, session_id, substring=None, category=None, since=None, until=None))]
fn session_query(
    store_root: String,
    session_id: String,
    substring: Option<String>,
    category: Option<String>,
    since: Option<String>,
    until: Option<String>,
) -> PyResult<Vec<String>> {
    let root = PathBuf::from(store_root);
    let filter = session::store::QueryFilter {
        substring: substring.as_deref(),
        category: category.as_deref(),
        since: since.as_deref(),
        until: until.as_deref(),
    };
    let events = session::store::query_events(&root, &session_id, filter)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    events
        .iter()
        .map(|e| serde_json::to_string(e).map_err(|e| PyRuntimeError::new_err(e.to_string())))
        .collect()
}

/// Loads every event in `session_id`'s file (the I/O lives here, in the
/// pyfunction) and renders it through the pure `build_snapshot`.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn session_snapshot(store_root: String, session_id: String) -> PyResult<String> {
    let root = PathBuf::from(store_root);
    let events = session::store::load_all(&root, &session_id)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(session::snapshot::build_snapshot(&events))
}

/// No I/O: deserializes `events_json` (one JSON array of event objects, the
/// same shape `session_query` returns) and renders it directly through the
/// pure `build_snapshot` -- lets the Python side test the snapshot builder
/// against an in-memory fixture without touching a store.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn session_snapshot_from_events(events_json: String) -> PyResult<String> {
    let events: Vec<session::SessionEvent> =
        serde_json::from_str(&events_json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(session::snapshot::build_snapshot(&events))
}

/// Rebuilds the tools-owned exact-cosine vector segment under `store_root`
/// from `vectors`/`paths`/`truncated_flags` (row-aligned, same length).
/// Unlike the lexical/symbol/xref lanes this OVERWRITES the store's single
/// segment on every call -- there is no append-per-batch segment set for
/// this lane, see vec_index's module doc.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn index_vectors(
    store_root: String,
    vectors: Vec<Vec<f32>>,
    paths: Vec<String>,
    truncated_flags: Vec<bool>,
) -> PyResult<()> {
    let store_root = PathBuf::from(store_root);
    vec_index::index_vectors(&store_root, &vectors, &paths, &truncated_flags)
        .map_err(PyRuntimeError::new_err)
}

/// Loads the tools-owned exact-cosine vector segment under `store_root`
/// and returns up to `k` (path, score) hits ranked by cosine similarity,
/// highest first. `query_vector` arrives pre-embedded from the caller --
/// this function never touches the brain embedder. A missing store
/// returns an empty result; a store built at a different embedding
/// dimension raises instead of silently returning empty (see
/// vec_index::search_vector's module doc).
#[cfg(feature = "extension-module")]
#[pyfunction]
fn search_vector(
    store_root: String,
    query_vector: Vec<f32>,
    k: usize,
) -> PyResult<Vec<(String, f64)>> {
    let store_root = PathBuf::from(store_root);
    vec_index::search_vector(&store_root, &query_vector, k).map_err(PyRuntimeError::new_err)
}

/// Re-reads `path` (already a path a prior search hit returned -- never a
/// raw query-supplied path) and returns the bounded matching-line window:
/// `(line_start_1based, line_end_1based, snippet_text, elided_lines)`. See
/// `snippet::extract_snippet`'s doc for the selection and bounding rules.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn extract_snippet(
    path: String,
    query: String,
    line_cap: usize,
) -> PyResult<(usize, usize, String, usize)> {
    snippet::extract_snippet(&path, &query, line_cap)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// RRF (K=60) combine over `lanes` -- each lane already rank-ordered
/// (index 0 == rank 1) and pre-normalized by the caller to the same join
/// key. Pure function, no I/O; see fusion's module doc for the dedup-
/// and-sum contract.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn rrf_merge(lanes: Vec<Vec<String>>) -> Vec<(String, f64)> {
    fusion::rrf_merge(&lanes)
}

/// True when `base_fused`'s top score is weak -- below `threshold`, or
/// the module's default when `threshold` is omitted. `base_fused` must
/// already be sorted by score descending (the `rrf_merge` contract).
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (base_fused, threshold=None))]
fn gate_is_weak(base_fused: Vec<(String, f64)>, threshold: Option<f64>) -> bool {
    fusion::gate_is_weak(&base_fused, threshold)
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn iai_mcp_tools_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(index_paths, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(manifest_stats, m)?)?;
    m.add_function(wrap_pyfunction!(search_with_scope, m)?)?;
    m.add_function(wrap_pyfunction!(index_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(search_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(symbol_manifest_stats, m)?)?;
    m.add_function(wrap_pyfunction!(search_symbol_with_scope, m)?)?;
    m.add_function(wrap_pyfunction!(index_both, m)?)?;
    m.add_function(wrap_pyfunction!(index_all, m)?)?;
    m.add_function(wrap_pyfunction!(index_xref, m)?)?;
    m.add_function(wrap_pyfunction!(search_xref, m)?)?;
    m.add_function(wrap_pyfunction!(definitions_for_reference, m)?)?;
    m.add_function(wrap_pyfunction!(index_ast_facts, m)?)?;
    m.add_function(wrap_pyfunction!(stale_paths, m)?)?;
    m.add_function(wrap_pyfunction!(ast_manifest_stats, m)?)?;
    m.add_function(wrap_pyfunction!(goto_definition, m)?)?;
    m.add_function(wrap_pyfunction!(find_references, m)?)?;
    m.add_function(wrap_pyfunction!(scan_literal, m)?)?;
    m.add_function(wrap_pyfunction!(literal_scan_read_cap_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(session_capture, m)?)?;
    m.add_function(wrap_pyfunction!(session_query, m)?)?;
    m.add_function(wrap_pyfunction!(session_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(session_snapshot_from_events, m)?)?;
    m.add_function(wrap_pyfunction!(index_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(search_vector, m)?)?;
    m.add_function(wrap_pyfunction!(extract_snippet, m)?)?;
    m.add_function(wrap_pyfunction!(rrf_merge, m)?)?;
    m.add_function(wrap_pyfunction!(gate_is_weak, m)?)?;
    Ok(())
}
