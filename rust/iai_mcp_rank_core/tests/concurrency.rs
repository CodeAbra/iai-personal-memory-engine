//! Concurrency proof over the columnar `RankIndex`: the GIL-forward-progress
//! contract and the non-torn-swap contract, both exercised
//! through the real PyO3 pymethod dispatch (`snapshot`/`feed`), not a
//! bare-Rust `DoubleBuffer` call -- a missing `py.detach` in the pymethod
//! body must fail this test, which a Rust-only call around the crate's
//! internals could not detect.
//!
//! Embeds a Python interpreter (`pyo3`'s `auto-initialize` dev-feature); the
//! interpreter is compiled against a plain system Python, so the venv's
//! `numpy` is put on `PYTHONPATH` before the first GIL acquisition of the
//! process.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

const DIM: usize = 384;
const N_RECORDS: usize = 12_000;
const WORDS_PER_DOC: usize = 15;
const VOCAB_SIZE: u32 = 400;
const EDGES_PER_RECORD: usize = 3;
const BURST_N: usize = 200;
// Calibrated against a warm run on dev hardware (~150-400 ms for a full
// wholesale CSR rebuild at N_RECORDS); set with headroom so this asserts
// "reliably slow", not a wall-clock lock.
const SLOW_FLOOR_MS: u64 = 40;
// A GIL-released background thread manages thousands of acquire/increment
// cycles in the time a wholesale rebuild takes; a floor two orders of
// magnitude below that catches "GIL never released" without being flaky
// under CI load.
const FORWARD_PROGRESS_FLOOR: u64 = 20;

static INIT: std::sync::Once = std::sync::Once::new();

// cargo test runs every #[test] fn in this binary as a separate OS thread
// within ONE process, sharing ONE GIL. Two GIL-forward-progress tests
// racing each other for that same GIL is real contention, not a false
// positive -- observed empirically starving a background counter to a
// single increment across a 2s window under default (parallel) cargo test
// threading. Each test holds this lock for its full body so the two never
// contend with each other; contention from outside this process is what
// the tests are actually meant to measure.
static TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The embedded interpreter is built against whatever `python3` compiled
/// `pyo3-ffi`; that build does not know about the project's venv, so numpy
/// is unimportable until its site-packages dir is on `PYTHONPATH`. Must run
/// before the process's first `Python::attach` (the env var is read once at
/// interpreter init).
fn ensure_interpreter_ready() {
    INIT.call_once(|| {
        if std::env::var_os("PYTHONPATH").is_some() {
            return;
        }
        if let Ok(out) = std::process::Command::new("python3")
            .args(["-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
            .output()
        {
            if out.status.success() {
                let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !path.is_empty() {
                    std::env::set_var("PYTHONPATH", path);
                }
            }
        }
    });
}

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn synthetic_vector(rng: &mut u32) -> Vec<f32> {
    (0..DIM)
        .map(|_| (xorshift32(rng) as f32 / u32::MAX as f32) - 0.5)
        .collect()
}

fn synthetic_surface(doc: usize, rng: &mut u32) -> String {
    let mut words: Vec<String> = (0..WORDS_PER_DOC)
        .map(|_| format!("token{}", xorshift32(rng) % VOCAB_SIZE))
        .collect();
    words.push(format!("doc{doc}"));
    words.join(" ")
}

fn synthetic_edges(node: usize, n: usize, rng: &mut u32) -> Vec<(u128, f32, String)> {
    (1..=EDGES_PER_RECORD)
        .map(|k| {
            let neighbor = ((node + k) % n) as u128 + 1;
            let weight = 0.5 + (xorshift32(rng) as f32 / u32::MAX as f32) * 0.5;
            (neighbor, weight, "hebbian".to_string())
        })
        .collect()
}

/// Builds a `RankIndex` at generation 1 through the real Python object
/// protocol (`getattr` + `call1`), not a direct Rust call to the pymethod
/// impl -- the whole point is to exercise the actual dispatch path a
/// Python caller uses.
fn build_index(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let m = PyModule::new(py, "iai_mcp_rank_core_concurrency_test")?;
    iai_mcp_rank_core::register(py, &m)?;
    let cls = m.getattr("RankIndex")?;

    let mut rng: u32 = 0x9E37_79B9;
    let mut ids = Vec::with_capacity(N_RECORDS);
    let mut vectors_flat = Vec::with_capacity(N_RECORDS * DIM);
    let mut edges: Vec<(u128, Vec<(u128, f32, String)>)> = Vec::with_capacity(N_RECORDS);
    let mut surfaces = Vec::with_capacity(N_RECORDS);
    for i in 0..N_RECORDS {
        let id = i as u128 + 1;
        ids.push(id);
        vectors_flat.extend(synthetic_vector(&mut rng));
        surfaces.push(synthetic_surface(i, &mut rng));
        edges.push((id, synthetic_edges(i, N_RECORDS, &mut rng)));
    }
    let aaak_index = vec![String::new(); N_RECORDS];
    let created_at = vec![String::new(); N_RECORDS];
    let stability = vec![0.5f32; N_RECORDS];
    let tier = vec!["episodic".to_string(); N_RECORDS];
    let tags = vec![Vec::<String>::new(); N_RECORDS];
    let salience_level = vec![0u8; N_RECORDS];
    let centrality = vec![0.0f32; N_RECORDS];
    let pending = vec![false; N_RECORDS];

    let matrix = numpy::ndarray::Array2::from_shape_vec((N_RECORDS, DIM), vectors_flat)
        .expect("N_RECORDS * DIM matches the flat vector length by construction");
    let py_arr = matrix.into_pyarray(py);

    // 13 positional args exceed pyo3's built-in tuple `PyCallArgs` impl
    // range -- build the argument tuple element-by-element instead.
    let args: Vec<Py<PyAny>> = vec![
        DIM.into_pyobject(py)?.into_any().unbind(),
        1u64.into_pyobject(py)?.into_any().unbind(),
        ids.into_pyobject(py)?.into_any().unbind(),
        py_arr.into_any().unbind(),
        edges.into_pyobject(py)?.into_any().unbind(),
        surfaces.into_pyobject(py)?.into_any().unbind(),
        aaak_index.into_pyobject(py)?.into_any().unbind(),
        created_at.into_pyobject(py)?.into_any().unbind(),
        stability.into_pyobject(py)?.into_any().unbind(),
        tier.into_pyobject(py)?.into_any().unbind(),
        tags.into_pyobject(py)?.into_any().unbind(),
        salience_level.into_pyobject(py)?.into_any().unbind(),
        centrality.into_pyobject(py)?.into_any().unbind(),
        pending.into_pyobject(py)?.into_any().unbind(),
    ];
    let args_tuple = PyTuple::new(py, args)?;
    let obj = cls.call1(&args_tuple)?;
    Ok(obj.unbind())
}

fn feed_upsert(
    py: Python<'_>,
    bound: &Bound<'_, PyAny>,
    id: u128,
    vector: Vec<f32>,
    surface: String,
) -> PyResult<()> {
    let kwargs = PyDict::new(py);
    let arr = numpy::ndarray::Array1::from_vec(vector).into_pyarray(py);
    kwargs.set_item("vector", arr)?;
    kwargs.set_item("surface", surface)?;
    bound.call_method("feed", ("upsert", id), Some(&kwargs))?;
    Ok(())
}

struct ReadSummary {
    generation: u64,
    n_ids: usize,
    n_rows: usize,
    n_degree: usize,
}

/// Calls the real `snapshot` pymethod and reduces its bulk-accessor tuple
/// to plain counts inside the GIL scope (the `Bound` return values cannot
/// outlive `py`).
fn read_summary(bound: &Bound<'_, PyAny>, request_gen: u64) -> PyResult<ReadSummary> {
    let tokens: Vec<String> = Vec::new();
    let result = bound.call_method1("snapshot", (request_gen, tokens))?;
    let tuple: &Bound<PyTuple> = result.cast()?;
    let generation: u64 = tuple.get_item(0)?.extract()?;
    let ids: Vec<u128> = tuple.get_item(1)?.extract()?;
    let matrix = tuple.get_item(2)?;
    let shape = matrix.getattr("shape")?;
    let n_rows: usize = shape.get_item(0)?.extract()?;
    let degree: Bound<PyDict> = tuple.get_item(3)?.extract()?;
    Ok(ReadSummary {
        generation,
        n_ids: ids.len(),
        n_rows,
        n_degree: degree.len(),
    })
}

/// A second thread must make measurable forward progress while
/// `RankIndex::fold`'s wholesale CSR rebuild runs -- proves `py.detach` is
/// actually releasing the GIL, not merely that no deadlock occurs. If
/// `py.detach` were removed from the pymethod body, the background thread
/// would block for the entire rebuild and `advanced` would collapse to ~0.
///
/// `snapshot`'s stale-generation path is no longer this crate's slow
/// operation -- it applies incrementally via the overlay (O(overlay), not
/// O(corpus)); `fold` is the off-critical-path operation that still pays
/// the wholesale-rebuild cost, so the GIL-release proof targets that
/// instead. The overlay is primed with one incremental apply first so
/// `fold` has real (if small) delta work to fold, matching how a real
/// caller would invoke it after some write activity, not on a pristine
/// buffer.
#[test]
fn gil_forward_progress_during_long_fold() {
    ensure_interpreter_ready();
    let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let index: Py<PyAny> = Python::attach(|py| build_index(py)).expect("build index");

    Python::attach(|py| -> PyResult<()> {
        let bound = index.bind(py);
        let mut rng: u32 = 0x1234_5678;
        feed_upsert(
            py,
            bound,
            (N_RECORDS + 1) as u128,
            synthetic_vector(&mut rng),
            "fold-trigger".to_string(),
        )?;
        let tokens: Vec<String> = Vec::new();
        bound.call_method1("snapshot", (2u64, tokens))?;
        Ok(())
    })
    .expect("prime the overlay via one incremental apply before folding");

    let counter = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let counter_bg = counter.clone();
    let stop_bg = stop.clone();
    let handle = thread::spawn(move || {
        while !stop_bg.load(Ordering::Relaxed) {
            Python::attach(|_py| {
                counter_bg.fetch_add(1, Ordering::Relaxed);
            });
        }
    });

    // Head start so the background thread is already contending for the
    // GIL before the slow call begins.
    thread::sleep(Duration::from_millis(20));
    let before = counter.load(Ordering::Relaxed);

    let start = Instant::now();
    Python::attach(|py| {
        let bound = index.bind(py);
        bound.call_method0("fold").expect("fold must succeed");
    });
    let elapsed = start.elapsed();

    stop.store(true, Ordering::Relaxed);
    handle.join().expect("background thread must not panic");
    let advanced = counter.load(Ordering::Relaxed) - before;
    println!(
        "concurrency: fold elapsed={elapsed:?} n_records={N_RECORDS} \
         second-thread-advanced={advanced} (floor={FORWARD_PROGRESS_FLOOR})"
    );

    assert!(
        elapsed >= Duration::from_millis(SLOW_FLOOR_MS),
        "fold finished in {elapsed:?}, expected >= {SLOW_FLOOR_MS} ms at N_RECORDS={N_RECORDS} \
         so this test measures GIL release, not a wall-clock race -- widen the fixture if this fires",
    );
    assert!(
        advanced >= FORWARD_PROGRESS_FLOOR,
        "second thread advanced its counter only {advanced} times during a {elapsed:?} fold \
         (floor {FORWARD_PROGRESS_FLOOR}) -- py.detach must release the GIL during \
         RankIndex::fold's rebuild, or a concurrent thread makes no progress",
    );
}

/// Non-torn-swap: while one thread performs the generation-2 rebuild+swap,
/// a second thread repeatedly reads the still-published generation 1.
/// Every successful read must report generation == 1 with self-consistent
/// column lengths (ids == matrix rows == degree entries) even though the
/// concurrent rebuild is assembling a differently-sized generation 2 in the
/// same window -- a torn buffer would leak a mismatched count or a
/// generation field disagreeing with its own content.
#[test]
fn concurrent_reader_never_observes_torn_buffer_during_swap() {
    ensure_interpreter_ready();
    let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let index: Arc<Py<PyAny>> = Arc::new(Python::attach(|py| build_index(py)).expect("build index"));

    Python::attach(|py| -> PyResult<()> {
        let bound = index.bind(py);
        let mut rng: u32 = 0xB0BA_B0BA;
        for i in 0..BURST_N {
            let id = (N_RECORDS + i) as u128 + 1;
            feed_upsert(
                py,
                bound,
                id,
                synthetic_vector(&mut rng),
                synthetic_surface(N_RECORDS + i, &mut rng),
            )?;
        }
        Ok(())
    })
    .expect("burst feed must not fail");

    let baseline = Python::attach(|py| read_summary(index.bind(py), 1)).expect("baseline read");
    assert_eq!(baseline.generation, 1);
    assert_eq!(baseline.n_ids, N_RECORDS);
    assert_eq!(baseline.n_rows, N_RECORDS);
    assert_eq!(baseline.n_degree, N_RECORDS);

    let stop = Arc::new(AtomicBool::new(false));
    let bad_read = Arc::new(AtomicBool::new(false));
    let reads_during = Arc::new(AtomicU64::new(0));

    let reader_index = index.clone();
    let reader_stop = stop.clone();
    let reader_bad = bad_read.clone();
    let reader_reads = reads_during.clone();
    let reader = thread::spawn(move || {
        while !reader_stop.load(Ordering::Relaxed) {
            let outcome = Python::attach(|py| read_summary(reader_index.bind(py), 1));
            match outcome {
                Ok(summary) => {
                    if summary.generation != 1
                        || summary.n_ids != N_RECORDS
                        || summary.n_ids != summary.n_rows
                        || summary.n_ids != summary.n_degree
                    {
                        reader_bad.store(true, Ordering::Relaxed);
                    }
                    reader_reads.fetch_add(1, Ordering::Relaxed);
                }
                // GenerationRegression once the writer has swapped past
                // generation 1 -- an explicit refusal, never a torn read.
                Err(_) => {}
            }
            // A brief yield between reads: each read is a full Python
            // round-trip (tuple + dict construction), not a trivial
            // counter increment -- an unthrottled tight loop here starves
            // the writer's own post-detach GIL reacquire for many seconds
            // (observed empirically), which is wasted wall time, not a
            // correctness signal this test needs.
            thread::sleep(Duration::from_micros(200));
        }
    });

    thread::sleep(Duration::from_millis(5));
    let writer_result =
        Python::attach(|py| read_summary(index.bind(py), 2)).expect("writer rebuild to generation 2 must succeed");

    stop.store(true, Ordering::Relaxed);
    reader.join().expect("reader thread must not panic");
    println!(
        "concurrency: concurrent-reads-during-swap={} bad_read={}",
        reads_during.load(Ordering::Relaxed),
        bad_read.load(Ordering::Relaxed)
    );

    assert!(
        !bad_read.load(Ordering::Relaxed),
        "torn read observed: a generation-1 read returned mismatched or wrong-generation columns \
         while the generation-2 rebuild was in flight"
    );
    assert!(
        reads_during.load(Ordering::Relaxed) > 0,
        "reader thread completed no reads concurrently with the writer's rebuild -- \
         widen the fixture so this test actually exercises concurrency",
    );

    assert_eq!(writer_result.generation, 2);
    assert_eq!(writer_result.n_ids, N_RECORDS + BURST_N);
    assert_eq!(writer_result.n_ids, writer_result.n_rows);
    assert_eq!(writer_result.n_ids, writer_result.n_degree);

    let post_swap = Python::attach(|py| read_summary(index.bind(py), 2)).expect("post-swap pure read");
    assert_eq!(post_swap.n_ids, N_RECORDS + BURST_N);
    assert_eq!(post_swap.n_ids, post_swap.n_rows);
    assert_eq!(post_swap.n_ids, post_swap.n_degree);
}

/// Two threads racing `snapshot()` with different stale target generations
/// against the same published base, with one pending op queued, must never
/// lose that op. The lower target may legitimately be refused with
/// `GenerationRegression` once the higher target's commit lands first, but
/// the higher target's own commit -- and any later matching-generation read
/// -- must always carry the pending op forward.
#[test]
fn concurrent_stale_snapshots_never_drop_a_drained_pending_op() {
    ensure_interpreter_ready();
    let _serial = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let index: Arc<Py<PyAny>> = Arc::new(Python::attach(|py| build_index(py)).expect("build index"));

    let extra_id = (N_RECORDS + 1) as u128;
    let mut rng: u32 = 0xC0FF_EE01;
    Python::attach(|py| {
        feed_upsert(
            py,
            index.bind(py),
            extra_id,
            synthetic_vector(&mut rng),
            "concurrent-target-race".to_string(),
        )
    })
    .expect("queue pending op");

    let barrier = Arc::new(Barrier::new(2));

    let idx_low = index.clone();
    let barrier_low = barrier.clone();
    let low = thread::spawn(move || {
        barrier_low.wait();
        Python::attach(|py| read_summary(idx_low.bind(py), 2))
    });

    let idx_high = index.clone();
    let barrier_high = barrier.clone();
    let high = thread::spawn(move || {
        barrier_high.wait();
        Python::attach(|py| read_summary(idx_high.bind(py), 3))
    });

    let low_result = low.join().expect("low-target thread must not panic");
    let high_result = high.join().expect("high-target thread must not panic");

    // The higher target can never be refused: nothing else can have
    // published past generation 3 with only these two racing callers.
    let high_summary = high_result.expect("higher-target snapshot must always succeed");
    assert_eq!(high_summary.generation, 3);
    assert_eq!(
        high_summary.n_ids,
        N_RECORDS + 1,
        "the pending upsert queued before the race must survive into the winning commit"
    );

    // The lower target either committed generation 2 (and must also carry
    // the pending op, since it drained it) or was refused with a
    // regression once generation 3 landed first -- both are correct.
    if let Ok(low_summary) = low_result {
        assert_eq!(low_summary.n_ids, N_RECORDS + 1);
    }

    let post = Python::attach(|py| read_summary(index.bind(py), 3)).expect("post-race matching-generation read");
    assert_eq!(
        post.n_ids,
        N_RECORDS + 1,
        "final published state must still contain the pending op after the race"
    );
}
