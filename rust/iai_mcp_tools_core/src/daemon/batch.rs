//! `tool_batch_run`: runs a list of commands under one of two concurrency
//! semantics. Concurrency 1 shares ONE timeout budget across the whole
//! list -- the first command to exhaust it cascades a `Skipped` outcome
//! onto every command still waiting, none of which are ever spawned.
//! Concurrency 2-8 gives each command its OWN timeout and runs up to that
//! many at once, but always assembles the output back into INPUT order,
//! regardless of which command actually finished first.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::exec::state::RunState;
use crate::exec::{self, BoundedResult, ExecError, SpawnSpec};

use super::socket::{bounded_result_to_json, ERR_INVALID_PARAMS};

/// One batch element's outcome. Distinct from `RunState`: `Skipped` and
/// `Failed` only ever happen at the batch-coordination layer above a single
/// run, never inside `watcher::watch` itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchState {
    Exited(i32),
    Killed(i32),
    TimedOut,
    /// Never spawned at all -- the shared budget was already exhausted by
    /// an earlier command in the same concurrency-1 batch.
    Skipped,
    /// The command could not even be spawned (e.g. the artifact store
    /// could not be prepared); carries the error text for the caller.
    Failed(String),
}

pub struct BatchElement {
    pub run_id: Option<String>,
    pub state: BatchState,
    pub bounded_tail: Option<BoundedResult>,
}

fn batch_state_to_str(state: &BatchState) -> String {
    match state {
        BatchState::Exited(code) => format!("exited:{code}"),
        BatchState::Killed(sig) => format!("killed:{sig}"),
        BatchState::TimedOut => "timed_out".to_string(),
        BatchState::Skipped => "skipped".to_string(),
        BatchState::Failed(msg) => format!("failed:{msg}"),
    }
}

fn to_batch_state(state: RunState) -> BatchState {
    match state {
        RunState::Exited(code) => BatchState::Exited(code),
        RunState::Killed(sig) => BatchState::Killed(sig),
        RunState::TimedOut => BatchState::TimedOut,
        // `watcher::watch` only ever returns a terminal outcome -- `Running`
        // and `Detached` are daemon-registry-only states for background
        // runs, unreachable from this synchronous wait-to-completion path.
        // Represented defensively rather than panicking.
        RunState::Running | RunState::Detached => {
            BatchState::Failed("unexpected non-terminal state from a synchronous wait".to_string())
        }
    }
}

/// Spawns `spec`, waits up to `timeout` (with the watcher's default grace
/// window before escalating to SIGKILL), and returns its id, terminal
/// state, and initial bounded output -- the single-command building block
/// both concurrency modes below run per command, just with a different
/// `timeout` value each supplies.
fn run_one_with_timeout(
    spec: SpawnSpec,
    timeout: Duration,
) -> Result<(String, RunState, BoundedResult), ExecError> {
    exec::sandbox::ensure_sandbox_ready()?;

    let run_id = exec::artifact::new_run_id();
    let dir = exec::artifact::run_dir(&run_id)?;
    let (stdout_file, _stdout_path) = exec::artifact::create_artifact_file(&dir, "stdout.log")?;
    let (stderr_file, _stderr_path) = exec::artifact::create_artifact_file(&dir, "stderr.log")?;

    let started_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let limits = exec::rlimit::effective_limits();
    let mut spawned =
        exec::spawn::spawn_bounded(&spec.program, &spec.args, stdout_file, stderr_file, &limits)?;
    let group = spawned.group;

    let watch_config = exec::watcher::WatchConfig {
        timeout,
        grace: exec::watcher::WatchConfig::default().grace,
    };
    let run_state = exec::watcher::watch(&mut spawned.child, group, &watch_config)?;
    let exit_code = match run_state {
        RunState::Exited(code) => Some(code),
        _ => None,
    };

    let meta = exec::artifact::RunMeta {
        run_id: run_id.clone(),
        cmd: spec.program,
        args: spec.args,
        pid: spawned.child.id(),
        pgid: group.meta_pgid(),
        started_at,
        exit_code,
        stdout_offset: 0,
        stderr_offset: 0,
        rlimits_applied: cfg!(unix),
    };
    exec::artifact::write_meta(&dir, &meta)?;

    let result = exec::fetch_result(&run_id)?;
    Ok((run_id, run_state, result))
}

/// Concurrency 1: one shared budget across the whole list, run strictly in
/// sequence. Once a command times out, every command still waiting is
/// marked `Skipped` and never spawned (cascading skip) -- the point is a
/// single time-boxed pass over the list, not N independent time-boxed
/// commands.
pub fn run_batch_concurrency_1(commands: Vec<SpawnSpec>, total_budget: Duration) -> Vec<BatchElement> {
    let mut results = Vec::with_capacity(commands.len());
    let start = Instant::now();
    let mut cascading_skip = false;

    for spec in commands {
        if cascading_skip {
            results.push(BatchElement {
                run_id: None,
                state: BatchState::Skipped,
                bounded_tail: None,
            });
            continue;
        }

        let remaining = total_budget.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            cascading_skip = true;
            results.push(BatchElement {
                run_id: None,
                state: BatchState::Skipped,
                bounded_tail: None,
            });
            continue;
        }

        match run_one_with_timeout(spec, remaining) {
            Ok((run_id, state, tail)) => {
                let batch_state = to_batch_state(state);
                if batch_state == BatchState::TimedOut {
                    cascading_skip = true;
                }
                results.push(BatchElement {
                    run_id: Some(run_id),
                    state: batch_state,
                    bounded_tail: Some(tail),
                });
            }
            Err(e) => {
                results.push(BatchElement {
                    run_id: None,
                    state: BatchState::Failed(e.to_string()),
                    bounded_tail: None,
                });
            }
        }
    }

    results
}

/// Concurrency 2-8: up to `concurrency` commands run at once, each with its
/// OWN `per_command_timeout` -- a slow command times out without affecting
/// its siblings. Each worker claims the next unclaimed input index from a
/// shared counter and writes its result into that exact index of a
/// pre-sized results vector, so the final collected order always matches
/// INPUT order regardless of which command actually finished first.
pub fn run_batch_concurrency_n(
    commands: Vec<SpawnSpec>,
    concurrency: usize,
    per_command_timeout: Duration,
) -> Vec<BatchElement> {
    let n = commands.len();
    let slots: Vec<Mutex<Option<SpawnSpec>>> = commands.into_iter().map(|c| Mutex::new(Some(c))).collect();
    let results: Vec<Mutex<Option<BatchElement>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next_index = AtomicUsize::new(0);
    let worker_count = concurrency.clamp(1, 8).min(n.max(1));

    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            let next_index = &next_index;
            let slots = &slots;
            let results = &results;
            scope.spawn(move || loop {
                let idx = next_index.fetch_add(1, Ordering::SeqCst);
                if idx >= n {
                    break;
                }
                let spec = slots[idx]
                    .lock()
                    .unwrap()
                    .take()
                    .expect("each index is claimed by exactly one worker");
                let element = match run_one_with_timeout(spec, per_command_timeout) {
                    Ok((run_id, state, tail)) => BatchElement {
                        run_id: Some(run_id),
                        state: to_batch_state(state),
                        bounded_tail: Some(tail),
                    },
                    Err(e) => BatchElement {
                        run_id: None,
                        state: BatchState::Failed(e.to_string()),
                        bounded_tail: None,
                    },
                };
                *results[idx].lock().unwrap() = Some(element);
            });
        }
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap().expect("every index must be filled by exactly one worker"))
        .collect()
}

fn batch_element_to_json(el: BatchElement) -> Value {
    serde_json::json!({
        "run_id": el.run_id,
        "state": batch_state_to_str(&el.state),
        "bounded_tail": el.bounded_tail.as_ref().map(bounded_result_to_json),
    })
}

/// Wire entry point: parses `commands`/`concurrency`/`timeout_ms`, picks the
/// concurrency-1-vs-2-8 code path, and returns a JSON array aligned 1:1
/// with the input command list.
pub fn dispatch_tool_batch_run(params: &Value) -> Result<Value, (i32, String)> {
    let commands_val = params
        .get("commands")
        .and_then(Value::as_array)
        .ok_or_else(|| (ERR_INVALID_PARAMS, "missing required param: 'commands'".to_string()))?;
    if commands_val.is_empty() {
        return Err((ERR_INVALID_PARAMS, "'commands' must not be empty".to_string()));
    }

    let mut specs = Vec::with_capacity(commands_val.len());
    for c in commands_val {
        let command = c
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| (ERR_INVALID_PARAMS, "each batch command needs a 'command' string".to_string()))?
            .to_string();
        let args: Vec<String> = c
            .get("args")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        specs.push(SpawnSpec { program: command, args, timeout: None });
    }

    let concurrency = params
        .get("concurrency")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .clamp(1, 8) as usize;
    let timeout = params
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(exec::watcher::DEFAULT_TIMEOUT_SECS));

    let elements = if concurrency == 1 {
        run_batch_concurrency_1(specs, timeout)
    } else {
        run_batch_concurrency_n(specs, concurrency, timeout)
    };

    Ok(Value::Array(elements.into_iter().map(batch_element_to_json).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_tools_root<T>(f: impl FnOnce() -> T) -> T {
        let _guard = exec::artifact::ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: ENV_LOCK serializes every test that touches IAI_TOOLS_ROOT.
        unsafe {
            std::env::set_var("IAI_TOOLS_ROOT", tmp.path());
        }
        let result = f();
        unsafe {
            std::env::remove_var("IAI_TOOLS_ROOT");
        }
        result
    }

    fn spec(program: &str, args: &[&str]) -> SpawnSpec {
        SpawnSpec {
            program: program.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            timeout: None,
        }
    }

    #[test]
    fn concurrency_1_all_commands_run_when_the_shared_budget_is_generous() {
        with_tools_root(|| {
            let results = run_batch_concurrency_1(
                vec![spec("true", &[]), spec("true", &[]), spec("true", &[])],
                Duration::from_secs(5),
            );
            assert_eq!(results.len(), 3);
            for el in &results {
                assert_eq!(el.state, BatchState::Exited(0));
                assert!(el.run_id.is_some());
            }
        });
    }

    #[test]
    fn concurrency_1_cascading_skip_marks_every_remaining_command_after_a_timeout() {
        with_tools_root(|| {
            let results = run_batch_concurrency_1(
                vec![
                    spec("true", &[]),
                    spec("sh", &["-c", "sleep 5"]),
                    spec("true", &[]),
                    spec("true", &[]),
                ],
                Duration::from_millis(150),
            );
            assert_eq!(results.len(), 4);
            assert_eq!(results[0].state, BatchState::Exited(0));
            assert!(results[0].run_id.is_some());

            assert_eq!(results[1].state, BatchState::TimedOut);
            assert!(results[1].run_id.is_some(), "the timing-out command must still have run");

            for skipped in &results[2..] {
                assert_eq!(skipped.state, BatchState::Skipped);
                assert!(skipped.run_id.is_none(), "a skipped command must never have been spawned");
                assert!(skipped.bounded_tail.is_none());
            }
        });
    }

    #[test]
    fn concurrency_n_gives_each_command_its_own_timeout_independent_of_siblings() {
        with_tools_root(|| {
            let start = Instant::now();
            let results = run_batch_concurrency_n(
                vec![
                    spec("true", &[]),
                    spec("sh", &["-c", "sleep 5"]),
                    spec("true", &[]),
                ],
                3,
                Duration::from_millis(150),
            );
            let elapsed = start.elapsed();

            assert_eq!(results.len(), 3);
            assert_eq!(results[0].state, BatchState::Exited(0));
            assert_eq!(results[1].state, BatchState::TimedOut);
            assert_eq!(
                results[2].state,
                BatchState::Exited(0),
                "a sibling command must complete normally, unaffected by another command's timeout"
            );
            assert!(
                elapsed < Duration::from_secs(2),
                "concurrent commands must not serialize to the sum of their durations, took {elapsed:?}"
            );
        });
    }

    #[test]
    fn concurrency_n_preserves_input_order_regardless_of_completion_order() {
        with_tools_root(|| {
            // Command 0 is deliberately the SLOWEST, so 1 and 2 finish
            // first -- the assembled output must still read 0, 1, 2.
            let results = run_batch_concurrency_n(
                vec![
                    spec("sh", &["-c", "sleep 0.2; echo first"]),
                    spec("sh", &["-c", "echo second"]),
                    spec("sh", &["-c", "echo third"]),
                ],
                3,
                Duration::from_secs(5),
            );
            assert_eq!(results.len(), 3);
            let stdout_of = |i: usize| {
                String::from_utf8_lossy(&results[i].bounded_tail.as_ref().unwrap().stdout).to_string()
            };
            assert!(stdout_of(0).contains("first"));
            assert!(stdout_of(1).contains("second"));
            assert!(stdout_of(2).contains("third"));
        });
    }

    #[test]
    fn dispatch_tool_batch_run_rejects_an_empty_command_list() {
        let err = dispatch_tool_batch_run(&serde_json::json!({"commands": []})).unwrap_err();
        assert_eq!(err.0, ERR_INVALID_PARAMS);
    }

    #[test]
    fn dispatch_tool_batch_run_over_json_returns_an_array_aligned_to_input_order() {
        with_tools_root(|| {
            let result = dispatch_tool_batch_run(&serde_json::json!({
                "commands": [
                    {"command": "sh", "args": ["-c", "echo one"]},
                    {"command": "sh", "args": ["-c", "echo two"]},
                ],
                "concurrency": 2,
                "timeout_ms": 5000,
            }))
            .unwrap();
            let arr = result.as_array().unwrap();
            assert_eq!(arr.len(), 2);
            assert!(arr[0]["bounded_tail"]["stdout"].as_str().unwrap().contains("one"));
            assert!(arr[1]["bounded_tail"]["stdout"].as_str().unwrap().contains("two"));
        });
    }
}
