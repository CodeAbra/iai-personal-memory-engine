//! setrlimit caps applied to the child in an async-signal-safe `pre_exec`
//! closure: every value is computed in the PARENT (getrlimit + clamp)
//! before the closure runs, so the closure itself only calls `setrlimit` --
//! no allocation, no lock, safe between fork and exec.
//!
//! Pitfall: a `rlim_max` requested above the inherited hard limit fails
//! `setrlimit` with EPERM, and a failing `pre_exec` closure fails the whole
//! spawn -- every cap here is clamped DOWN to the inherited hard limit,
//! never raised past it. RLIMIT_AS is intentionally not set: unenforced on
//! macOS, and left unset on Linux too so the wall-clock timeout stays the
//! one cross-platform backstop.

#[cfg(unix)]
use std::io;

/// `getrlimit`/`setrlimit`'s resource-id type is `c_int` on macOS/BSD and on
/// musl, but `__rlimit_resource_t` (u32) on glibc -- RLIMIT_* constants match
/// whichever the target defines. Both arms are POSIX-only.
#[cfg(all(unix, target_os = "linux", target_env = "gnu"))]
type RlimitResource = libc::__rlimit_resource_t;
#[cfg(all(unix, not(all(target_os = "linux", target_env = "gnu"))))]
type RlimitResource = libc::c_int;

/// Coarse CPU-seconds backstop; the wall-clock timeout is the real lever,
/// so this is generous enough to never kill a legitimate multi-minute
/// build or test run.
pub const CPU_SECS_CAP: u64 = 3600;
/// Open-fd ceiling, well above a parallel build's needs; bounds fd
/// exhaustion.
pub const FD_CAP: u64 = 4096;
/// Bytes. Bounds any single file the child writes, including its own
/// stdout.log; exceeding it delivers SIGXFSZ to the child. Well above any
/// legitimate log.
pub const FSIZE_CAP: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct EffectiveLimits {
    pub cpu_secs: u64,
    pub fds: u64,
    pub fsize: u64,
}

/// Pure clamp arithmetic, factored out of the `getrlimit` syscall so it is
/// unit-testable without mutating process-wide rlimit state.
#[cfg(unix)]
fn clamp(cap: u64, hard: libc::rlim_t) -> u64 {
    if hard == libc::RLIM_INFINITY {
        return cap;
    }
    cap.min(hard as u64)
}

/// `getrlimit`s the CURRENT process's hard limit for `resource` and
/// returns `min(cap, hard)`.
#[cfg(unix)]
fn clamp_to_hard(resource: RlimitResource, cap: u64) -> u64 {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `resource` is a valid RLIMIT_* constant; `limit` is a
    // stack-local out-parameter sized for the call.
    let rc = unsafe { libc::getrlimit(resource, &mut limit) };
    if rc != 0 {
        return cap;
    }
    clamp(cap, limit.rlim_max)
}

/// Computes the effective caps in the PARENT, before fork -- the pre_exec
/// closure receives only these already-clamped values.
pub fn effective_limits() -> EffectiveLimits {
    effective_limits_with_caps(CPU_SECS_CAP, FD_CAP, FSIZE_CAP)
}

#[cfg(unix)]
fn effective_limits_with_caps(cpu_cap: u64, fd_cap: u64, fsize_cap: u64) -> EffectiveLimits {
    EffectiveLimits {
        cpu_secs: clamp_to_hard(libc::RLIMIT_CPU, cpu_cap),
        fds: clamp_to_hard(libc::RLIMIT_NOFILE, fd_cap),
        fsize: clamp_to_hard(libc::RLIMIT_FSIZE, fsize_cap),
    }
}

/// No `setrlimit` here; caps pass through unclamped and unenforced.
#[cfg(not(unix))]
fn effective_limits_with_caps(cpu_cap: u64, fd_cap: u64, fsize_cap: u64) -> EffectiveLimits {
    EffectiveLimits {
        cpu_secs: cpu_cap,
        fds: fd_cap,
        fsize: fsize_cap,
    }
}

/// Test-only override for `FSIZE_CAP` -- lets a test trip the guard with a
/// tiny cap instead of a multi-GiB write.
#[cfg(test)]
pub fn effective_limits_for_test(fsize_cap: u64) -> EffectiveLimits {
    effective_limits_with_caps(CPU_SECS_CAP, FD_CAP, fsize_cap)
}

/// Applies `limits` to the CURRENT process via `setrlimit`.
///
/// # Safety
/// Must only be called from a `pre_exec` closure running in the forked
/// child before `exec`, per `CommandExt::pre_exec`'s own safety contract:
/// async-signal-safe only. Every value here is already computed by the
/// caller -- this function itself allocates nothing and locks nothing.
#[cfg(unix)]
pub unsafe fn apply(limits: &EffectiveLimits) -> io::Result<()> {
    // SAFETY: forwarding the caller's contract; each call below uses only
    // a stack-local `rlimit` built from already-plain integers.
    unsafe {
        set_one(libc::RLIMIT_CPU, limits.cpu_secs)?;
        set_one(libc::RLIMIT_NOFILE, limits.fds)?;
        set_one(libc::RLIMIT_FSIZE, limits.fsize)?;
    }
    Ok(())
}

#[cfg(unix)]
unsafe fn set_one(resource: RlimitResource, value: u64) -> io::Result<()> {
    let limit = libc::rlimit {
        rlim_cur: value as libc::rlim_t,
        rlim_max: value as libc::rlim_t,
    };
    // SAFETY: `resource` is a valid RLIMIT_* constant and `limit` is a
    // valid stack-local `rlimit` struct for the call.
    let rc = unsafe { libc::setrlimit(resource, &limit) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::spawn;
    use std::fs::{File, OpenOptions};
    use std::os::unix::process::ExitStatusExt;
    use std::path::Path;

    fn open_tmp_file(dir: &Path, name: &str) -> (File, std::path::PathBuf) {
        let path = dir.join(name);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        (file, path)
    }

    #[test]
    fn clamp_never_exceeds_a_lowered_hard_limit() {
        assert_eq!(clamp(FD_CAP, 256), 256);
        assert_eq!(clamp(100, 4096), 100);
        assert_eq!(clamp(FD_CAP, libc::RLIM_INFINITY), FD_CAP);
    }

    /// `ulimit -f`'s block unit is shell/platform dependent -- discovered
    /// empirically below from a known cap rather than hardcoded, since it
    /// varies across `sh` implementations.
    fn ulimit_f_blocks(dir: &Path, limits: &EffectiveLimits, tag: &str) -> u64 {
        let (stdout_file, stdout_path) =
            open_tmp_file(dir, &format!("ulimit-f-{tag}-stdout.log"));
        let (stderr_file, _p) = open_tmp_file(dir, &format!("ulimit-f-{tag}-stderr.log"));
        let mut spawned =
            spawn::spawn_bounded("sh", &["-c".to_string(), "ulimit -f".to_string()], stdout_file, stderr_file, limits)
                .unwrap();
        spawned.child.wait().unwrap();
        std::fs::read_to_string(&stdout_path)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    #[test]
    fn ulimit_probe_reports_clamped_effective_caps() {
        let tmp = tempfile::tempdir().unwrap();

        // 1 MiB is an exact multiple of every plausible ulimit block unit
        // (512 or 1024 bytes); reporting it back tells us this host's
        // conversion factor without assuming either one.
        let probe_cap = 1024 * 1024;
        let probe_limits = effective_limits_for_test(probe_cap);
        let probe_blocks = ulimit_f_blocks(tmp.path(), &probe_limits, "probe");
        let block_size = probe_cap / probe_blocks;

        let (stdout_file, stdout_path) = open_tmp_file(tmp.path(), "stdout.log");
        let (stderr_file, _p) = open_tmp_file(tmp.path(), "stderr.log");
        let limits = effective_limits();

        let mut spawned = spawn::spawn_bounded(
            "sh",
            &[
                "-c".to_string(),
                "ulimit -t; ulimit -n; ulimit -f".to_string(),
            ],
            stdout_file,
            stderr_file,
            &limits,
        )
        .unwrap();
        spawned.child.wait().unwrap();

        let out = std::fs::read_to_string(&stdout_path).unwrap();
        let mut lines = out.lines();
        let cpu: u64 = lines.next().unwrap().trim().parse().unwrap();
        let fds: u64 = lines.next().unwrap().trim().parse().unwrap();
        let fsize_blocks: u64 = lines.next().unwrap().trim().parse().unwrap();

        assert_eq!(cpu, limits.cpu_secs);
        assert_eq!(fds, limits.fds);
        assert_eq!(fsize_blocks * block_size, limits.fsize);
    }

    #[test]
    fn fsize_cap_trips_sigxfsz_on_a_shell_free_runaway_writer() {
        let tmp = tempfile::tempdir().unwrap();
        let (stdout_file, _stdout_path) = open_tmp_file(tmp.path(), "stdout.log");
        let (stderr_file, _p) = open_tmp_file(tmp.path(), "stderr.log");
        // Tiny override: `yes` fills this in well under a millisecond of
        // real time, so the test stays fast and deterministic.
        let limits = effective_limits_for_test(4096);

        let mut spawned =
            spawn::spawn_bounded("yes", &[], stdout_file, stderr_file, &limits).unwrap();
        let status = spawned.child.wait().unwrap();

        assert_eq!(
            status.signal(),
            Some(libc::SIGXFSZ),
            "the direct child (no shell wrapper) must be terminated by SIGXFSZ"
        );
    }
}
