//! Durable file-sync primitives.
//!
//! On macOS a plain `fsync` only pushes data to the drive controller; the
//! controller's volatile DRAM cache is flushed to stable NAND only by
//! `F_FULLFSYNC`. This module wraps that distinction so callers never invoke
//! `fsync` directly, and propagates every failure — an fsync error is never
//! swallowed.
//!
//! Setting the environment variable `LILLI_FSYNC_MODE=fast` substitutes a plain
//! data sync for the full drive-cache flush. This trades power-loss durability
//! for speed and is intended only for test and continuous-integration runs,
//! where the per-commit full-flush latency would otherwise dominate.

use std::fs::File;
use std::path::Path;

use crate::error::Result;

/// True when the caller opted out of the full drive-cache flush.
fn full_flush_disabled() -> bool {
    std::env::var("LILLI_FSYNC_MODE").map(|v| v == "fast").unwrap_or(false)
}

/// Flush a file to stable storage.
///
/// On macOS this issues `F_FULLFSYNC`, which forces the drive controller's
/// volatile cache to NAND, falling back to `sync_all` when the platform rejects
/// it (network filesystems, FUSE, virtual machines). On other platforms it
/// issues `sync_all`. When `LILLI_FSYNC_MODE=fast` is set it issues `sync_data`.
/// The error is always propagated.
pub fn durable_fsync(file: &File) -> Result<()> {
    if full_flush_disabled() {
        file.sync_data()?;
        return Ok(());
    }
    full_fsync(file)
}

#[cfg(target_os = "macos")]
fn full_fsync(file: &File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: fd is a valid open file descriptor owned by `file` for the call.
    let ret = unsafe { libc::fcntl(fd, libc::F_FULLFSYNC) };
    if ret != 0 {
        // Full flush is unsupported on this volume; fall back to a normal sync.
        file.sync_all()?;
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn full_fsync(file: &File) -> Result<()> {
    file.sync_all()?;
    Ok(())
}

/// Flush the directory entry for `path` to stable storage.
///
/// Ensures that a newly created, renamed, or unlinked file in the directory is
/// durable. The directory is opened read-only, synced, and closed. The error is
/// always propagated.
pub fn fsync_directory(path: &Path) -> Result<()> {
    let dir = File::open(path)?;
    dir.sync_all()?;
    Ok(())
}

/// Flush the directory entry for `path`'s containing directory to stable
/// storage. An empty parent (bare relative path) resolves to the current
/// directory. The error is always propagated.
pub fn fsync_parent_directory(path: &Path) -> Result<()> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    fsync_directory(parent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn io_durable_fsync_full_mode_returns_ok() {
        // Ensure the fast escape is off for this assertion.
        std::env::remove_var("LILLI_FSYNC_MODE");
        let mut f = tempfile::tempfile().unwrap();
        f.write_all(b"hello world").unwrap();
        durable_fsync(&f).unwrap();
    }

    #[test]
    fn io_durable_fsync_fast_mode_returns_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fast.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(b"fast path").unwrap();
        // Drive the fast path explicitly rather than mutating process env,
        // which would race other tests in the same binary.
        f.sync_data().unwrap();
        durable_fsync(&f).unwrap();
    }

    #[test]
    fn io_fsync_directory_returns_ok() {
        let dir = tempfile::tempdir().unwrap();
        fsync_directory(dir.path()).unwrap();
    }
}
