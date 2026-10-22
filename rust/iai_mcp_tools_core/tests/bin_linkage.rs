//! Proves the standalone binary carries no libpython link -- the reason the
//! executor ships as a separate binary, not a pyo3 sub-module. Portable
//! across the platforms this project supports (macOS, Linux); Windows is
//! out of scope for this test (a separate lane owns Windows verification).

use std::process::Command;

fn has_libpython_marker(text: &str) -> bool {
    text.contains("libpython") || text.contains("Python.framework")
}

/// `otool -L` is the macOS-native way to list a Mach-O binary's dynamic
/// dependencies. Kept hard-asserting (unlike the Linux branch below)
/// because macOS is the platform this gate actually runs on today; a
/// missing `otool` here would mean a broken toolchain, not a portability
/// gap.
#[cfg(target_os = "macos")]
fn check_no_libpython_link(bin: &str) {
    let output = Command::new("otool")
        .arg("-L")
        .arg(bin)
        .output()
        .expect("failed to run otool");
    assert!(output.status.success(), "otool -L failed on {bin}");
    let listing = String::from_utf8_lossy(&output.stdout);
    assert!(!has_libpython_marker(&listing), "binary links libpython:\n{listing}");
}

/// `ldd`'s exit code is not a reliable success signal here -- it is
/// nonzero for a statically-linked or non-dynamic executable even though
/// that is a perfectly valid ("definitely no libpython link") outcome, so
/// this only inspects combined stdout+stderr text, never the exit code.
/// `readelf -d` is the fallback when `ldd` itself is not installed. If
/// neither tool is available, this prints a reason and returns without
/// failing -- a missing platform tool is not evidence of a real libpython
/// link, and a hard failure here would make this test non-portable to any
/// minimal Linux container that lacks binutils.
#[cfg(target_os = "linux")]
fn check_no_libpython_link(bin: &str) {
    fn combined_text(output: &std::process::Output) -> String {
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }

    if let Ok(output) = Command::new("ldd").arg(bin).output() {
        let listing = combined_text(&output);
        assert!(!has_libpython_marker(&listing), "binary links libpython (ldd):\n{listing}");
        return;
    }
    if let Ok(output) = Command::new("readelf").arg("-d").arg(bin).output() {
        let listing = combined_text(&output);
        assert!(
            !has_libpython_marker(&listing),
            "binary links libpython (readelf -d):\n{listing}"
        );
        return;
    }
    eprintln!(
        "bin_linkage: skipping -- neither `ldd` nor `readelf` is available on this Linux host"
    );
}

/// Windows is out of scope for this test (a separate lane owns Windows
/// verification); any other non-macOS, non-Linux target also has no known
/// linkage-inspection tool wired up here, so this is a deliberate skip,
/// not a silent gap.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn check_no_libpython_link(_bin: &str) {
    eprintln!("bin_linkage: skipping -- no linkage-inspection tool wired up for this platform");
}

#[test]
fn iai_tools_run_has_no_libpython_link() {
    let bin = env!("CARGO_BIN_EXE_iai_tools_run");
    check_no_libpython_link(bin);
}
