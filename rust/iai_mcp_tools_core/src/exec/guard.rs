//! Source guard: walks every `.rs` file under `src/exec/`, `src/bin/`, and
//! `src/daemon/` and fails if any of them (other than this file) contains a
//! non-ASCII byte or a dev-process metadata marker shaped like a
//! phase/plan/task marker or a decision/requirement code. This file is the
//! sanctioned exception -- a marker scanner necessarily carries the literal
//! shapes it looks for as data, not as a description of ITS OWN content.

use std::path::{Path, PathBuf};

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Detects an uppercase-letter code immediately followed by `-<digits>`
/// (the general shape of a decision/requirement code such as "EXEC-01" or
/// "DEC-6") -- a pattern, not a fixed list tied to any one phase.
fn find_hyphenated_code(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_uppercase() {
            let start = i;
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_uppercase() {
                j += 1;
            }
            let letters = j - start;
            if letters >= 3 && j < bytes.len() && bytes[j] == b'-' {
                let digit_start = j + 1;
                let mut k = digit_start;
                while k < bytes.len() && bytes[k].is_ascii_digit() {
                    k += 1;
                }
                if k > digit_start {
                    return Some(&text[start..k]);
                }
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    None
}

/// Detects a planning word (the caller passes it, e.g. the Phase/Plan/Task/Wave
/// family) immediately followed by an ASCII digit -- the numbered-marker shape
/// the anti-slop guard rejects.
fn find_numbered_word<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let bytes = text.as_bytes();
    let wb = word.as_bytes();
    if bytes.len() < wb.len() {
        return None;
    }
    let mut i = 0;
    while i + wb.len() < bytes.len() {
        if &bytes[i..i + wb.len()] == wb && bytes[i + wb.len()].is_ascii_digit() {
            let end = (i + wb.len() + 1).min(bytes.len());
            return Some(&text[i..end]);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_and_bin_sources_are_english_only_and_marker_free() {
        let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        collect_rs_files(&crate_root.join("src/exec"), &mut files).unwrap();
        collect_rs_files(&crate_root.join("src/bin"), &mut files).unwrap();
        collect_rs_files(&crate_root.join("src/daemon"), &mut files).unwrap();
        assert!(
            !files.is_empty(),
            "guard scanned zero files -- path resolution is broken"
        );

        for path in files {
            if path.file_name().and_then(|n| n.to_str()) == Some("guard.rs") {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            assert!(bytes.is_ascii(), "{} contains a non-ASCII byte", path.display());

            let text = String::from_utf8_lossy(&bytes);
            for word in ["Phase ", "Plan ", "Task ", "Wave "] {
                assert!(
                    find_numbered_word(&text, word).is_none(),
                    "{} contains a dev-process marker shaped like {word:?}",
                    path.display()
                );
            }
            assert!(
                find_hyphenated_code(&text).is_none(),
                "{} contains a decision/requirement-code-shaped marker",
                path.display()
            );
        }
    }
}
