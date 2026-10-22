//! Byte-for-byte exact-substring scan over the whole-file manifest's
//! currently-indexed file set. Unlike the tokenizing lanes (`tokenize.rs`
//! drops short runs and hashes surviving tokens), this scan re-reads raw
//! bytes and compares them literally -- no case-folding, no decode, no
//! parser -- so it is the one lane that can PROVE a literal's absence
//! rather than report an honest "unknown".

use std::fs;
use std::io::Read as _;
use std::path::Path;

use crate::index::{self, IndexError};
use crate::snippet::MAX_SNIPPET_READ_BYTES;

/// Result of one literal scan: `hits` is (absolute path, 1-based
/// first-match line), one entry per file containing the literal --
/// presence only, no per-file occurrence count (an early-exit scan on
/// first match cannot cheaply count further occurrences). Every manifest
/// entry lands in exactly one of `files_scanned` / `files_truncated` /
/// `files_unreadable`, so the three sum to `n_files`.
pub struct LiteralScanResult {
    pub hits: Vec<(String, u32)>,
    pub files_scanned: u64,
    pub files_unreadable: u64,
    pub files_truncated: u64,
    pub n_files: u64,
    pub generation: u64,
}

/// 1-based line number of the first `\n`-delimited line containing byte
/// offset `match_offset` in `buf` -- counts `\n` bytes up to the offset,
/// never delegates to the fuzzy tokenizer-anchored `select_snippet` (which
/// can pick the wrong line for a literal the tokenizer would drop).
fn line_of_offset(buf: &[u8], match_offset: usize) -> u32 {
    1 + buf[..match_offset].iter().filter(|&&b| b == b'\n').count() as u32
}

/// First byte offset of `needle` in `haystack`, or `None`. Hand-rolled
/// (no `regex`/`memchr` dependency): scans for the needle's first byte,
/// then compares the full tail -- bounded, no backtracking beyond a
/// single candidate-offset retry per mismatch.
fn find_literal(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    let first = needle[0];
    let last_start = haystack.len() - needle.len();
    let mut i = 0usize;
    while i <= last_start {
        if haystack[i] == first && &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Scans every file in the whole-file manifest under `store_root` for the
/// exact byte sequence `literal`. Reads `file_index` (paths) and
/// `generation` off ONE `index::load_manifest` bind -- never a torn N/G
/// pair. An empty `literal` yields no hits (there is nothing to prove
/// present or absent). Each file's read is capped at
/// `MAX_SNIPPET_READ_BYTES`; a read that hits the cap with no in-prefix
/// match is excluded from `files_scanned` (unproven absence) and counted
/// in `files_truncated` instead -- a truncated read that DOES match is
/// still a valid hit. A file that fails to open/read is excluded from
/// `files_scanned` and counted in `files_unreadable`, mirroring
/// `Segment::build`'s skip-and-continue discipline (never `?`-propagates
/// one bad file out of the whole scan).
pub fn scan(store_root: &Path, literal: &str) -> Result<LiteralScanResult, IndexError> {
    let set = index::load_manifest(store_root)?;
    let n_files = set.file_index.len() as u64;
    let generation = set.generation;

    let needle = literal.as_bytes();
    let mut hits: Vec<(String, u32)> = Vec::new();
    let mut files_scanned: u64 = 0;
    let mut files_unreadable: u64 = 0;
    let mut files_truncated: u64 = 0;

    if needle.is_empty() {
        return Ok(LiteralScanResult {
            hits,
            files_scanned,
            files_unreadable,
            files_truncated,
            n_files,
            generation,
        });
    }

    for path in set.file_index.keys() {
        let mut file = match fs::File::open(path) {
            Ok(f) => f,
            Err(_) => {
                files_unreadable += 1;
                continue;
            }
        };
        let mut buf = Vec::new();
        let bytes_read = match file.by_ref().take(MAX_SNIPPET_READ_BYTES).read_to_end(&mut buf) {
            Ok(n) => n,
            Err(_) => {
                files_unreadable += 1;
                continue;
            }
        };
        let truncated = bytes_read as u64 >= MAX_SNIPPET_READ_BYTES;

        match find_literal(&buf, needle) {
            Some(offset) => {
                hits.push((path.clone(), line_of_offset(&buf, offset)));
                files_scanned += 1;
            }
            None => {
                if truncated {
                    files_truncated += 1;
                } else {
                    files_scanned += 1;
                }
            }
        }
    }

    Ok(LiteralScanResult {
        hits,
        files_scanned,
        files_unreadable,
        files_truncated,
        n_files,
        generation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::append_batch;
    use std::fs as stdfs;
    use std::path::PathBuf;

    fn write_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        stdfs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn present_literal_is_found_with_first_match_line() {
        let dir = std::env::temp_dir().join(format!("exactscan-present-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", b"line one\nline two zebracornliteral here\nline three\n");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let result = scan(&store_root, "zebracornliteral").unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].0, a.to_string_lossy());
        assert_eq!(result.hits[0].1, 2, "the literal is on the second line");
        assert_eq!(result.files_scanned, 1);
        assert_eq!(result.files_unreadable, 0);
        assert_eq!(result.files_truncated, 0);
        assert_eq!(result.n_files, 1);

        stdfs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn same_literal_in_two_files_returns_both_hits() {
        // Proves the loop early-exits per FILE (not the whole scan after
        // the first matching file): both files must be found as hits.
        let dir = std::env::temp_dir().join(format!("exactscan-twofiles-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", b"sharedliteralmarker in alpha\n");
        let b = write_file(&dir, "beta.txt", b"sharedliteralmarker in beta\n");
        append_batch(&store_root, &[a.clone(), b.clone()]).unwrap();

        let result = scan(&store_root, "sharedliteralmarker").unwrap();
        let hit_paths: Vec<&str> = result.hits.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(result.hits.len(), 2, "both files must be reported as hits");
        assert!(hit_paths.contains(&a.to_string_lossy().as_ref()));
        assert!(hit_paths.contains(&b.to_string_lossy().as_ref()));
        assert_eq!(result.files_scanned, 2);

        stdfs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn case_variant_literal_is_not_found() {
        let dir = std::env::temp_dir().join(format!("exactscan-casevariant-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", b"CaseSensitiveMarker present here\n");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let result = scan(&store_root, "casesensitivemarker").unwrap();
        assert!(result.hits.is_empty(), "raw-byte compare must not case-fold");
        assert_eq!(result.files_scanned, 1);

        stdfs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_on_disk_is_excluded_from_scanned_and_counted_unreadable() {
        let dir = std::env::temp_dir().join(format!("exactscan-missing-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", b"deletablemarker content\n");
        let b = write_file(&dir, "beta.txt", b"unrelated content\n");
        append_batch(&store_root, &[a.clone(), b.clone()]).unwrap();
        stdfs::remove_file(&a).unwrap();

        let result = scan(&store_root, "deletablemarker").unwrap();
        assert!(result.hits.is_empty());
        assert_eq!(result.n_files, 2);
        assert_eq!(result.files_scanned, 1);
        assert_eq!(result.files_unreadable, 1);
        assert_eq!(result.files_truncated, 0);

        stdfs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn truncated_tail_file_is_excluded_from_scanned_and_counted_truncated() {
        let dir = std::env::temp_dir().join(format!("exactscan-truncated-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        // Index a SMALL file first (under the admission cap), THEN grow it
        // on disk past the read cap with the literal in the unread tail --
        // reversed order would keep the file out of file_index entirely.
        let a = write_file(&dir, "alpha.txt", b"small content, indexed first\n");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let filler = vec![b'x'; 8192];
        let mut file = stdfs::OpenOptions::new().append(true).open(&a).unwrap();
        let mut written: u64 = 0;
        while written < MAX_SNIPPET_READ_BYTES {
            use std::io::Write as _;
            file.write_all(&filler).unwrap();
            written += filler.len() as u64;
        }
        use std::io::Write as _;
        file.write_all(b"unreadtailliteral\n").unwrap();
        drop(file);

        let result = scan(&store_root, "unreadtailliteral").unwrap();
        assert!(result.hits.is_empty(), "the literal lives past the read cap");
        assert_eq!(result.n_files, 1);
        assert_eq!(result.files_scanned, 0);
        assert_eq!(result.files_truncated, 1);
        assert_eq!(result.files_unreadable, 0);

        stdfs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn truncated_read_with_prefix_match_is_still_a_valid_hit() {
        let dir = std::env::temp_dir().join(format!("exactscan-truncmatch-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", b"prefixliteralmarker at the start\n");
        append_batch(&store_root, &[a.clone()]).unwrap();

        let filler = vec![b'y'; 8192];
        let mut file = stdfs::OpenOptions::new().append(true).open(&a).unwrap();
        let mut written: u64 = 0;
        while written < MAX_SNIPPET_READ_BYTES {
            use std::io::Write as _;
            file.write_all(&filler).unwrap();
            written += filler.len() as u64;
        }
        drop(file);

        let result = scan(&store_root, "prefixliteralmarker").unwrap();
        assert_eq!(result.hits.len(), 1, "a match within the read prefix is valid even though the read was truncated");
        assert_eq!(result.files_scanned, 1);
        assert_eq!(result.files_truncated, 0);

        stdfs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bucket_counts_sum_to_n_files() {
        let dir = std::env::temp_dir().join(format!("exactscan-buckets-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", b"aaa content\n");
        let b = write_file(&dir, "beta.txt", b"bbb content\n");
        let c = write_file(&dir, "gamma.txt", b"ccc content\n");
        append_batch(&store_root, &[a.clone(), b.clone(), c.clone()]).unwrap();
        stdfs::remove_file(&b).unwrap();

        let result = scan(&store_root, "never_indexed_anywhere_zzz").unwrap();
        assert_eq!(
            result.files_scanned + result.files_truncated + result.files_unreadable,
            result.n_files
        );

        stdfs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_literal_yields_no_hits_and_no_error() {
        let dir = std::env::temp_dir().join(format!("exactscan-empty-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", b"anything at all\n");
        append_batch(&store_root, &[a]).unwrap();

        let result = scan(&store_root, "").unwrap();
        assert!(result.hits.is_empty());

        stdfs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn absent_literal_over_the_whole_corpus_reports_zero_hits_with_scope() {
        let dir = std::env::temp_dir().join(format!("exactscan-absent-{}", std::process::id()));
        stdfs::create_dir_all(&dir).unwrap();
        let store_root = dir.join("store");

        let a = write_file(&dir, "alpha.txt", b"present content only\n");
        append_batch(&store_root, &[a]).unwrap();

        let result = scan(&store_root, "genuinely_absent_literal_zzz").unwrap();
        assert!(result.hits.is_empty());
        assert_eq!(result.files_scanned, 1);
        assert_eq!(result.n_files, 1);
        assert!(result.generation >= 1);

        stdfs::remove_dir_all(&dir).ok();
    }
}
