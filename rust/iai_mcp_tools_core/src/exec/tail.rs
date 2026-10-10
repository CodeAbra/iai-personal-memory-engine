//! Byte/line-capped tail read of an artifact file: returns the window from a
//! stored offset up to whichever cap hits first, with an elision count. This
//! is the RETURN-TO-CONTEXT bound (what re-enters a caller's context on a
//! fetch) -- distinct from an on-disk FSIZE cap (what lands on disk at all,
//! a separate future knob); the two bounds are independent.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::ExecError;

/// Bytes returned per fetch window.
pub const TAIL_BYTE_CAP: u64 = 64 * 1024;
/// Lines returned per fetch window (whichever cap is hit first wins).
pub const TAIL_LINE_CAP: usize = 2000;

#[derive(Debug, Clone)]
pub struct TailBound {
    pub byte_cap: u64,
    pub line_cap: usize,
}

impl TailBound {
    pub fn default_bound() -> Self {
        TailBound {
            byte_cap: TAIL_BYTE_CAP,
            line_cap: TAIL_LINE_CAP,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TailWindow {
    pub data: Vec<u8>,
    pub new_offset: u64,
    pub elided_bytes: u64,
}

/// Reads from `offset` up to the byte/line cap, re-`stat`ing the file on
/// every call -- correct for a still-growing file (a later plan's detached
/// run); liveness is never inferred from mtime here.
pub fn read_bounded(path: &Path, offset: u64, bound: &TailBound) -> Result<TailWindow, ExecError> {
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    if offset >= size {
        return Ok(TailWindow {
            data: Vec::new(),
            new_offset: offset,
            elided_bytes: 0,
        });
    }
    file.seek(SeekFrom::Start(offset))?;

    let available = size - offset;
    let byte_cap = bound.byte_cap.max(1);
    let read_len = available.min(byte_cap);
    let mut buf = vec![0u8; read_len as usize];
    file.read_exact(&mut buf)?;

    // Line cap applies on top of the byte-capped window: if hit before all
    // read bytes are consumed, elide the remainder of THIS window (the rest
    // stays on disk, untouched).
    let mut consumed = buf.len();
    if bound.line_cap > 0 {
        let mut lines_seen = 0usize;
        for (i, &b) in buf.iter().enumerate() {
            if b == b'\n' {
                lines_seen += 1;
                if lines_seen >= bound.line_cap {
                    consumed = i + 1;
                    break;
                }
            }
        }
    }

    let elided_in_window = (buf.len() - consumed) as u64;
    let elided_after_window = available - read_len;
    buf.truncate(consumed);
    let new_offset = offset + consumed as u64;

    Ok(TailWindow {
        data: buf,
        new_offset,
        elided_bytes: elided_in_window + elided_after_window,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn read_bounded_sees_appended_bytes_on_a_growing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("growing.log");
        std::fs::write(&path, b"first\n").unwrap();
        let bound = TailBound::default_bound();

        let first = read_bounded(&path, 0, &bound).unwrap();
        assert_eq!(first.data, b"first\n");

        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(b"second\n").unwrap();
        drop(f);

        let second = read_bounded(&path, first.new_offset, &bound).unwrap();
        assert_eq!(second.data, b"second\n");
    }

    #[test]
    fn read_bounded_elides_past_the_line_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("lines.log");
        let mut content = String::new();
        for i in 0..(TAIL_LINE_CAP + 500) {
            content.push_str(&format!("{i}\n"));
        }
        std::fs::write(&path, &content).unwrap();

        let bound = TailBound::default_bound();
        let window = read_bounded(&path, 0, &bound).unwrap();
        let lines = window.data.iter().filter(|&&b| b == b'\n').count();
        assert_eq!(lines, TAIL_LINE_CAP);
        assert!(window.elided_bytes > 0);
    }

    #[test]
    fn read_bounded_elides_past_the_byte_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bytes.log");
        let content = vec![b'x'; (TAIL_BYTE_CAP * 2) as usize];
        std::fs::write(&path, &content).unwrap();

        let bound = TailBound {
            byte_cap: TAIL_BYTE_CAP,
            line_cap: 0,
        };
        let window = read_bounded(&path, 0, &bound).unwrap();
        assert_eq!(window.data.len() as u64, TAIL_BYTE_CAP);
        assert!(window.elided_bytes > 0);
    }

    #[test]
    fn read_bounded_past_end_of_file_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("short.log");
        std::fs::write(&path, b"hi\n").unwrap();
        let window = read_bounded(&path, 3, &TailBound::default_bound()).unwrap();
        assert!(window.data.is_empty());
        assert_eq!(window.new_offset, 3);
        assert_eq!(window.elided_bytes, 0);
    }
}
