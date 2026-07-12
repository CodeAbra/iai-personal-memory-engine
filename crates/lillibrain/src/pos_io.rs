//! Cross-platform positional file I/O.
//!
//! The WAL, journal, and pager code writes and reads fixed-size pages at known
//! offsets. On Unix that's spelled `FileExt::{read_exact_at, write_all_at}`.
//! Windows has `seek_read` / `seek_write`, but they can short-read/short-write
//! and they don't provide read_exact / write_all semantics on their own.
//!
//! `PosIo` gives every platform the read_exact_at / write_all_at surface the
//! rest of the crate expects. The Unix impl is a straight delegation. The
//! Windows impl loops around `seek_read` / `seek_write`, matching the Unix
//! trait's short-transfer behaviour so callers can treat both platforms the
//! same.

use std::fs::File;
use std::io::{Error, ErrorKind, Result};

pub trait PosIo {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()>;
    fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()>;
}

#[cfg(unix)]
impl PosIo for File {
    #[inline]
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, offset)
    }

    #[inline]
    fn write_all_at(&self, buf: &[u8], offset: u64) -> Result<()> {
        std::os::unix::fs::FileExt::write_all_at(self, buf, offset)
    }
}

#[cfg(windows)]
impl PosIo for File {
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> Result<()> {
        while !buf.is_empty() {
            match std::os::windows::fs::FileExt::seek_read(self, buf, offset) {
                Ok(0) => break,
                Ok(n) => {
                    let tmp = buf;
                    buf = &mut tmp[n..];
                    offset += n as u64;
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        if buf.is_empty() {
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::UnexpectedEof,
                "failed to fill whole buffer",
            ))
        }
    }

    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> Result<()> {
        while !buf.is_empty() {
            match std::os::windows::fs::FileExt::seek_write(self, buf, offset) {
                Ok(0) => {
                    return Err(Error::new(
                        ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    ));
                }
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::PosIo;
    use std::io::Write;

    #[test]
    fn round_trip_write_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("posio.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&[0u8; 32]).unwrap();
        drop(f);

        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        f.write_all_at(&[1, 2, 3, 4], 8).unwrap();
        let mut buf = [0u8; 4];
        f.read_exact_at(&mut buf, 8).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn read_past_eof_is_unexpected_eof() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("posio_eof.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&[0u8; 4]).unwrap();
        drop(f);

        let f = std::fs::File::open(&path).unwrap();
        let mut buf = [0u8; 8];
        let err = f.read_exact_at(&mut buf, 0).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }
}
