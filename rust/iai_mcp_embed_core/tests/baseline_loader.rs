//! Minimal NPY format reader for float32 little-endian C-order arrays.
//!
//! Test-only helper module. Used by `numeric_parity.rs` to load the frozen
//! baseline `vectors.npy` without requiring a runtime `numpy` Python dep.
//!
//! Supports NPY v1.0 and v2.0 headers (header_len = u16 / u32 little-endian).
//! Asserts `dtype == '<f4'` and `fortran_order == False`.

use std::fs;
use std::io;
use std::path::Path;

/// Load a float32 little-endian, C-order .npy file into a flat `Vec<f32>`.
///
/// Returns `(data, shape)` where `data.len() == shape.iter().product()`.
/// Errors on header parse failure or dtype mismatch.
pub fn load_npy_f32<P: AsRef<Path>>(path: P) -> io::Result<(Vec<f32>, Vec<usize>)> {
    // nosemgrep
    let bytes = fs::read(path.as_ref())?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a numpy npy file (magic mismatch)",
        ));
    }
    let major = bytes[6];
    let _minor = bytes[7];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 => {
            if bytes.len() < 12 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "npy v2 header truncated",
                ));
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported npy version",
            ))
        }
    };

    if bytes.len() < header_start + header_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "npy header truncated",
        ));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // dtype must be little-endian float32 (numpy descr '<f4')
    if !(header.contains("'<f4'") || header.contains("'descr': '<f4'")) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("dtype not <f4: {header}"),
        ));
    }
    if header.contains("fortran_order': True") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fortran-order not supported",
        ));
    }

    // Parse 'shape': (N, M, ...)
    let shape_start = header
        .find("'shape':")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no shape key in header"))?;
    let paren_open = header[shape_start..]
        .find('(')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no shape open paren"))?;
    let paren_close = header[shape_start..]
        .find(')')
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no shape close paren"))?;
    let shape_str = &header[shape_start + paren_open + 1..shape_start + paren_close];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let n_elem: usize = shape.iter().product();
    let data_offset = header_start + header_len;
    let expected_bytes = n_elem * 4;
    if bytes.len() < data_offset + expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "data section truncated",
        ));
    }
    let mut data = Vec::with_capacity(n_elem);
    for chunk in bytes[data_offset..data_offset + expected_bytes].chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((data, shape))
}
