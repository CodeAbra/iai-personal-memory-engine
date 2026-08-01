//! Variable-length integer codec and serial-type record encoder/decoder.
//!
//! Implements the variable-length integer (varint) format used by the B-tree
//! leaf cell payloads, plus a serial-type record format: a varint header size,
//! one serial-type code per column, then tightly packed column bodies.
//!
//! Every function here is pure: no I/O, no shared state. The B-tree's cell
//! framing and the row serialization above the storage seam both build on it.
//!
//! Blob columns round-trip byte-for-byte. A stored blob is an opaque slice of
//! bytes — it is never decoded, trimmed, normalized, or re-wrapped on the way
//! out; the exact input bytes are returned, including embedded NUL bytes,
//! multi-byte UTF-8 sequences, and trailing whitespace.

use crate::error::{Result, StoreError};

/// A decoded record column value.
///
/// `Blob` is the opaque-bytes carrier: its contents are returned exactly as
/// stored, with no interpretation. `Text` decodes a UTF-8 column to a string.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A NULL column (zero body bytes).
    Null,
    /// A signed 64-bit integer column.
    Int(i64),
    /// An IEEE-754 double-precision column.
    Float(f64),
    /// A UTF-8 text column.
    Text(String),
    /// An opaque byte column, returned verbatim.
    Blob(Vec<u8>),
}

// ---------------------------------------------------------------------------
// Varint encode / decode
// ---------------------------------------------------------------------------

/// Encode an integer as a 1-to-9-byte varint.
///
/// Negative values are masked to 64-bit unsigned first. Bytes 1 through 8 each
/// carry seven payload bits, most-significant first, with the high bit set as a
/// continuation flag. The ninth byte (only present for the largest values)
/// carries a full eight payload bits with no continuation flag.
pub fn encode_varint(value: i64) -> Vec<u8> {
    let mut v = value as u64;

    let n = if v <= 0x7F {
        1
    } else if v <= 0x3FFF {
        2
    } else if v <= 0x1F_FFFF {
        3
    } else if v <= 0xFFF_FFFF {
        4
    } else if v <= 0x7_FFFF_FFFF {
        5
    } else if v <= 0x3FF_FFFF_FFFF {
        6
    } else if v <= 0x1_FFFF_FFFF_FFFF {
        7
    } else if v <= 0xFF_FFFF_FFFF_FFFF {
        8
    } else {
        9
    };

    let mut buf = vec![0u8; n];
    if n < 9 {
        // N bytes: seven-bit groups from the low end up, stored MSB-first with
        // continuation bits on every byte but the last.
        for i in (0..n).rev() {
            buf[i] = ((v & 0x7F) as u8) | if i < n - 1 { 0x80 } else { 0 };
            v >>= 7;
        }
    } else {
        // Nine-byte case: the last byte takes a full eight bits; the first eight
        // each take seven bits with the continuation flag set.
        buf[8] = (v & 0xFF) as u8;
        v >>= 8;
        for i in (0..8).rev() {
            buf[i] = ((v & 0x7F) as u8) | 0x80;
            v >>= 7;
        }
    }
    buf
}

/// Decode a varint starting at `data[offset]`.
///
/// Returns `(value, bytes_consumed)`. Fails when the buffer ends before the
/// continuation-bit run terminates.
pub fn decode_varint(data: &[u8], offset: usize) -> Result<(i64, usize)> {
    let mut value: u64 = 0;
    for i in 0..9 {
        let byte = *data.get(offset + i).ok_or_else(|| StoreError::Integrity {
            detail: "varint: buffer too short".to_string(),
        })?;
        if i < 8 {
            value = (value << 7) | u64::from(byte & 0x7F);
            if byte & 0x80 == 0 {
                return Ok((value as i64, i + 1));
            }
        } else {
            value = (value << 8) | u64::from(byte);
            return Ok((value as i64, 9));
        }
    }
    Err(StoreError::Integrity {
        detail: "varint: malformed".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Serial-type record encode / decode
// ---------------------------------------------------------------------------
//
// Record layout:
//   varint(header_size)            — total bytes in the header, includes itself
//   varint(serial_type_1) ...      — one serial-type code per column
//   <body_1> <body_2> ...          — column bodies, tightly packed
//
// Serial-type codes:
//   0          NULL                (zero body bytes)
//   1          signed 8-bit integer
//   2          signed 16-bit integer
//   4          signed 32-bit integer
//   6          signed 64-bit integer
//   7          IEEE-754 double
//   8          integer 0           (zero body bytes)
//   9          integer 1           (zero body bytes)
//   N>=12 even TEXT of (N-12)/2 bytes
//   N>=13 odd  BLOB of (N-13)/2 bytes

/// Return the serial-type code for a value.
fn serial_type_for(value: &Value) -> i64 {
    match value {
        Value::Null => 0,
        Value::Blob(b) => 13 + 2 * b.len() as i64,
        Value::Float(_) => 7,
        Value::Text(s) => 12 + 2 * s.len() as i64,
        Value::Int(v) => {
            if *v == 0 {
                8
            } else if *v == 1 {
                9
            } else if (-(1i64 << 7)..(1i64 << 7)).contains(v) {
                1
            } else if (-(1i64 << 15)..(1i64 << 15)).contains(v) {
                2
            } else if (-(1i64 << 31)..(1i64 << 31)).contains(v) {
                4
            } else {
                6
            }
        }
    }
}

/// Return the body bytes for a value given its serial type.
fn body_for(value: &Value, serial_type: i64) -> Vec<u8> {
    match value {
        Value::Null => Vec::new(),
        Value::Int(v) => match serial_type {
            8 | 9 => Vec::new(),
            1 => (*v as i8).to_be_bytes().to_vec(),
            2 => (*v as i16).to_be_bytes().to_vec(),
            4 => (*v as i32).to_be_bytes().to_vec(),
            _ => v.to_be_bytes().to_vec(),
        },
        Value::Float(f) => f.to_be_bytes().to_vec(),
        Value::Text(s) => s.as_bytes().to_vec(),
        Value::Blob(b) => b.clone(),
    }
}

/// Return the number of body bytes a serial-type code consumes.
fn body_size(serial_type: i64) -> Result<usize> {
    let sz = match serial_type {
        0 | 8 | 9 => 0,
        1 => 1,
        2 => 2,
        4 => 4,
        6 | 7 => 8,
        st if st >= 13 && st % 2 == 1 => ((st - 13) / 2) as usize,
        st if st >= 12 && st % 2 == 0 => ((st - 12) / 2) as usize,
        st => {
            return Err(StoreError::Integrity {
                detail: format!("record: unknown serial type {st}"),
            })
        }
    };
    Ok(sz)
}

/// Encode a column list as a serial-type record.
pub fn encode_record(values: &[Value]) -> Vec<u8> {
    let serial_types: Vec<i64> = values.iter().map(serial_type_for).collect();

    let mut type_bytes = Vec::new();
    for st in &serial_types {
        type_bytes.extend_from_slice(&encode_varint(*st));
    }

    // The header_size field counts itself; iterate to a fixed point because the
    // varint width of header_size depends on header_size.
    let mut header_size_varint = encode_varint((encode_varint(0).len() + type_bytes.len()) as i64);
    for _ in 0..3 {
        let candidate = (header_size_varint.len() + type_bytes.len()) as i64;
        let next = encode_varint(candidate);
        if next.len() + type_bytes.len() == candidate as usize {
            header_size_varint = next;
            break;
        }
        header_size_varint = next;
    }

    let mut out = Vec::with_capacity(header_size_varint.len() + type_bytes.len());
    out.extend_from_slice(&header_size_varint);
    out.extend_from_slice(&type_bytes);
    for (v, st) in values.iter().zip(&serial_types) {
        out.extend_from_slice(&body_for(v, *st));
    }
    out
}

/// Decode a serial-type record starting at `data[offset]`.
///
/// Returns `(values, bytes_consumed)`. Blob columns are returned as exact byte
/// copies of their on-record slice — no normalization. Fails on a malformed
/// header or a body that runs past the buffer.
pub fn decode_record(data: &[u8], offset: usize) -> Result<(Vec<Value>, usize)> {
    let start = offset;
    let (header_size, n) = decode_varint(data, offset)?;
    let mut pos = offset.checked_add(n).ok_or_else(|| StoreError::Integrity {
        detail: "record: header offset overflow".to_string(),
    })?;
    let header_end = checked_header_end(start, header_size, data.len())?;

    let mut serial_types: Vec<i64> = Vec::new();
    while pos < header_end {
        let (st, m) = decode_varint(data, pos)?;
        pos = pos.checked_add(m).ok_or_else(|| StoreError::Integrity {
            detail: "record: header offset overflow".to_string(),
        })?;
        serial_types.push(st);
    }

    let mut body_offset = header_end;
    let mut values: Vec<Value> = Vec::with_capacity(serial_types.len());
    for st in serial_types {
        let sz = body_size(st)?;
        values.push(decode_cell(st, sz, data, body_offset)?);
        body_offset = body_offset
            .checked_add(sz)
            .ok_or_else(|| StoreError::Integrity {
                detail: "record: body offset overflow".to_string(),
            })?;
    }

    Ok((values, body_offset - start))
}

/// Compute the record's header-end offset (`start + header_size`), failing with a
/// typed error when the addition overflows or the header runs past the buffer.
fn checked_header_end(start: usize, header_size: i64, data_len: usize) -> Result<usize> {
    if header_size < 0 {
        return Err(StoreError::Integrity {
            detail: "record: negative header size".to_string(),
        });
    }
    let header_end =
        start
            .checked_add(header_size as usize)
            .ok_or_else(|| StoreError::Integrity {
                detail: "record: header end offset overflow".to_string(),
            })?;
    if header_end > data_len {
        return Err(StoreError::Integrity {
            detail: "record: header runs past buffer".to_string(),
        });
    }
    Ok(header_end)
}

/// Decode the serial-type record at `data[offset]`, materializing ONLY the columns
/// whose index is `true` in `wanted`; every other column's body is skipped (its
/// size is read from the header and the offset advanced without copying the body)
/// and returned as `Value::Null` placeholder.
///
/// This is a read-side projection: a column the caller did not request is never
/// decoded, so a wide row whose large BLOB columns are unwanted is read without
/// copying those bytes. For every requested column the decoded value is identical
/// to [`decode_record`] — the two share `decode_cell`, so a requested column round
/// trips byte-for-byte. A `Null` returned for an unwanted column is a placeholder,
/// never a stored value; the caller must not read unrequested positions.
///
/// `wanted` shorter than the record's column count treats the missing tail as
/// unwanted; longer than the column count ignores the surplus flags.
pub fn decode_record_columns(data: &[u8], offset: usize, wanted: &[bool]) -> Result<Vec<Value>> {
    let start = offset;
    let (header_size, n) = decode_varint(data, offset)?;
    let mut pos = offset.checked_add(n).ok_or_else(|| StoreError::Integrity {
        detail: "record: header offset overflow".to_string(),
    })?;
    let header_end = checked_header_end(start, header_size, data.len())?;

    let mut serial_types: Vec<i64> = Vec::new();
    while pos < header_end {
        let (st, m) = decode_varint(data, pos)?;
        pos = pos.checked_add(m).ok_or_else(|| StoreError::Integrity {
            detail: "record: header offset overflow".to_string(),
        })?;
        serial_types.push(st);
    }

    let mut body_offset = header_end;
    let mut values: Vec<Value> = Vec::with_capacity(serial_types.len());
    for (i, st) in serial_types.into_iter().enumerate() {
        let sz = body_size(st)?;
        if wanted.get(i).copied().unwrap_or(false) {
            values.push(decode_cell(st, sz, data, body_offset)?);
        } else {
            values.push(Value::Null);
        }
        body_offset = body_offset
            .checked_add(sz)
            .ok_or_else(|| StoreError::Integrity {
                detail: "record: body offset overflow".to_string(),
            })?;
    }
    Ok(values)
}

/// Decode one cell of serial type `st` (body size `sz`) at `data[body_offset]`.
/// The single source of truth for per-type body interpretation, shared by the
/// full-row and projected decoders so a requested column is identical on both.
fn decode_cell(st: i64, sz: usize, data: &[u8], body_offset: usize) -> Result<Value> {
    // The slice end is computed with a checked add so a body size that would wrap
    // the offset becomes a typed error rather than a panic on the range itself.
    let body_end = |len: usize| -> Result<usize> {
        body_offset
            .checked_add(len)
            .ok_or_else(|| StoreError::Integrity {
                detail: "record: body end offset overflow".to_string(),
            })
    };
    let value = match st {
        0 => Value::Null,
        8 => Value::Int(0),
        9 => Value::Int(1),
        7 => {
            let raw = data
                .get(body_offset..body_end(8)?)
                .ok_or_else(|| StoreError::Integrity {
                    detail: "record: float body past end".to_string(),
                })?;
            Value::Float(f64::from_be_bytes(raw.try_into().unwrap()))
        }
        st if st >= 13 && st % 2 == 1 => {
            let raw =
                data.get(body_offset..body_end(sz)?)
                    .ok_or_else(|| StoreError::Integrity {
                        detail: "record: blob body past end".to_string(),
                    })?;
            Value::Blob(raw.to_vec())
        }
        st if st >= 12 && st % 2 == 0 => {
            let raw =
                data.get(body_offset..body_end(sz)?)
                    .ok_or_else(|| StoreError::Integrity {
                        detail: "record: text body past end".to_string(),
                    })?;
            let s = String::from_utf8(raw.to_vec()).map_err(|_| StoreError::Integrity {
                detail: "record: text column is not valid UTF-8".to_string(),
            })?;
            Value::Text(s)
        }
        _ => {
            let raw =
                data.get(body_offset..body_end(sz)?)
                    .ok_or_else(|| StoreError::Integrity {
                        detail: "record: integer body past end".to_string(),
                    })?;
            let mut acc: i64 = if raw.first().is_some_and(|b| b & 0x80 != 0) {
                -1
            } else {
                0
            };
            for &b in raw {
                acc = (acc << 8) | i64::from(b);
            }
            Value::Int(acc)
        }
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_record_columns_matches_full_decode_for_wanted() {
        // A wide row with a large BLOB in the middle: requesting only a subset of
        // columns must return those columns byte-identically to the full decode,
        // and Null for the skipped ones — including when the skipped column is the
        // large blob and when a wanted column follows it.
        let row = vec![
            Value::Int(7),
            Value::Text("id-cafe".to_string()),
            Value::Blob(vec![0xab; 1500]),
            Value::Text("tail".to_string()),
            Value::Int(1),
        ];
        let bytes = encode_record(&row);
        let full = decode_record(&bytes, 0).unwrap().0;

        // Want columns 0, 1, 3, 4 (skip the big blob at 2).
        let wanted = [true, true, false, true, true];
        let proj = decode_record_columns(&bytes, 0, &wanted).unwrap();
        assert_eq!(proj.len(), full.len());
        assert_eq!(proj[0], full[0]);
        assert_eq!(proj[1], full[1]);
        assert_eq!(proj[2], Value::Null, "skipped blob is a Null placeholder");
        assert_eq!(
            proj[3], full[3],
            "a wanted column after a skipped blob still decodes"
        );
        assert_eq!(proj[4], full[4]);

        // Wanting the blob returns it byte-for-byte.
        let want_blob = [false, false, true, false, false];
        let pb = decode_record_columns(&bytes, 0, &want_blob).unwrap();
        assert_eq!(pb[2], full[2]);
    }

    #[test]
    fn decode_record_columns_short_wanted_treats_tail_as_unwanted() {
        let row = vec![Value::Int(1), Value::Text("x".into()), Value::Int(2)];
        let bytes = encode_record(&row);
        let proj = decode_record_columns(&bytes, 0, &[true]).unwrap();
        assert_eq!(proj, vec![Value::Int(1), Value::Null, Value::Null]);
    }

    #[test]
    fn varint_roundtrips_one_to_nine_bytes() {
        let cases: &[(i64, usize)] = &[
            (0, 1),
            (0x7F, 1),
            (0x80, 2),
            (0x3FFF, 2),
            (0x4000, 3),
            (0x1F_FFFF, 3),
            (0x20_0000, 4),
            (0xFFF_FFFF, 4),
            (0x1000_0000, 5),
            (0x7_FFFF_FFFF, 5),
            (0x8_0000_0000, 6),
            (0x3FF_FFFF_FFFF, 6),
            (0x400_0000_0000, 7),
            (0x1_FFFF_FFFF_FFFF, 7),
            (0x2_0000_0000_0000, 8),
            (0xFF_FFFF_FFFF_FFFF, 8),
            (0x100_0000_0000_0000, 9),
            (i64::MAX, 9),
            (-1, 9),
            (i64::MIN, 9),
        ];
        for &(value, want_len) in cases {
            let enc = encode_varint(value);
            assert_eq!(enc.len(), want_len, "len mismatch for {value:#x}");
            let (dec, consumed) = decode_varint(&enc, 0).unwrap();
            assert_eq!(dec, value, "value mismatch for {value:#x}");
            assert_eq!(consumed, want_len, "consumed mismatch for {value:#x}");
        }
    }

    #[test]
    fn record_int_blob_roundtrip() {
        let values = vec![Value::Int(-42), Value::Blob(b"abc\x00def".to_vec())];
        let enc = encode_record(&values);
        let (dec, consumed) = decode_record(&enc, 0).unwrap();
        assert_eq!(dec, values);
        assert_eq!(consumed, enc.len());
    }

    #[test]
    fn record_all_serial_types_roundtrip() {
        let values = vec![
            Value::Null,
            Value::Int(0),
            Value::Int(1),
            Value::Int(-128),
            Value::Int(127),
            Value::Int(-32768),
            Value::Int(2_000_000_000),
            Value::Int(i64::MIN),
            Value::Int(i64::MAX),
            Value::Float(std::f64::consts::PI),
            Value::Float(-0.0),
            Value::Text("hello".to_string()),
            Value::Blob(vec![0, 1, 2, 255]),
        ];
        let enc = encode_record(&values);
        let (dec, consumed) = decode_record(&enc, 0).unwrap();
        assert_eq!(dec, values);
        assert_eq!(consumed, enc.len());
    }

    #[test]
    fn decode_varint_rejects_short_buffer() {
        let truncated = [0x80u8];
        assert!(decode_varint(&truncated, 0).is_err());
    }

    use proptest::prelude::*;

    fn any_value() -> impl Strategy<Value = Value> {
        prop_oneof![
            Just(Value::Null),
            any::<i64>().prop_map(Value::Int),
            any::<f64>()
                .prop_filter("finite", |f| f.is_finite())
                .prop_map(Value::Float),
            "\\PC*".prop_map(Value::Text),
            prop::collection::vec(any::<u8>(), 0..96).prop_map(Value::Blob),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        // For any row and any wanted mask, the projected decode equals the full
        // decode on every wanted column and is Null on every unwanted column.
        #[test]
        fn projected_decode_agrees_with_full(
            row in prop::collection::vec(any_value(), 0..12),
            mask_seed in prop::collection::vec(any::<bool>(), 0..12),
        ) {
            let bytes = encode_record(&row);
            let full = decode_record(&bytes, 0).unwrap().0;
            let mut wanted: Vec<bool> = mask_seed;
            while wanted.len() < row.len() { wanted.push(false); }
            let proj = decode_record_columns(&bytes, 0, &wanted).unwrap();
            prop_assert_eq!(proj.len(), full.len());
            for i in 0..full.len() {
                if wanted.get(i).copied().unwrap_or(false) {
                    prop_assert_eq!(&proj[i], &full[i]);
                } else {
                    prop_assert_eq!(&proj[i], &Value::Null);
                }
            }
        }
    }
}
