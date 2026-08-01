//! The SQL-row ↔ record bridge.
//!
//! A SQL row is a vector of typed cells ([`lillibrain::Value`]); a stored record
//! is the opaque byte payload of a B-tree leaf cell. This module adapts one to
//! the other by deferring entirely to the storage layer's serial-type codec —
//! it does not re-implement the varint / serial-type logic, which lives in
//! `lillibrain::codec` and is property-tested there.
//!
//! TEXT and BLOB cells round-trip byte-for-byte: the literal surface of a stored
//! value is opaque bytes, never trimmed, normalized, or re-encoded. A round trip
//! of any cell vector returns an identical vector, including embedded NUL bytes,
//! multi-byte sequences, and trailing whitespace.

use lillibrain::{decode_record, decode_record_columns, encode_record, Value};

use crate::error::{EngineError, Result};

/// Encode a SQL row of typed cells into a stored record's byte payload.
pub fn encode_row(values: &[Value]) -> Vec<u8> {
    encode_record(values)
}

/// Decode a stored record's byte payload back into a SQL row of typed cells.
///
/// Fails when the payload is malformed (a truncated header or a body that runs
/// past the buffer); the carried message is the storage layer's own text.
pub fn decode_row(bytes: &[u8]) -> Result<Vec<Value>> {
    let (values, _consumed) =
        decode_record(bytes, 0).map_err(|e| EngineError::parse(format!("row decode: {e}")))?;
    Ok(values)
}

/// Decode a stored record materializing only the columns flagged `true` in
/// `wanted`; unwanted columns (e.g. the large embedding BLOB on a filtering scan
/// that neither filters nor projects it) are skipped without copying their body
/// and returned as a `Value::Null` placeholder. A wanted column decodes identically
/// to [`decode_row`]. The caller must request every column the WHERE and the
/// projection read, and never read an unrequested position.
pub fn decode_row_columns(bytes: &[u8], wanted: &[bool]) -> Result<Vec<Value>> {
    decode_record_columns(bytes, 0, wanted)
        .map_err(|e| EngineError::parse(format!("row decode: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn round_trips_mixed_row() {
        let row = vec![
            Value::Int(1),
            Value::Text("id-a".to_string()),
            Value::Null,
            Value::Blob(vec![0x00, 0xff]),
        ];
        let bytes = encode_row(&row);
        assert_eq!(decode_row(&bytes).unwrap(), row);
    }

    #[test]
    fn null_maps_to_zero_serial_type() {
        // A NULL cell contributes zero body bytes; a row of three NULLs decodes
        // back to three NULLs with no body to interpret.
        let row = vec![Value::Null, Value::Null, Value::Null];
        let bytes = encode_row(&row);
        assert_eq!(decode_row(&bytes).unwrap(), row);
    }

    #[test]
    fn verbatim_text_corpus_round_trips_byte_for_byte() {
        // NUL, emoji, CJK, RTL, and trailing whitespace must survive intact.
        let corpus = [
            "plain",
            "with\ttab and  double space",
            "trailing whitespace   ",
            "embedded\0nul",
            "emoji 🧠🔬 mix",
            "日本語のテキスト",
            "مرحبا بالعالم", // RTL Arabic
            "combining é vs e\u{0301}",
            "",
        ];
        for s in corpus {
            let row = vec![Value::Text(s.to_string())];
            let bytes = encode_row(&row);
            let back = decode_row(&bytes).unwrap();
            assert_eq!(back, row, "text cell {s:?} did not round-trip");
            if let Value::Text(out) = &back[0] {
                assert_eq!(out.as_bytes(), s.as_bytes(), "byte-level drift on {s:?}");
            }
        }
    }

    #[test]
    fn verbatim_blob_corpus_round_trips_byte_for_byte() {
        let corpus: &[&[u8]] = &[
            b"",
            b"\x00",
            b"\x00\x01\x02\xfe\xff",
            &[0xff; 256],
            b"not-utf8: \xc3\x28 \xff",
        ];
        for &b in corpus {
            let row = vec![Value::Blob(b.to_vec())];
            let bytes = encode_row(&row);
            let back = decode_row(&bytes).unwrap();
            assert_eq!(back, vec![Value::Blob(b.to_vec())]);
            if let Value::Blob(out) = &back[0] {
                assert_eq!(out.as_slice(), b, "blob bytes drifted");
            }
        }
    }

    #[test]
    fn malformed_payload_is_an_error() {
        // A lone continuation-flagged byte is not a complete record header.
        assert!(decode_row(&[0x80]).is_err());
    }

    // A strategy over arbitrary cell values, including byte sequences that are
    // not valid UTF-8 (carried as Blob) and strings with control characters.
    fn any_value() -> impl Strategy<Value = Value> {
        prop_oneof![
            Just(Value::Null),
            any::<i64>().prop_map(Value::Int),
            any::<f64>()
                .prop_filter("finite", |f| f.is_finite())
                .prop_map(Value::Float),
            "\\PC*".prop_map(Value::Text),
            prop::collection::vec(any::<u8>(), 0..64).prop_map(Value::Blob),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn arbitrary_row_round_trips(row in prop::collection::vec(any_value(), 0..12)) {
            let bytes = encode_row(&row);
            let back = decode_row(&bytes).unwrap();
            prop_assert_eq!(back, row);
        }
    }
}
