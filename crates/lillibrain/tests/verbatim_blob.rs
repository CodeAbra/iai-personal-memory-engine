//! Verbatim byte round-trip checks for the record codec.
//!
//! The stored blob is opaque ciphertext from the engine's view. These checks
//! prove that a blob is returned byte-for-byte: embedded NUL bytes, multi-byte
//! UTF-8 (emoji, CJK), right-to-left marks, and trailing whitespace all survive
//! an encode/decode round-trip with raw-slice equality.

use lillibrain::codec::{decode_record, encode_record, Value};

/// Encode a single blob column, decode it, and return the recovered bytes.
fn blob_roundtrip(payload: &[u8]) -> Vec<u8> {
    let enc = encode_record(&[Value::Blob(payload.to_vec())]);
    let (dec, consumed) = decode_record(&enc, 0).unwrap();
    assert_eq!(consumed, enc.len(), "decode consumed wrong byte count");
    assert_eq!(dec.len(), 1, "expected exactly one column");
    match &dec[0] {
        Value::Blob(b) => b.clone(),
        other => panic!("expected a blob column, got {other:?}"),
    }
}

#[test]
fn blob_with_embedded_nul_bytes_is_verbatim() {
    let payload = b"head\x00\x00mid\x00tail";
    assert_eq!(blob_roundtrip(payload), payload);
}

#[test]
fn blob_with_emoji_is_verbatim() {
    let payload = "fire 🔥 rocket 🚀 family 👨‍👩‍👧‍👦".as_bytes();
    assert_eq!(blob_roundtrip(payload), payload);
}

#[test]
fn blob_with_cjk_is_verbatim() {
    let payload = "日本語と中文と한국어".as_bytes();
    assert_eq!(blob_roundtrip(payload), payload);
}

#[test]
fn blob_with_rtl_marks_is_verbatim() {
    // Arabic + Hebrew interleaved with explicit RTL/LTR override marks.
    let payload = "\u{202B}مرحبا\u{200F}שלום\u{202C}world".as_bytes();
    assert_eq!(blob_roundtrip(payload), payload);
}

#[test]
fn blob_with_trailing_whitespace_is_verbatim() {
    let payload = b"value with trailing spaces and tabs \t  \n\r  ";
    let recovered = blob_roundtrip(payload);
    assert_eq!(recovered, payload, "trailing whitespace was altered");
    assert_eq!(
        recovered.len(),
        payload.len(),
        "trailing bytes were trimmed"
    );
}

#[test]
fn blob_full_corpus_in_one_record_is_verbatim() {
    let nul = b"a\x00b\x00\x00c".to_vec();
    let emoji = "🔥🚀✨".as_bytes().to_vec();
    let cjk = "漢字テスト".as_bytes().to_vec();
    let rtl = "\u{202B}اختبار\u{202C}".as_bytes().to_vec();
    let trailing = b"keep me   \t\n".to_vec();

    let values = vec![
        Value::Blob(nul.clone()),
        Value::Blob(emoji.clone()),
        Value::Blob(cjk.clone()),
        Value::Blob(rtl.clone()),
        Value::Blob(trailing.clone()),
    ];
    let enc = encode_record(&values);
    let (dec, consumed) = decode_record(&enc, 0).unwrap();
    assert_eq!(consumed, enc.len());
    assert_eq!(dec[0], Value::Blob(nul));
    assert_eq!(dec[1], Value::Blob(emoji));
    assert_eq!(dec[2], Value::Blob(cjk));
    assert_eq!(dec[3], Value::Blob(rtl));
    assert_eq!(dec[4], Value::Blob(trailing));
}

#[test]
fn empty_blob_roundtrips() {
    assert_eq!(blob_roundtrip(b""), b"");
}
