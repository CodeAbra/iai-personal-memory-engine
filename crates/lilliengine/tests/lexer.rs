//! Lexer parity tests: the Rust tokenizer must emit the same token stream the
//! reference Python lexer produces for the supported subset.

use lilliengine::lexer::{tokenize, Token};

/// Reduce a token stream to comparable `(kind, rendered_value)` pairs, where the
/// rendered value mirrors what the reference Python `Token.value` carries (raw
/// source text for STRING/BLOB, the decimal text for numbers, the operator
/// string for OP, etc.). Keeps the assertions terse and parity-focused.
fn kinds(toks: &[Token]) -> Vec<&'static str> {
    toks.iter()
        .map(|t| match t {
            Token::Keyword(_) => "KEYWORD",
            Token::Ident(_) => "IDENT",
            Token::Integer(_) => "INTEGER",
            Token::Real(_) => "REAL",
            Token::Str(_) => "STRING",
            Token::Blob(_) => "BLOB",
            Token::Placeholder => "PLACEHOLDER",
            Token::NamedParam(_) => "NAMED_PARAM",
            Token::Op(_) => "OP",
            Token::Punct(_) => "PUNCT",
            Token::Eof => "EOF",
        })
        .collect()
}

#[test]
fn select_star_from_where_placeholder() {
    let toks = tokenize("SELECT * FROM records WHERE id = ?").unwrap();
    assert_eq!(
        kinds(&toks),
        vec![
            "KEYWORD", // SELECT
            "PUNCT",   // *
            "KEYWORD", // FROM
            "IDENT",   // records
            "KEYWORD", // WHERE
            "IDENT",   // id
            "OP",      // =
            "PLACEHOLDER",
            "EOF",
        ]
    );
}

#[test]
fn keyword_case_insensitive() {
    let toks = tokenize("select Insert UPDATE").unwrap();
    assert_eq!(kinds(&toks), vec!["KEYWORD", "KEYWORD", "KEYWORD", "EOF"]);
}

#[test]
fn non_keyword_identifier() {
    let toks = tokenize("records").unwrap();
    assert!(matches!(toks[0], Token::Ident(ref s) if s == "records"));
}

#[test]
fn named_param_carries_name_without_colon() {
    let toks = tokenize(":pat").unwrap();
    match &toks[0] {
        Token::NamedParam(name) => assert_eq!(name, "pat"),
        other => panic!("expected NamedParam, got {other:?}"),
    }
    assert!(matches!(toks[1], Token::Eof));
}

#[test]
fn positional_placeholder() {
    let toks = tokenize("?").unwrap();
    assert!(matches!(toks[0], Token::Placeholder));
}

#[test]
fn blob_literal_keeps_source_form() {
    let toks = tokenize("x'00ff'").unwrap();
    match &toks[0] {
        Token::Blob(raw) => assert_eq!(raw, "x'00ff'"),
        other => panic!("expected Blob, got {other:?}"),
    }
}

#[test]
fn string_literal_keeps_surrounding_quotes() {
    // The reference lexer keeps the raw quoted form; the parser strips the
    // outer quotes later. The single-quote regex is not doubled-quote aware,
    // so 'it''s' splits into two STRING tokens exactly as the reference does.
    let toks = tokenize("'hello'").unwrap();
    match &toks[0] {
        Token::Str(raw) => assert_eq!(raw, "'hello'"),
        other => panic!("expected Str, got {other:?}"),
    }

    let split = tokenize("'it''s'").unwrap();
    assert_eq!(kinds(&split), vec!["STRING", "STRING", "EOF"]);
    match (&split[0], &split[1]) {
        (Token::Str(a), Token::Str(b)) => {
            assert_eq!(a, "'it'");
            assert_eq!(b, "'s'");
        }
        other => panic!("expected two Str tokens, got {other:?}"),
    }
}

#[test]
fn string_with_backslash_escape() {
    let toks = tokenize("'a\\nb'").unwrap();
    match &toks[0] {
        Token::Str(raw) => assert_eq!(raw, "'a\\nb'"),
        other => panic!("expected Str, got {other:?}"),
    }
}

#[test]
fn integer_vs_real() {
    let real = tokenize("1.5").unwrap();
    assert!(matches!(real[0], Token::Real(_)));
    let int = tokenize("42").unwrap();
    assert!(matches!(int[0], Token::Integer(_)));
    match (&real[0], &int[0]) {
        (Token::Real(f), Token::Integer(i)) => {
            assert_eq!(*f, 1.5);
            assert_eq!(*i, 42);
        }
        _ => unreachable!(),
    }
}

#[test]
fn operators_are_single_tokens_and_normalized() {
    let toks = tokenize("a <= b >= c != d <> e == f < g > h").unwrap();
    let ops: Vec<&str> = toks
        .iter()
        .filter_map(|t| match t {
            Token::Op(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    // == normalizes to =; every other comparison operator is carried verbatim.
    assert_eq!(ops, vec!["<=", ">=", "!=", "<>", "=", "<", ">"]);
}

#[test]
fn whitespace_and_comments_skipped() {
    let toks = tokenize("SELECT  --a comment\n  *  FROM t").unwrap();
    assert_eq!(
        kinds(&toks),
        vec!["KEYWORD", "PUNCT", "KEYWORD", "IDENT", "EOF"]
    );
}

#[test]
fn bracket_quoted_identifier_unwrapped() {
    let toks = tokenize("[records]").unwrap();
    match &toks[0] {
        Token::Ident(s) => assert_eq!(s, "records"),
        other => panic!("expected Ident, got {other:?}"),
    }
}

#[test]
fn punctuation_tokens() {
    let toks = tokenize("(),;.*").unwrap();
    assert_eq!(
        kinds(&toks),
        vec!["PUNCT", "PUNCT", "PUNCT", "PUNCT", "PUNCT", "PUNCT", "EOF"]
    );
}

#[test]
fn unrecognized_char_fails_loud() {
    let err = tokenize("SELECT @ FROM t").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("unexpected token near:"), "message was: {msg}");
}

#[test]
fn empty_input_is_just_eof() {
    let toks = tokenize("   ").unwrap();
    assert_eq!(kinds(&toks), vec!["EOF"]);
}
