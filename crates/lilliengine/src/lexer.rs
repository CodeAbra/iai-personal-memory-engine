//! Single-pass SQL tokenizer.
//!
//! Ports the reference scanner: an ordered set of matchers tried left-to-right
//! at each position, first match wins. Whitespace and `--` line comments are
//! discarded. Every input that no matcher accepts is a fail-loud error rather
//! than a silently skipped character, so a stray symbol never produces a
//! truncated token stream.
//!
//! Token values mirror the reference: STRING and BLOB carry their raw source
//! form (surrounding quotes included — the parser strips them); `==` normalizes
//! to `=`; a bracket-quoted `[name]` identifier is carried unwrapped; a named
//! `:name` parameter carries the name without its leading colon.

use crate::error::{EngineError, Result};

/// A single lexer token.
///
/// The variants line up one-to-one with the reference token kinds
/// (KEYWORD / IDENT / INTEGER / REAL / STRING / BLOB / PLACEHOLDER /
/// NAMED_PARAM / OP / PUNCT / EOF). `Str` and `Blob` hold the raw source text
/// including the surrounding quotes; the parser strips and decodes them.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A reserved keyword (matched case-insensitively against a fixed set).
    Keyword(String),
    /// An identifier (table / column name), or a bracket-quoted `[name]`.
    Ident(String),
    /// An integer literal.
    Integer(i64),
    /// A real (floating-point) literal.
    Real(f64),
    /// A single-quoted string literal, carried with its surrounding quotes.
    Str(String),
    /// A hex blob literal `x'..'`, carried with its prefix and quotes.
    Blob(String),
    /// A positional `?` bind placeholder.
    Placeholder,
    /// A named `:name` bind placeholder, carrying the name without the colon.
    NamedParam(String),
    /// A comparison / equality operator (`==` normalized to `=`).
    Op(String),
    /// A single punctuation character from the set `(),;.*`.
    Punct(char),
    /// End of input.
    Eof,
}

/// The reserved keyword set, upper-cased.
const KEYWORDS: &[&str] = &[
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IS",
    "NULL",
    "IN",
    "LIKE",
    "LIMIT",
    "INTO",
    "VALUES",
    "SET",
    "ON",
    "CONFLICT",
    "DO",
    "NOTHING",
    "EXCLUDED",
    "CREATE",
    "TABLE",
    "IF",
    "EXISTS",
    "INDEX",
    "ALTER",
    "ADD",
    "COLUMN",
    "PRAGMA",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "INTEGER",
    "TEXT",
    "REAL",
    "BLOB",
    "PRIMARY",
    "KEY",
    "AUTOINCREMENT",
    "UNIQUE",
    "DEFAULT",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "COUNT",
    "SUM",
    "MIN",
    "MAX",
    "IGNORE",
    "REPLACE",
    "TRUE",
    "FALSE",
];

/// True when `word` (any case) is a reserved keyword.
fn is_keyword(word: &str) -> bool {
    let upper = word.to_ascii_uppercase();
    KEYWORDS.iter().any(|k| *k == upper)
}

/// Tokenize a SQL string into a token list ending in [`Token::Eof`].
///
/// Discards whitespace and `--` line comments in a single pass. Returns
/// [`EngineError::ParseError`] when a character cannot be matched — the same
/// fail-loud guard the reference's scanner-remainder check provides.
pub fn tokenize(sql: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = sql.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut out: Vec<Token> = Vec::new();

    while i < n {
        let c = chars[i];

        // -- line comment: discard to end of line.
        if c == '-' && i + 1 < n && chars[i + 1] == '-' {
            i += 2;
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // Whitespace: discard.
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Operators: == | != | <> | <= | >= | [=<>]  (longest-first within rule).
        if let Some(len) = match_two_char_op(&chars, i) {
            let op: String = chars[i..i + len].iter().collect();
            let normalized = if op == "==" { "=".to_string() } else { op };
            out.push(Token::Op(normalized));
            i += len;
            continue;
        }
        if c == '=' || c == '<' || c == '>' {
            out.push(Token::Op(c.to_string()));
            i += 1;
            continue;
        }

        // ? positional placeholder.
        if c == '?' {
            out.push(Token::Placeholder);
            i += 1;
            continue;
        }

        // :name named parameter.
        if c == ':' && i + 1 < n && is_ident_start(chars[i + 1]) {
            let mut j = i + 1;
            while j < n && is_ident_continue(chars[j]) {
                j += 1;
            }
            let name: String = chars[i + 1..j].iter().collect();
            out.push(Token::NamedParam(name));
            i = j;
            continue;
        }

        // Punctuation: ( ) , ; . *
        if matches!(c, '(' | ')' | ',' | ';' | '.' | '*') {
            out.push(Token::Punct(c));
            i += 1;
            continue;
        }

        // Real: [0-9]+\.[0-9]+   (checked before the integer rule).
        if c.is_ascii_digit() {
            let mut j = i;
            while j < n && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j < n && chars[j] == '.' && j + 1 < n && chars[j + 1].is_ascii_digit() {
                let mut k = j + 1;
                while k < n && chars[k].is_ascii_digit() {
                    k += 1;
                }
                let text: String = chars[i..k].iter().collect();
                let f: f64 = text
                    .parse()
                    .map_err(|_| EngineError::parse(format!("invalid real literal: {text:?}")))?;
                out.push(Token::Real(f));
                i = k;
                continue;
            }
            // Integer: [0-9]+
            let text: String = chars[i..j].iter().collect();
            let v: i64 = text
                .parse()
                .map_err(|_| EngineError::parse(format!("invalid integer literal: {text:?}")))?;
            out.push(Token::Integer(v));
            i = j;
            continue;
        }

        // String: '(?:[^'\\]|\\.)*'  — backslash-escape aware, not doubled-quote
        // aware (matching the reference regex exactly).
        if c == '\'' {
            if let Some(end) = match_string(&chars, i) {
                let raw: String = chars[i..=end].iter().collect();
                out.push(Token::Str(raw));
                i = end + 1;
                continue;
            }
            return Err(EngineError::parse(format!(
                "unexpected token near: {:?}",
                remainder(&chars, i)
            )));
        }

        // Blob: [xX]'[0-9a-fA-F]*'   (after the string rule, before the ident rule).
        if (c == 'x' || c == 'X') && i + 1 < n && chars[i + 1] == '\'' {
            if let Some(end) = match_blob(&chars, i) {
                let raw: String = chars[i..=end].iter().collect();
                out.push(Token::Blob(raw));
                i = end + 1;
                continue;
            }
            // An `x'` that is not a valid hex blob falls through to the ident
            // rule (so a bare `x` identifier still tokenizes), mirroring the
            // reference where the blob alternative simply does not match.
        }

        // Bracket-quoted identifier: \[[^\]]+\]  -> unwrapped IDENT.
        if c == '[' {
            if let Some(end) = chars[i + 1..n].iter().position(|&ch| ch == ']') {
                let close = i + 1 + end;
                if close > i + 1 {
                    let name: String = chars[i + 1..close].iter().collect();
                    out.push(Token::Ident(name));
                    i = close + 1;
                    continue;
                }
            }
            return Err(EngineError::parse(format!(
                "unexpected token near: {:?}",
                remainder(&chars, i)
            )));
        }

        // Identifier / keyword: [A-Za-z_][A-Za-z0-9_]*
        if is_ident_start(c) {
            let mut j = i;
            while j < n && is_ident_continue(chars[j]) {
                j += 1;
            }
            let word: String = chars[i..j].iter().collect();
            if is_keyword(&word) {
                out.push(Token::Keyword(word));
            } else {
                out.push(Token::Ident(word));
            }
            i = j;
            continue;
        }

        // No matcher accepted this character — fail loud.
        return Err(EngineError::parse(format!(
            "unexpected token near: {:?}",
            remainder(&chars, i)
        )));
    }

    out.push(Token::Eof);
    Ok(out)
}

/// First 20 characters from `start`, for the fail-loud message (mirrors the
/// reference `remainder[:20]`).
fn remainder(chars: &[char], start: usize) -> String {
    chars[start..(start + 20).min(chars.len())].iter().collect()
}

/// Length (2) of a two-character operator at `i`, or `None`.
fn match_two_char_op(chars: &[char], i: usize) -> Option<usize> {
    if i + 1 >= chars.len() {
        return None;
    }
    let pair = (chars[i], chars[i + 1]);
    match pair {
        ('=', '=') | ('!', '=') | ('<', '>') | ('<', '=') | ('>', '=') => Some(2),
        _ => None,
    }
}

/// Index of the closing quote of a `'...'` string starting at `i`, or `None`
/// when the literal is unterminated. Backslash escapes any next character; a
/// bare `'` ends the literal (no doubled-quote handling, per the reference).
fn match_string(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    let mut j = i + 1;
    while j < n {
        match chars[j] {
            '\\' => {
                // Escapes the following character; skip both.
                j += 2;
            }
            '\'' => return Some(j),
            _ => j += 1,
        }
    }
    None
}

/// Index of the closing quote of a `x'<hex>'` blob starting at `i`, or `None`
/// when the body is not all hex digits or the literal is unterminated.
fn match_blob(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    // chars[i] is x/X, chars[i+1] is the opening quote.
    let mut j = i + 2;
    while j < n {
        match chars[j] {
            '\'' => return Some(j),
            c if c.is_ascii_hexdigit() => j += 1,
            _ => return None,
        }
    }
    None
}

/// True when `c` can start an identifier.
fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic()
}

/// True when `c` can continue an identifier.
fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_ascii_alphanumeric()
}
