//! Three-valued expression evaluation with sqlite3 comparison and NULL parity.
//!
//! This module reproduces the reference engine's SQL value semantics exactly:
//! Kleene three-valued AND / OR / NOT, NULL-propagating comparisons with numeric
//! type-affinity coercion (`1 = 1.0`, `'5' = 5`), `LIKE` / `IN` / `COALESCE`, the
//! `datetime()` scalar (a hand-port, byte-equal to sqlite3's output, degrading an
//! unparseable timestamp to NULL rather than raising), and the ORDER BY sort key
//! (NULLs-first, numeric-before-text, with a descending wrapper).
//!
//! A SQL NULL is carried as [`Value::Null`]. Comparison helpers return
//! `Option<bool>` where `None` is the SQL unknown (NULL) truth value. The
//! evaluator returns an [`EvalVal`] so a boolean predicate result and a scalar
//! value are distinct: `filter_op` keeps a row only when the predicate is
//! `EvalVal::Bool(Some(true))`.

use std::collections::HashMap;

use lillibrain::Value;

/// A row of decoded cells keyed by column name, preserving column order.
///
/// Column order is preserved so `SELECT *` and positional access reproduce the
/// schema's declaration order; name lookup backs the expression evaluator.
#[derive(Debug, Clone, Default)]
pub struct Row {
    cells: Vec<(String, Value)>,
    index: HashMap<String, usize>,
}

impl Row {
    /// An empty row.
    pub fn new() -> Self {
        Row::default()
    }

    /// Build a row from `(name, value)` pairs in column order.
    pub fn from_pairs(pairs: Vec<(String, Value)>) -> Self {
        let mut row = Row::default();
        for (name, value) in pairs {
            row.set(name, value);
        }
        row
    }

    /// Set `name` to `value`, appending it when new or replacing it in place.
    pub fn set(&mut self, name: impl Into<String>, value: Value) {
        let name = name.into();
        match self.index.get(&name) {
            Some(&i) => self.cells[i].1 = value,
            None => {
                self.index.insert(name.clone(), self.cells.len());
                self.cells.push((name, value));
            }
        }
    }

    /// The value for `name`, or `None` when the column is absent from the row.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.index.get(name).map(|&i| &self.cells[i].1)
    }

    /// The value for `name`, treating an absent column as SQL NULL.
    pub fn get_or_null(&self, name: &str) -> Value {
        self.get(name).cloned().unwrap_or(Value::Null)
    }

    /// True when the row carries a column called `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// The `(name, value)` pairs in column order.
    pub fn pairs(&self) -> &[(String, Value)] {
        &self.cells
    }
}

/// The result of evaluating an expression against a row.
///
/// `Val` is a scalar value (a column read, a literal, a `COALESCE` / `datetime`
/// result). `Bool` is a three-valued predicate truth: `Some(true)`, `Some(false)`,
/// or `None` (the SQL unknown). The two are distinct so a predicate's NULL truth
/// (`Bool(None)`) is not confused with a NULL scalar (`Val(Value::Null)`).
#[derive(Debug, Clone, PartialEq)]
pub enum EvalVal {
    /// A scalar value.
    Val(Value),
    /// A three-valued boolean truth (`None` == SQL unknown / NULL).
    Bool(Option<bool>),
}

impl EvalVal {
    /// Reduce the result to a scalar value: a true predicate is `Int(1)`, a false
    /// predicate is `Int(0)`, and an unknown predicate is `Null`.
    fn into_value(self) -> Value {
        match self {
            EvalVal::Val(v) => v,
            EvalVal::Bool(Some(true)) => Value::Int(1),
            EvalVal::Bool(Some(false)) => Value::Int(0),
            EvalVal::Bool(None) => Value::Null,
        }
    }

    /// The three-valued truth of this result for predicate evaluation.
    ///
    /// A `Bool` keeps its truth directly. A scalar `Null` is unknown; any other
    /// scalar is truthy by its sqlite3 truthiness (a non-zero number / non-empty
    /// numeric — here matched against the reference's `bool(value)` test).
    fn truth(&self) -> Option<bool> {
        match self {
            EvalVal::Bool(b) => *b,
            EvalVal::Val(Value::Null) => None,
            EvalVal::Val(v) => Some(value_is_truthy(v)),
        }
    }
}

/// The sqlite3-style truthiness of a non-NULL value, matching Python's `bool()`
/// as the reference's three-valued operators apply it: a zero number / empty
/// string / empty blob is false, everything else true.
fn value_is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Int(i) => *i != 0,
        Value::Float(f) => *f != 0.0,
        Value::Text(s) => !s.is_empty(),
        Value::Blob(b) => !b.is_empty(),
    }
}

// ---------------------------------------------------------------------------
// Three-valued comparison and boolean operators
// ---------------------------------------------------------------------------

/// Coerce two operands for comparison following sqlite3 type affinity.
///
/// When one side is an integer and the other a numeric-looking text, the text is
/// parsed to the matching numeric so `'5' = 5` and `'5' < 6` hold; a non-numeric
/// text is left untouched (the class-rank comparison then keeps numbers before
/// text). Mirrors the reference `_coerce_for_compare`, extended to the
/// `Float`/`Text` pair so a real column compares against a numeric string too.
fn coerce_for_compare(a: Value, b: Value) -> (Value, Value) {
    match (&a, &b) {
        (Value::Int(_), Value::Text(s)) => match s.parse::<i64>() {
            Ok(i) => (a, Value::Int(i)),
            Err(_) => (a, b),
        },
        (Value::Text(s), Value::Int(_)) => match s.parse::<i64>() {
            Ok(i) => (Value::Int(i), b),
            Err(_) => (a, b),
        },
        (Value::Float(_), Value::Text(s)) => match s.parse::<f64>() {
            Ok(f) => (a, Value::Float(f)),
            Err(_) => (a, b),
        },
        (Value::Text(s), Value::Float(_)) => match s.parse::<f64>() {
            Ok(f) => (Value::Float(f), b),
            Err(_) => (a, b),
        },
        _ => (a, b),
    }
}

/// Compare two coerced values, returning the ascending ordering or `None` when
/// the two belong to different storage classes whose order is fixed by class
/// rank alone (NULL < number < text < blob) — never an in-class value compare.
fn order_compare(a: &Value, b: &Value) -> std::cmp::Ordering {
    let (ra, ka) = order_key(a);
    let (rb, kb) = order_key(b);
    match ra.cmp(&rb) {
        std::cmp::Ordering::Equal => ka.cmp(&kb),
        other => other,
    }
}

/// Three-valued comparison: any NULL operand yields `None` (SQL unknown).
///
/// `op` is one of `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`. Equality first applies
/// numeric affinity coercion so `1 = 1.0` and `'5' = 5` are true; ordering uses
/// the storage-class collation so a number always sorts before text.
pub fn sql_compare(op: &str, a: Value, b: Value) -> Option<bool> {
    if matches!(a, Value::Null) || matches!(b, Value::Null) {
        return None;
    }
    let (a, b) = coerce_for_compare(a, b);
    let ord = order_compare(&a, &b);
    let result = match op {
        "=" | "==" => ord == std::cmp::Ordering::Equal,
        "!=" | "<>" => ord != std::cmp::Ordering::Equal,
        "<" => ord == std::cmp::Ordering::Less,
        "<=" => ord != std::cmp::Ordering::Greater,
        ">" => ord == std::cmp::Ordering::Greater,
        ">=" => ord != std::cmp::Ordering::Less,
        _ => return None,
    };
    Some(result)
}

/// Kleene three-valued AND: a definite FALSE on either side wins; otherwise a
/// NULL operand yields unknown.
pub fn sql_and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    if a == Some(false) || b == Some(false) {
        return Some(false);
    }
    if a.is_none() || b.is_none() {
        return None;
    }
    Some(true)
}

/// Kleene three-valued OR: a definite TRUE on either side wins; otherwise a NULL
/// operand yields unknown.
pub fn sql_or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    if a == Some(true) || b == Some(true) {
        return Some(true);
    }
    if a.is_none() || b.is_none() {
        return None;
    }
    Some(false)
}

/// Three-valued NOT: `NOT NULL` is NULL.
pub fn sql_not(a: Option<bool>) -> Option<bool> {
    a.map(|x| !x)
}

// ---------------------------------------------------------------------------
// LIKE / IN / COALESCE
// ---------------------------------------------------------------------------

/// ASCII case-insensitive `LIKE`: `%` matches any run, `_` matches one character.
///
/// A NULL value or pattern yields NULL. A non-text value is coerced to its text
/// form first (sqlite3 stringifies the LHS for LIKE). The pattern is matched
/// character-by-character with backtracking over `%`, mirroring the reference's
/// regex semantics without a regex dependency.
pub fn like_match(value: &Value, pattern: &Value) -> Option<bool> {
    if matches!(value, Value::Null) || matches!(pattern, Value::Null) {
        return None;
    }
    let pat = match pattern {
        Value::Text(s) => s.clone(),
        _ => return Some(false),
    };
    let text = value_to_text(value);
    Some(like_ci(text.as_bytes(), pat.as_bytes()))
}

/// Render a value to the text form sqlite3 uses for a LIKE left operand.
fn value_to_text(v: &Value) -> String {
    match v {
        Value::Text(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => crate::catalog::render_float_text(*f),
        Value::Blob(b) => String::from_utf8_lossy(b).into_owned(),
        Value::Null => String::new(),
    }
}

/// ASCII case-insensitive wildcard match with `%` (any run) and `_` (one char).
fn like_ci(text: &[u8], pat: &[u8]) -> bool {
    // Recursive backtracking match over the two byte slices. `%` consumes any
    // run (including empty); `_` consumes exactly one byte; everything else
    // matches a single byte ASCII-case-insensitively.
    let mut ti = 0usize;
    let mut pi = 0usize;
    let mut star_pat: Option<usize> = None;
    let mut star_text = 0usize;
    while ti < text.len() {
        if pi < pat.len() && pat[pi] == b'%' {
            star_pat = Some(pi);
            star_text = ti;
            pi += 1;
        } else if pi < pat.len() && (pat[pi] == b'_' || ascii_eq_ci(pat[pi], text[ti])) {
            ti += 1;
            pi += 1;
        } else if let Some(sp) = star_pat {
            pi = sp + 1;
            star_text += 1;
            ti = star_text;
        } else {
            return false;
        }
    }
    while pi < pat.len() && pat[pi] == b'%' {
        pi += 1;
    }
    pi == pat.len()
}

fn ascii_eq_ci(a: u8, b: u8) -> bool {
    a.eq_ignore_ascii_case(&b)
}

// ---------------------------------------------------------------------------
// datetime() scalar
// ---------------------------------------------------------------------------

/// Normalise a timestamp value to sqlite3 `datetime()` output, or NULL.
///
/// Parses an ISO-`T` or space-separated `YYYY-MM-DD[ HH:MM:SS[.ffffff]][Z|±HH:MM]`
/// timestamp (a bare date yields midnight), converts an offset-bearing value to
/// UTC, truncates to whole seconds, drops the timezone, and returns the
/// second-resolution space form `YYYY-MM-DD HH:MM:SS`. A NULL argument
/// propagates as NULL; an unparseable argument also yields NULL rather than
/// raising — matching sqlite3, so a single corrupt row degrades instead of
/// aborting the query.
pub fn datetime_scalar(value: &Value) -> Value {
    match value {
        Value::Null => Value::Null,
        // The 'now' time value resolves to the current UTC second, matching
        // sqlite3's single-argument datetime('now').
        Value::Text(s) if s.trim().eq_ignore_ascii_case("now") => Value::Text(now_utc_datetime()),
        Value::Text(s) => parse_datetime(s).map(Value::Text).unwrap_or(Value::Null),
        // sqlite3's datetime() coerces a number through its text form; a numeric
        // timestamp string is the only parseable shape, so a bare number is
        // unparseable and degrades to NULL.
        Value::Int(i) => parse_datetime(&i.to_string())
            .map(Value::Text)
            .unwrap_or(Value::Null),
        Value::Float(f) => parse_datetime(&crate::catalog::render_float_text(*f))
            .map(Value::Text)
            .unwrap_or(Value::Null),
        Value::Blob(_) => Value::Null,
    }
}

/// Evaluate `datetime(value, modifier, ...)` like sqlite3 for the supported
/// modifier set: `'now'` (the value argument), `'unixepoch'` (interpret the value
/// as Unix epoch seconds), and `'utc'` / `'localtime'` (no-ops on this UTC-only
/// brain). The bare single-argument form (no modifiers) is bit-identical to
/// [`datetime_scalar`], so the literal-ISO path the store/recall layer issues is
/// unchanged. An unsupported modifier or an unparseable value degrades to NULL,
/// matching sqlite3's per-row degradation.
pub fn datetime_with_modifiers(value: &Value, modifiers: &[String]) -> Value {
    if modifiers.is_empty() {
        return datetime_scalar(value);
    }
    // 'now' as the time value yields the current UTC second.
    let mut base: Option<String> = match value {
        Value::Text(s) if s.trim().eq_ignore_ascii_case("now") => Some(now_utc_datetime()),
        _ => None,
    };
    // 'unixepoch' reinterprets the numeric value as Unix epoch seconds.
    let mut i = 0;
    while i < modifiers.len() {
        let m = modifiers[i].trim().to_ascii_lowercase();
        match m.as_str() {
            "unixepoch" => {
                let secs = match value {
                    Value::Int(n) => Some(*n),
                    Value::Float(f) => Some(*f as i64),
                    Value::Text(s) => s.trim().parse::<i64>().ok(),
                    _ => None,
                };
                base = secs.map(datetime_from_unix_epoch);
            }
            // UTC-only brain: these are accepted no-ops on an already-UTC value.
            "utc" | "localtime" => {
                if base.is_none() {
                    base = text_of(datetime_scalar(value));
                }
            }
            // An unsupported modifier degrades the whole call to NULL.
            _ => return Value::Null,
        }
        i += 1;
    }
    // No epoch / now modifier consumed the value: fall back to parsing it as a
    // timestamp (the modifiers were UTC/localtime no-ops).
    let resolved = base.or_else(|| text_of(datetime_scalar(value)));
    resolved.map(Value::Text).unwrap_or(Value::Null)
}

/// The string body of a `Value::Text`, else `None`.
fn text_of(v: Value) -> Option<String> {
    match v {
        Value::Text(s) => Some(s),
        _ => None,
    }
}

/// The current UTC time as the second-resolution `YYYY-MM-DD HH:MM:SS` form.
fn now_utc_datetime() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    datetime_from_unix_epoch(secs)
}

/// Convert Unix epoch seconds (UTC) to the `YYYY-MM-DD HH:MM:SS` form.
fn datetime_from_unix_epoch(epoch_secs: i64) -> String {
    let days = epoch_secs.div_euclid(86_400);
    let rem = epoch_secs.rem_euclid(86_400);
    let (y, mo, d) = civil_from_days(days);
    let hh = (rem / 3600) as u32;
    let mi = ((rem % 3600) / 60) as u32;
    let ss = (rem % 60) as u32;
    format_dt((y, mo, d), hh, mi, ss)
}

/// Convert a day count since 1970-01-01 to a civil `(year, month, day)`, using
/// the standard days-from-civil inverse (valid across the full proleptic
/// Gregorian range).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Parse a timestamp string to the normalised `YYYY-MM-DD HH:MM:SS` UTC form,
/// or `None` when it does not parse.
fn parse_datetime(raw: &str) -> Option<String> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    // Split an optional timezone suffix: a trailing 'Z' (UTC) or a signed
    // ±HH:MM / ±HHMM offset after the time portion.
    let (body, offset_minutes) = split_offset(s)?;
    // Body is "YYYY-MM-DD" optionally followed by a 'T' or space and a time.
    let (date_part, time_part) = if let Some(idx) = body.find(['T', 't', ' ']) {
        (&body[..idx], Some(&body[idx + 1..]))
    } else {
        (body.as_str(), None)
    };
    let (y, mo, d) = parse_date(date_part)?;
    let (mut hh, mut mi, ss) = match time_part {
        Some(t) if !t.is_empty() => parse_time(t)?,
        _ => (0, 0, 0),
    };
    // Apply the timezone offset by converting to UTC: subtract the offset.
    if offset_minutes != 0 {
        let mut total = (hh as i64) * 60 + mi as i64 - offset_minutes as i64;
        // Wrap within a single day window; date roll-over is normalised below.
        let mut day_delta = 0i64;
        while total < 0 {
            total += 24 * 60;
            day_delta -= 1;
        }
        while total >= 24 * 60 {
            total -= 24 * 60;
            day_delta += 1;
        }
        hh = (total / 60) as u32;
        mi = (total % 60) as u32;
        // ss is unaffected by a whole-minute timezone offset.
        return Some(format_dt(roll_date(y, mo, d, day_delta)?, hh, mi, ss));
    }
    Some(format_dt((y, mo, d), hh, mi, ss))
}

/// Split a trailing timezone designator, returning the body and the offset in
/// minutes east of UTC (0 for a naive value or a 'Z' suffix).
fn split_offset(s: &str) -> Option<(String, i32)> {
    if let Some(stripped) = s.strip_suffix('Z').or_else(|| s.strip_suffix('z')) {
        return Some((stripped.to_string(), 0));
    }
    // Look for a sign that begins a timezone offset: it must follow a time
    // (i.e. appear after a 'T'/space and a ':' in the time part). Scan from the
    // end for '+' or '-' that is preceded by a digit and a ':' earlier.
    let bytes = s.as_bytes();
    // Find the date/time separator so we only treat a sign in the time portion
    // (not the date separators) as an offset.
    let sep = s.find(['T', 't', ' ']);
    if let Some(sep_idx) = sep {
        for i in (sep_idx + 1..bytes.len()).rev() {
            let c = bytes[i];
            if c == b'+' || c == b'-' {
                let body = &s[..i];
                let off = &s[i..];
                let minutes = parse_offset(off)?;
                return Some((body.to_string(), minutes));
            }
        }
    }
    Some((s.to_string(), 0))
}

/// Parse a `±HH:MM` or `±HHMM` or `±HH` offset to minutes east of UTC.
fn parse_offset(off: &str) -> Option<i32> {
    let sign = match off.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let rest = &off[1..];
    let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
    let (h, m) = match digits.len() {
        2 => (digits.parse::<i32>().ok()?, 0),
        4 => (
            digits[..2].parse::<i32>().ok()?,
            digits[2..].parse::<i32>().ok()?,
        ),
        _ => return None,
    };
    Some(sign * (h * 60 + m))
}

/// Parse `YYYY-MM-DD` into its components.
fn parse_date(s: &str) -> Option<(i64, u32, u32)> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let y = parts[0].parse::<i64>().ok()?;
    let mo = parts[1].parse::<u32>().ok()?;
    let d = parts[2].parse::<u32>().ok()?;
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    Some((y, mo, d))
}

/// Parse `HH:MM[:SS[.ffffff]]` into whole-second components (the fraction is
/// truncated). A missing seconds field defaults to 0.
fn parse_time(s: &str) -> Option<(u32, u32, u32)> {
    let core = s.split('.').next().unwrap_or(s);
    let parts: Vec<&str> = core.split(':').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }
    let hh = parts[0].parse::<u32>().ok()?;
    let mi = parts[1].parse::<u32>().ok()?;
    let ss = if parts.len() == 3 {
        parts[2].parse::<u32>().ok()?
    } else {
        0
    };
    if hh > 23 || mi > 59 || ss > 59 {
        return None;
    }
    Some((hh, mi, ss))
}

/// Roll a date forward or backward by whole days (offset normalisation only
/// ever moves a single day in practice).
fn roll_date(y: i64, mo: u32, d: u32, delta: i64) -> Option<(i64, u32, u32)> {
    if delta == 0 {
        return Some((y, mo, d));
    }
    let mut day = d as i64 + delta;
    let mut month = mo as i64;
    let mut year = y;
    loop {
        if day < 1 {
            month -= 1;
            if month < 1 {
                month = 12;
                year -= 1;
            }
            day += days_in_month(year, month as u32) as i64;
        } else if day > days_in_month(year, month as u32) as i64 {
            day -= days_in_month(year, month as u32) as i64;
            month += 1;
            if month > 12 {
                month = 1;
                year += 1;
            }
        } else {
            break;
        }
    }
    Some((year, month as u32, day as u32))
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

fn format_dt(date: (i64, u32, u32), hh: u32, mi: u32, ss: u32) -> String {
    let (y, mo, d) = date;
    format!("{y:04}-{mo:02}-{d:02} {hh:02}:{mi:02}:{ss:02}")
}

// ---------------------------------------------------------------------------
// ORDER BY sort key
// ---------------------------------------------------------------------------

/// Storage-class rank for ordering: NULL < number < text < blob. Cross-class
/// order uses the rank alone so a number is never compared against text.
const CLASS_NULL: u8 = 0;
const CLASS_NUMBER: u8 = 1;
const CLASS_TEXT: u8 = 2;
const CLASS_BLOB: u8 = 3;

/// A within-class sort key that orders numbers, text, and blobs correctly while
/// remaining `Ord` (numbers compare by value, including int↔real).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SortKey {
    /// The NULL placeholder; never participates in a within-class compare.
    Null,
    /// An integer, compared exactly as i64 — NEVER through f64, where
    /// adjacent integers past 2^53 collapse to the same double and an
    /// equality predicate silently matches the wrong row.
    Int(i64),
    /// A real; ordered against integers via the exact int-float comparison.
    Float(f64),
    /// A text value ordered lexicographically by bytes.
    Text(String),
    /// A blob value ordered lexicographically by bytes.
    Blob(Vec<u8>),
}

/// Exact ordering of an i64 against an f64 without precision loss: the float
/// is split into its truncated integer part and fraction, so magnitudes past
/// 2^53 (where `i as f64` rounds) still order correctly. A NaN compares
/// Equal, mirroring the previous total-order fallback (stored data never
/// carries NaN; Ord must stay consistent for sorting).
fn int_float_cmp(i: i64, f: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if f.is_nan() {
        return Ordering::Equal;
    }
    if f < -9_223_372_036_854_775_808.0 {
        return Ordering::Greater;
    }
    if f >= 9_223_372_036_854_775_808.0 {
        return Ordering::Less;
    }
    let t = f.trunc();
    let ft = t as i64;
    match i.cmp(&ft) {
        Ordering::Equal => {
            if f > t {
                Ordering::Less
            } else if f < t {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
        other => other,
    }
}

impl Eq for SortKey {}

impl PartialOrd for SortKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (SortKey::Null, SortKey::Null) => std::cmp::Ordering::Equal,
            (SortKey::Int(a), SortKey::Int(b)) => a.cmp(b),
            (SortKey::Float(a), SortKey::Float(b)) => {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
            (SortKey::Int(a), SortKey::Float(b)) => int_float_cmp(*a, *b),
            (SortKey::Float(a), SortKey::Int(b)) => int_float_cmp(*b, *a).reverse(),
            (SortKey::Text(a), SortKey::Text(b)) => a.as_bytes().cmp(b.as_bytes()),
            (SortKey::Blob(a), SortKey::Blob(b)) => a.cmp(b),
            // Different variants never co-occur within a class compare (the rank
            // separates them); fall back to a stable equal.
            _ => std::cmp::Ordering::Equal,
        }
    }
}

#[cfg(test)]
mod int_float_cmp_tests {
    use super::{int_float_cmp, order_key, SortKey};
    use lillibrain::Value;
    use std::cmp::Ordering;

    #[test]
    fn adjacent_giants_do_not_collapse() {
        // Past 2^53 both round to the same f64; the exact path must still
        // tell them apart (the differential fuzzer's shrunk counterexample).
        let a = -4_611_686_018_427_387_649_i64;
        let b = -4_611_686_018_427_387_648_i64;
        let (_, ka) = order_key(&Value::Int(a));
        let (_, kb) = order_key(&Value::Int(b));
        assert_ne!(ka.cmp(&kb), Ordering::Equal);
        assert_eq!(ka.cmp(&kb), Ordering::Less);
    }

    #[test]
    fn int_float_equality_and_fractions() {
        assert_eq!(int_float_cmp(1, 1.0), Ordering::Equal);
        assert_eq!(int_float_cmp(2, 2.5), Ordering::Less);
        assert_eq!(int_float_cmp(3, 2.5), Ordering::Greater);
        assert_eq!(int_float_cmp(-2, -2.5), Ordering::Greater);
        assert_eq!(int_float_cmp(-3, -2.5), Ordering::Less);
    }

    #[test]
    fn floats_beyond_i64_range_order_by_sign() {
        assert_eq!(int_float_cmp(i64::MAX, 1e19), Ordering::Less);
        assert_eq!(int_float_cmp(i64::MIN, -1e19), Ordering::Greater);
    }

    #[test]
    fn giant_int_vs_nearby_float() {
        // 2^62 as f64 is exact; 2^62+1 is not representable — the exact
        // comparison must still order them.
        let giant = (1_i64 << 62) + 1;
        let f = (1_i64 << 62) as f64;
        assert_eq!(int_float_cmp(giant, f), Ordering::Greater);
    }

    #[test]
    fn cross_variant_orderings_reverse_consistently() {
        let i = SortKey::Int(10);
        let f = SortKey::Float(9.5);
        assert_eq!(i.cmp(&f), Ordering::Greater);
        assert_eq!(f.cmp(&i), Ordering::Less);
    }
}

/// Map a value to its ascending `(class_rank, within_class_key)` sort key,
/// reproducing sqlite3's NULLs-first, numeric-before-text collation order.
pub fn order_key(value: &Value) -> (u8, SortKey) {
    match value {
        Value::Null => (CLASS_NULL, SortKey::Null),
        Value::Int(i) => (CLASS_NUMBER, SortKey::Int(*i)),
        Value::Float(f) => (CLASS_NUMBER, SortKey::Float(*f)),
        Value::Text(s) => (CLASS_TEXT, SortKey::Text(s.clone())),
        Value::Blob(b) => (CLASS_BLOB, SortKey::Blob(b.clone())),
    }
}

/// A composite ORDER BY key element wrapped for a direction. Ascending keys sort
/// naturally; a descending key inverts the comparison so a single stable sort
/// yields descending order while leaving equal-key rows in their prior order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectedKey {
    /// The ascending sort key.
    pub key: (u8, SortKey),
    /// True when this element sorts descending.
    pub desc: bool,
}

impl PartialOrd for DirectedKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DirectedKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let base = self.key.cmp(&other.key);
        if self.desc {
            base.reverse()
        } else {
            base
        }
    }
}

/// Build a directed sort key for one ORDER BY element, normalising a
/// `datetime(<col>)` key to its second-resolution value first.
pub fn sort_value(value: &Value, desc: bool, as_datetime: bool) -> DirectedKey {
    let v = if as_datetime {
        datetime_scalar(value)
    } else {
        value.clone()
    };
    DirectedKey {
        key: order_key(&v),
        desc,
    }
}

// ---------------------------------------------------------------------------
// Expression evaluator
// ---------------------------------------------------------------------------

use crate::ast::Expr;

/// Evaluate an expression against a row, returning a three-valued result.
///
/// Placeholders (`Placeholder` / `NamedParam`) are expected to be resolved into
/// literals before evaluation (the planner binds them); an unresolved one
/// evaluates to NULL. `Excluded` references are resolved by the writer before
/// evaluation and also yield NULL here.
pub fn eval_expr(expr: &Expr, row: &Row) -> EvalVal {
    match expr {
        Expr::Literal(v) => EvalVal::Val(v.clone()),
        Expr::Placeholder => EvalVal::Val(Value::Null),
        Expr::NamedParam(_) => EvalVal::Val(Value::Null),
        Expr::Excluded(_) => EvalVal::Val(Value::Null),
        // A dotted `table.col` reference matches a row cell exactly first (a
        // literally-named "a.b" column wins); otherwise it unqualifies to the
        // bare column name, since `Row` (single-table, no JOIN) stores cells
        // undotted.
        Expr::Column(name) => EvalVal::Val(match row.get(name) {
            Some(v) => v.clone(),
            None => match name.rsplit_once('.') {
                Some((_, col)) => row.get_or_null(col),
                None => Value::Null,
            },
        }),
        Expr::BinOp { op, left, right } => eval_binop(op, left, right, row),
        Expr::UnaryOp { op, operand } => {
            let v = eval_expr(operand, row);
            if op.eq_ignore_ascii_case("NOT") {
                EvalVal::Bool(sql_not(v.truth()))
            } else {
                EvalVal::Val(Value::Null)
            }
        }
        Expr::IsNull { operand, negated } => {
            let v = eval_expr(operand, row).into_value();
            let is_null = matches!(v, Value::Null);
            EvalVal::Bool(Some(if *negated { !is_null } else { is_null }))
        }
        Expr::InList {
            operand,
            items,
            text_set,
        } => eval_in_list(operand, items, text_set.as_deref(), row),
        Expr::Coalesce(args) => {
            for arg in args {
                let v = eval_expr(arg, row).into_value();
                if !matches!(v, Value::Null) {
                    return EvalVal::Val(v);
                }
            }
            EvalVal::Val(Value::Null)
        }
        Expr::FuncCall { name, args } => {
            if name == "datetime" && !args.is_empty() {
                let arg = eval_expr(&args[0], row).into_value();
                if args.len() == 1 {
                    // The single-argument form is bit-identical to the literal-ISO
                    // path the store/recall layer issues.
                    EvalVal::Val(datetime_scalar(&arg))
                } else {
                    // Trailing arguments are modifier strings (e.g. 'unixepoch').
                    let modifiers: Vec<String> = args[1..]
                        .iter()
                        .map(|a| match eval_expr(a, row).into_value() {
                            Value::Text(s) => s,
                            _ => String::new(),
                        })
                        .collect();
                    EvalVal::Val(datetime_with_modifiers(&arg, &modifiers))
                }
            } else {
                EvalVal::Val(Value::Null)
            }
        }
    }
}

fn eval_binop(op: &str, left: &Expr, right: &Expr, row: &Row) -> EvalVal {
    let upper = op.to_ascii_uppercase();
    match upper.as_str() {
        "AND" => {
            let l = eval_expr(left, row).truth();
            let r = eval_expr(right, row).truth();
            EvalVal::Bool(sql_and(l, r))
        }
        "OR" => {
            let l = eval_expr(left, row).truth();
            let r = eval_expr(right, row).truth();
            EvalVal::Bool(sql_or(l, r))
        }
        "LIKE" => {
            let l = eval_expr(left, row).into_value();
            let r = eval_expr(right, row).into_value();
            EvalVal::Bool(like_match(&l, &r))
        }
        "NOT LIKE" => {
            let l = eval_expr(left, row).into_value();
            let r = eval_expr(right, row).into_value();
            EvalVal::Bool(sql_not(like_match(&l, &r)))
        }
        _ => {
            let l = eval_expr(left, row).into_value();
            let r = eval_expr(right, row).into_value();
            // Column affinity: a numeric literal compared to a text-affinity column
            // takes TEXT affinity (no numeric coercion of the column's text). The
            // column's stored storage class is the affinity signal, because write
            // affinity already normalized each stored cell to its declared
            // affinity's class (a numeric-looking string in an INTEGER column was
            // stored as the integer). So a text column value with a numeric literal
            // is compared as text, matching sqlite3's `text_col = 5` -> FALSE for
            // a '05' cell, while the numeric-numeric and text-text paths are
            // untouched.
            let (l, r) = apply_literal_text_affinity(left, right, l, r);
            EvalVal::Bool(sql_compare(&upper, l, r))
        }
    }
}

/// When a column whose stored value is text is compared to a numeric literal,
/// render the literal as its sqlite3 text form so the comparison runs under TEXT
/// affinity (no numeric coercion of the column's text). Only the column-vs-literal
/// shape with a text column value and a numeric literal is rewritten; every other
/// pairing is returned unchanged, leaving the numeric-numeric and text-text paths
/// exactly as they were.
fn apply_literal_text_affinity(left: &Expr, right: &Expr, l: Value, r: Value) -> (Value, Value) {
    let is_col = |e: &Expr| matches!(e, Expr::Column(_));
    let is_lit = |e: &Expr| matches!(e, Expr::Literal(_));
    // Column on the left, numeric literal on the right.
    if is_col(left) && is_lit(right) {
        if let (Value::Text(_), num @ (Value::Int(_) | Value::Float(_))) = (&l, &r) {
            return (l, Value::Text(numeric_text(num)));
        }
    }
    // Numeric literal on the left, column on the right.
    if is_lit(left) && is_col(right) {
        if let (num @ (Value::Int(_) | Value::Float(_)), Value::Text(_)) = (&l, &r) {
            return (Value::Text(numeric_text(num)), r);
        }
    }
    (l, r)
}

/// The sqlite3 text rendering of a numeric value (the form a numeric literal takes
/// under TEXT affinity).
fn numeric_text(v: &Value) -> String {
    match v {
        Value::Int(i) => i.to_string(),
        Value::Float(f) => crate::catalog::render_float_text(*f),
        _ => String::new(),
    }
}

fn eval_in_list(
    operand: &Expr,
    items: &[Expr],
    text_set: Option<&std::collections::HashSet<String>>,
    row: &Row,
) -> EvalVal {
    let lhs = eval_expr(operand, row).into_value();
    if matches!(lhs, Value::Null) {
        return EvalVal::Bool(None);
    }
    // O(1) membership fast path: the binder attaches `text_set` ONLY when
    // every item is a TEXT literal, and TEXT=TEXT comparison is byte
    // equality — so for a TEXT operand the set probe is semantically
    // identical to the linear scan below (and no item can be NULL, so a
    // miss is a definite false). A non-TEXT operand needs the coercing
    // comparison and falls through to the linear path.
    if let (Some(set), Value::Text(s)) = (text_set, &lhs) {
        return EvalVal::Bool(Some(set.contains(s)));
    }
    let mut found_null = false;
    for item in items {
        let iv = eval_expr(item, row).into_value();
        if matches!(iv, Value::Null) {
            found_null = true;
            continue;
        }
        if sql_compare("=", lhs.clone(), iv) == Some(true) {
            return EvalVal::Bool(Some(true));
        }
    }
    EvalVal::Bool(if found_null { None } else { Some(false) })
}

/// Evaluate a WHERE predicate to its three-valued truth for `filter_op`.
///
/// A row passes the filter only when this is `Some(true)`; `Some(false)` and
/// `None` (NULL / unknown) both reject the row, matching the reference's
/// `result is not SqlNull and result` test.
pub fn eval_predicate(expr: &Expr, row: &Row) -> Option<bool> {
    eval_expr(expr, row).truth()
}

/// Evaluate an expression to a scalar value (collapsing a boolean result to
/// `Int(1)`/`Int(0)`/`Null`), used by aliased / COALESCE projection.
pub fn eval_scalar(expr: &Expr, row: &Row) -> Value {
    eval_expr(expr, row).into_value()
}
