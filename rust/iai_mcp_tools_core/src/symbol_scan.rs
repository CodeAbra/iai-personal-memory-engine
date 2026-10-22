//! Hand-rolled single-line definition scan (no regex, no parser) across
//! Python, Rust, and TypeScript/JS.
//!
//! A single-line scan cannot see comments, strings, or docstrings; the
//! definition keyword must be the first non-whitespace token on the line
//! (after stripping a small fixed per-language modifier-prefix list), which
//! bounds trailing-comment false positives but not full-line comments or
//! docstrings that themselves start with a definition keyword. Accepted
//! under the no-parser constraint.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolKind {
    Def,
    Unknown,
}

/// Selects which per-language matcher `scan_file_for_language` applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    Python,
    Rust,
    TsJs,
}

pub struct SymbolCandidate {
    pub name: String,
    #[allow(dead_code)] // round-tripped for a later plan's diagnostics; unread by this slice
    pub line: u32,
    pub kind: SymbolKind,
}

/// Rust visibility/async modifiers stripped before the keyword check, in a
/// fixed order tried every loop iteration -- handles `pub async fn` (two
/// strips) without needing to special-case modifier ordering.
const RUST_MODIFIERS: &[&str] = &["pub(crate) ", "pub ", "async "];

/// TS/JS export modifiers stripped before the keyword check. Longest-first
/// so `export default ` is consumed whole rather than leaving a dangling
/// `default `.
const TSJS_MODIFIERS: &[&str] = &["export default ", "export "];

/// Repeatedly strips any matching prefix from `modifiers` until none match.
/// Bounded: each successful strip consumes at least one character, so the
/// loop terminates within `line.len()` iterations -- no backtracking.
fn strip_modifiers<'a>(mut s: &'a str, modifiers: &[&str]) -> &'a str {
    loop {
        let mut stripped_any = false;
        for m in modifiers {
            if let Some(rest) = s.strip_prefix(m) {
                s = rest;
                stripped_any = true;
                break;
            }
        }
        if !stripped_any {
            return s;
        }
    }
}

/// Scans forward from the start of `s` for an `[A-Za-z_][A-Za-z0-9_]*`
/// identifier, skipping leading whitespace. Self-contained -- does not
/// reach into `tokenize.rs`'s private `find_raw_tokens`.
fn next_identifier(s: &str) -> Option<String> {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n && chars[i].is_whitespace() {
        i += 1;
    }
    if i >= n || !(chars[i].is_ascii_alphabetic() || chars[i] == '_') {
        return None;
    }
    let start = i;
    i += 1;
    while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
        i += 1;
    }
    Some(chars[start..i].iter().collect())
}

/// An `[A-Za-z_][A-Za-z0-9_]*` identifier anchored at byte offset 0 of `s`
/// (no leading-whitespace skip, unlike `next_identifier`) -- used by the
/// arrow-field signal to require the identifier be the line's first token.
/// Returns the identifier and its byte length (ASCII-only identifier
/// characters, so byte length equals char count).
fn leading_identifier(s: &str) -> Option<(String, usize)> {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() || !(chars[0].is_ascii_alphabetic() || chars[0] == '_') {
        return None;
    }
    let mut i = 1;
    while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
        i += 1;
    }
    let name: String = chars[..i].iter().collect();
    let byte_len = name.len();
    Some((name, byte_len))
}

/// Every `[A-Za-z_][A-Za-z0-9_]*` identifier run anywhere in `line`.
fn extract_all_identifiers(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if chars[i].is_ascii_alphabetic() || chars[i] == '_' {
            let start = i;
            i += 1;
            while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else {
            i += 1;
        }
    }
    out
}

/// Object/class-field arrow-method signal: `name: (...) => ...` with no
/// `const`/`function`/`class` prefix. Bounded to the same-line
/// colon-then-arrow shape; a colon-bearing construct unrelated to a field
/// definition that happens to also contain `=>` later on the line is an
/// accepted false positive under the no-parser constraint.
fn arrow_field_def(line: &str) -> Option<String> {
    let (name, len) = leading_identifier(line)?;
    let rest = line[len..].trim_start();
    let after_colon = rest.strip_prefix(':')?;
    if after_colon.contains("=>") {
        Some(name)
    } else {
        None
    }
}

/// Scans one Python line: a line whose first non-whitespace token is
/// `def ` or `class ` yields ONE `Def` candidate (the identifier following
/// the keyword); every extracted identifier on any other line is
/// `Unknown`. A full-line comment or docstring beginning with the keyword
/// is an accepted false positive (documented at module level, not fixed).
fn scan_python_line(line: &str, line_no: u32, out: &mut Vec<SymbolCandidate>) {
    let trimmed = line.trim_start();
    for kw in ["def ", "class "] {
        if let Some(rest) = trimmed.strip_prefix(kw) {
            if let Some(name) = next_identifier(rest) {
                out.push(SymbolCandidate {
                    name,
                    line: line_no,
                    kind: SymbolKind::Def,
                });
            }
            return;
        }
    }
    for name in extract_all_identifiers(line) {
        out.push(SymbolCandidate {
            name,
            line: line_no,
            kind: SymbolKind::Unknown,
        });
    }
}

/// Scans one Rust line: modifiers (`pub`, `pub(crate)`, `async`) are
/// stripped before the keyword check, so `pub fn f`, `pub(crate) struct S`,
/// and `pub async fn f` all tag Def just like their bare forms.
fn scan_rust_line(line: &str, line_no: u32, out: &mut Vec<SymbolCandidate>) {
    let trimmed = line.trim_start();
    let stripped = strip_modifiers(trimmed, RUST_MODIFIERS);
    for kw in ["fn ", "struct ", "enum ", "trait ", "impl "] {
        if let Some(rest) = stripped.strip_prefix(kw) {
            if let Some(name) = next_identifier(rest) {
                out.push(SymbolCandidate {
                    name,
                    line: line_no,
                    kind: SymbolKind::Def,
                });
            }
            return;
        }
    }
    for name in extract_all_identifiers(line) {
        out.push(SymbolCandidate {
            name,
            line: line_no,
            kind: SymbolKind::Unknown,
        });
    }
}

/// Scans one TypeScript/JS line: `export`/`export default` are stripped
/// before the keyword check. `const` alone tags Def on the bound
/// identifier regardless of whether the bound value is function-valued
/// (pins the `const X = 5` / `const f = () => {}` reading); the arrow
/// signal is a separate trigger for an object/class-field arrow method
/// that has no `const` prefix (`foo: () => {}`).
fn scan_tsjs_line(line: &str, line_no: u32, out: &mut Vec<SymbolCandidate>) {
    let trimmed = line.trim_start();
    let stripped = strip_modifiers(trimmed, TSJS_MODIFIERS);

    for kw in ["class ", "function "] {
        if let Some(rest) = stripped.strip_prefix(kw) {
            if let Some(name) = next_identifier(rest) {
                out.push(SymbolCandidate {
                    name,
                    line: line_no,
                    kind: SymbolKind::Def,
                });
            }
            return;
        }
    }
    if let Some(rest) = stripped.strip_prefix("const ") {
        if let Some(name) = next_identifier(rest) {
            out.push(SymbolCandidate {
                name,
                line: line_no,
                kind: SymbolKind::Def,
            });
        }
        return;
    }
    if let Some(name) = arrow_field_def(stripped) {
        out.push(SymbolCandidate {
            name,
            line: line_no,
            kind: SymbolKind::Def,
        });
        return;
    }

    for name in extract_all_identifiers(line) {
        out.push(SymbolCandidate {
            name,
            line: line_no,
            kind: SymbolKind::Unknown,
        });
    }
}

/// Scans every line of `text` under the given language's matcher, returning
/// every candidate found in file order. Bounded, linear single pass -- no
/// unbounded buffering beyond the already-read file text.
pub fn scan_file_for_language(text: &str, lang: Language) -> Vec<SymbolCandidate> {
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx as u32;
        match lang {
            Language::Python => scan_python_line(line, line_no, &mut out),
            Language::Rust => scan_rust_line(line, line_no, &mut out),
            Language::TsJs => scan_tsjs_line(line, line_no, &mut out),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def_names(cands: &[SymbolCandidate]) -> Vec<&str> {
        cands
            .iter()
            .filter(|c| c.kind == SymbolKind::Def)
            .map(|c| c.name.as_str())
            .collect()
    }

    #[test]
    fn def_line_yields_a_def_candidate() {
        let cands = scan_file_for_language(
            "def zebracorn_finder():\n    return 1\n",
            Language::Python,
        );
        let def = cands
            .iter()
            .find(|c| c.kind == SymbolKind::Def)
            .expect("expected one Def candidate");
        assert_eq!(def.name, "zebracorn_finder");
        assert_eq!(def.line, 0);
    }

    #[test]
    fn class_line_yields_a_def_candidate() {
        let cands = scan_file_for_language("class Zebracorn:\n    pass\n", Language::Python);
        let def = cands
            .iter()
            .find(|c| c.kind == SymbolKind::Def)
            .expect("expected one Def candidate");
        assert_eq!(def.name, "Zebracorn");
    }

    #[test]
    fn mention_only_line_yields_unknown_candidates() {
        let cands = scan_file_for_language("result = zebracorn_finder()\n", Language::Python);
        assert!(cands
            .iter()
            .any(|c| c.name == "zebracorn_finder" && c.kind == SymbolKind::Unknown));
        assert!(!cands
            .iter()
            .any(|c| c.name == "zebracorn_finder" && c.kind == SymbolKind::Def));
    }

    #[test]
    fn keyword_not_first_token_is_not_a_def() {
        // Trailing-comment-style false positive is bounded: the keyword
        // must be the FIRST non-whitespace token on the line.
        let cands = scan_file_for_language(
            "x = 1  # def not_a_real_def():\n",
            Language::Python,
        );
        assert!(!cands
            .iter()
            .any(|c| c.name == "not_a_real_def" && c.kind == SymbolKind::Def));
    }

    #[test]
    fn rust_fn_struct_enum_trait_impl_yield_def_candidates() {
        let text = "fn alpha_marker() {}\n\
                     struct BetaHolder;\n\
                     enum GammaChoice {}\n\
                     trait DeltaBehavior {}\n\
                     impl EpsilonTarget {}\n";
        let cands = scan_file_for_language(text, Language::Rust);
        let names = def_names(&cands);
        assert!(names.contains(&"alpha_marker"));
        assert!(names.contains(&"BetaHolder"));
        assert!(names.contains(&"GammaChoice"));
        assert!(names.contains(&"DeltaBehavior"));
        assert!(names.contains(&"EpsilonTarget"));
    }

    #[test]
    fn rust_pub_fn_after_modifier_strip_yields_def() {
        let cands = scan_file_for_language("pub fn zebracorn_open() {}\n", Language::Rust);
        assert!(def_names(&cands).contains(&"zebracorn_open"));
    }

    #[test]
    fn rust_pub_crate_struct_after_modifier_strip_yields_def() {
        let cands =
            scan_file_for_language("pub(crate) struct ZebracornStore;\n", Language::Rust);
        assert!(def_names(&cands).contains(&"ZebracornStore"));
    }

    #[test]
    fn rust_pub_async_fn_after_modifier_strip_yields_def() {
        let cands = scan_file_for_language("pub async fn zebracorn_fetch() {}\n", Language::Rust);
        assert!(def_names(&cands).contains(&"zebracorn_fetch"));
    }

    #[test]
    fn rust_mention_only_line_stays_unknown() {
        let cands = scan_file_for_language("let x = zebracorn_open();\n", Language::Rust);
        assert!(!def_names(&cands).contains(&"zebracorn_open"));
        assert!(cands
            .iter()
            .any(|c| c.name == "zebracorn_open" && c.kind == SymbolKind::Unknown));
    }

    #[test]
    fn tsjs_function_and_export_function_yield_def_candidates() {
        let cands = scan_file_for_language("function zebracornMake() {}\n", Language::TsJs);
        assert!(def_names(&cands).contains(&"zebracornMake"));

        let cands = scan_file_for_language("export function zebracornMake() {}\n", Language::TsJs);
        assert!(def_names(&cands).contains(&"zebracornMake"));
    }

    #[test]
    fn tsjs_class_and_export_default_class_yield_def_candidates() {
        let cands = scan_file_for_language("class ZebracornWidget {}\n", Language::TsJs);
        assert!(def_names(&cands).contains(&"ZebracornWidget"));

        let cands =
            scan_file_for_language("export default class ZebracornWidget {}\n", Language::TsJs);
        assert!(def_names(&cands).contains(&"ZebracornWidget"));
    }

    #[test]
    fn tsjs_const_binding_tags_def_on_bound_identifier() {
        // Pinned: `const X = 5` tags Def(X) -- const alone marks any
        // top-level const binding, function-valued or not.
        let cands = scan_file_for_language("const zebracornCount = 5;\n", Language::TsJs);
        assert!(def_names(&cands).contains(&"zebracornCount"));
    }

    #[test]
    fn tsjs_const_arrow_function_tags_def_on_bound_identifier() {
        // Pinned: `const f = () => {}` tags Def(f) via the const branch,
        // not the separate arrow-field signal.
        let cands = scan_file_for_language("const zebracornRun = () => {};\n", Language::TsJs);
        assert!(def_names(&cands).contains(&"zebracornRun"));
    }

    #[test]
    fn tsjs_object_field_arrow_method_tags_def_via_arrow_signal() {
        // Pinned: `foo: () => {}` (no const prefix) tags Def(foo) via the
        // arrow signal -- the object/class-field arrow-method shape.
        let cands = scan_file_for_language("zebracornHandler: () => {},\n", Language::TsJs);
        assert!(def_names(&cands).contains(&"zebracornHandler"));
    }

    #[test]
    fn tsjs_line_without_definition_keyword_stays_unknown() {
        let cands =
            scan_file_for_language("result = zebracornMake();\n", Language::TsJs);
        assert!(!def_names(&cands).contains(&"zebracornMake"));
        assert!(cands
            .iter()
            .any(|c| c.name == "zebracornMake" && c.kind == SymbolKind::Unknown));
    }
}
