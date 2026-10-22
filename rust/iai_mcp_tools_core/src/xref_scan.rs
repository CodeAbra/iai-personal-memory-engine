//! Hand-rolled single-line reference/call-site scan (no regex, no parser)
//! across Python, Rust, and TypeScript/JS -- mirrors symbol_scan.rs's
//! per-language dispatch, modifier-stripping, and never-panic posture.
//! Recognized shapes: Python `from <module> import <name>` and plain
//! `import <module>`; Rust `use <path>::<name>`; TS/JS
//! `import { <name> } from '<module>'`; a bare `name(` call site in any
//! of the three. A line missing every recognized shape yields no
//! candidate. A recognized shape whose target cannot be pinned to a
//! single name (a wildcard import, an empty braced import list) yields an
//! `Unresolved` candidate carrying the raw text instead of being dropped
//! -- resolution against other files happens one layer up, in
//! xref_segment.rs; this scanner only decides whether IT could extract a
//! clean name.

use std::path::{Path, PathBuf};

use crate::symbol_scan::Language;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XrefKind {
    Import,
    Call,
    Unresolved,
}

pub struct XrefCandidate {
    pub module: Option<String>,
    pub name: String,
    pub kind: XrefKind,
    /// True only for a Python plain `import <module>` -- the whole module
    /// is the reference target, so resolution (one layer up) must not
    /// require `name` to be independently confirmed as a definition
    /// inside the target file.
    pub whole_module: bool,
    #[allow(dead_code)] // round-tripped for future diagnostics; no current consumer reads it
    pub line: u32,
}

fn candidate(
    module: Option<String>,
    name: String,
    kind: XrefKind,
    whole_module: bool,
    line: u32,
) -> XrefCandidate {
    XrefCandidate {
        module,
        name,
        kind,
        whole_module,
        line,
    }
}

const IMPORT_MARKER: &str = " import ";

/// Control-flow keywords that precede `(` in ordinary source but are
/// never a reference/call target -- shared across all three languages
/// since none of them use these words as callable identifiers.
const CALL_DENY: &[&str] = &["if", "while", "for", "return", "switch", "catch", "match", "elif"];

/// Scans forward from the start of `s` for an `[A-Za-z_][A-Za-z0-9_]*`
/// identifier, skipping leading whitespace. Self-contained -- mirrors
/// symbol_scan.rs's own `next_identifier` by shape rather than reaching
/// into it (that helper is private to its module).
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

/// Rust visibility/async modifiers stripped before the `use`/def keyword
/// check -- mirrors symbol_scan.rs's own `RUST_MODIFIERS` list (private
/// there, copied here by shape).
const RUST_MODIFIERS: &[&str] = &["pub(crate) ", "pub ", "async "];
/// TS/JS export modifiers, longest-first so `export default ` is consumed
/// whole -- mirrors symbol_scan.rs's own `TSJS_MODIFIERS`.
const TSJS_MODIFIERS: &[&str] = &["export default ", "export "];

/// Repeatedly strips any matching prefix from `modifiers` until none
/// match. Bounded: each successful strip consumes at least one character,
/// so the loop terminates within `line.len()` iterations -- no
/// backtracking. Mirrors symbol_scan.rs's own `strip_modifiers`.
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

fn is_python_def_line(trimmed: &str) -> bool {
    trimmed.starts_with("def ") || trimmed.starts_with("class ")
}

fn is_rust_def_line(stripped: &str) -> bool {
    ["fn ", "struct ", "enum ", "trait ", "impl "]
        .iter()
        .any(|kw| stripped.starts_with(kw))
}

/// Only `class `/`function ` risk a false self-call (the def name sits
/// immediately before `(`); a `const ` binding's own name is always
/// followed by `=`, never `(`, so it needs no exclusion and its RHS call
/// sites (`const x = helper();`) must stay scannable.
fn is_tsjs_def_line(stripped: &str) -> bool {
    ["class ", "function "].iter().any(|kw| stripped.starts_with(kw))
}

/// Every bare `name(` call site on `line` (optional spaces between the
/// identifier and the paren, a bounded scan, no backtracking): skips a
/// deny-listed control-flow keyword and skips a name immediately preceded
/// by `.` (a member call on some receiver -- resolution covers only plain
/// top-level names, not member calls).
fn scan_call_sites(line: &str, line_no: u32, out: &mut Vec<XrefCandidate>) {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if chars[i].is_ascii_alphabetic() || chars[i] == '_' {
            let start = i;
            i += 1;
            while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let name: String = chars[start..i].iter().collect();

            let preceded_by_dot = start > 0 && chars[start - 1] == '.';
            let mut j = i;
            while j < n && chars[j] == ' ' {
                j += 1;
            }
            let is_call = j < n && chars[j] == '(';

            if is_call && !preceded_by_dot && !CALL_DENY.contains(&name.as_str()) {
                out.push(candidate(None, name, XrefKind::Call, false, line_no));
            }
        } else {
            i += 1;
        }
    }
}

/// Scans one line for `from <module> import <name>`: the module token is
/// every byte between `from ` and ` import ` (dots kept verbatim, so a
/// relative import's leading dots survive into `module`); the name is
/// the first identifier after ` import ` (a multi-name or aliased import
/// list yields only its first name -- an accepted narrowing). A wildcard
/// (`from <module> import *`) yields an `Unresolved` candidate carrying
/// the raw `*` rather than being dropped.
fn scan_python_from_import(after_from: &str, line_no: u32, out: &mut Vec<XrefCandidate>) {
    let Some(marker_at) = after_from.find(IMPORT_MARKER) else {
        return;
    };
    let module = after_from[..marker_at].trim();
    if module.is_empty() {
        return;
    }
    let after_import = &after_from[marker_at + IMPORT_MARKER.len()..];
    let raw = after_import.trim();
    if raw.is_empty() {
        return;
    }
    match next_identifier(after_import) {
        Some(name) => out.push(candidate(Some(module.to_string()), name, XrefKind::Import, false, line_no)),
        None => out.push(candidate(
            Some(module.to_string()),
            raw.to_string(),
            XrefKind::Unresolved,
            false,
            line_no,
        )),
    }
}

/// Scans one line for a plain `import <module>` (no `from`): the whole
/// module is the reference target (`whole_module = true`), so resolution
/// only needs to find a matching file, never a specific name defined in
/// it. Dotted module paths (`import a.b.c`) resolve on their first
/// identifier segment -- an accepted narrowing.
fn scan_python_plain_import(after_import: &str, line_no: u32, out: &mut Vec<XrefCandidate>) {
    let Some(name) = next_identifier(after_import) else {
        return;
    };
    out.push(candidate(Some(name.clone()), name, XrefKind::Import, true, line_no));
}

fn scan_python_line(line: &str, line_no: u32, out: &mut Vec<XrefCandidate>) {
    let trimmed = line.trim_start();
    if let Some(after_from) = trimmed.strip_prefix("from ") {
        scan_python_from_import(after_from, line_no, out);
        return;
    }
    if let Some(after_import) = trimmed.strip_prefix("import ") {
        scan_python_plain_import(after_import, line_no, out);
        return;
    }
    if is_python_def_line(trimmed) {
        return;
    }
    scan_call_sites(line, line_no, out);
}

/// Scans one Rust `use <path>;` line (after visibility/async modifiers
/// are stripped): the target name is the final `::`-separated path
/// segment, `module` is everything before it. A glob (`use path::*;`) or
/// an empty/malformed braced group (`use path::{};`) yields an
/// `Unresolved` candidate carrying the raw tail; a braced group
/// (`use path::{A, B};`) yields only its first name -- an accepted
/// narrowing, mirroring the Python multi-name-import reading.
fn scan_rust_use(rest: &str, line_no: u32, out: &mut Vec<XrefCandidate>) {
    let rest = rest.trim_end();
    let rest = rest.strip_suffix(';').unwrap_or(rest).trim_end();
    if rest.is_empty() {
        return;
    }

    let (module_part, tail) = match rest.rfind("::") {
        Some(pos) => (&rest[..pos], &rest[pos + 2..]),
        None => ("", rest),
    };
    let module = if module_part.is_empty() {
        None
    } else {
        Some(module_part.to_string())
    };

    if let Some(inner) = tail.strip_prefix('{') {
        let inner = inner.strip_suffix('}').unwrap_or(inner);
        match next_identifier(inner) {
            Some(name) => out.push(candidate(module, name, XrefKind::Import, false, line_no)),
            None => out.push(candidate(
                module,
                tail.to_string(),
                XrefKind::Unresolved,
                false,
                line_no,
            )),
        }
        return;
    }

    match next_identifier(tail) {
        Some(name) if name == tail => {
            out.push(candidate(module, name, XrefKind::Import, false, line_no))
        }
        _ => out.push(candidate(
            module,
            tail.to_string(),
            XrefKind::Unresolved,
            false,
            line_no,
        )),
    }
}

fn scan_rust_line(line: &str, line_no: u32, out: &mut Vec<XrefCandidate>) {
    let trimmed = line.trim_start();
    let stripped = strip_modifiers(trimmed, RUST_MODIFIERS);
    if let Some(rest) = stripped.strip_prefix("use ") {
        scan_rust_use(rest, line_no, out);
        return;
    }
    if is_rust_def_line(stripped) {
        return;
    }
    scan_call_sites(line, line_no, out);
}

/// Extracts a `'...'`- or `"..."`-quoted module path anchored at the
/// start of `s`. Returns `None` (never panics) when `s` does not open on
/// a quote or the quote is unterminated.
fn extract_quoted_module(s: &str) -> Option<String> {
    let mut chars = s.chars();
    let quote = chars.next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &s[quote.len_utf8()..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

/// Scans one TS/JS `import { <name> } from '<module>';` line: only the
/// named-import shape (braced) is recognized -- a default or namespace
/// import (`import Foo from '...'`, `import * as ns from '...'`) yields
/// no candidate, out of scope. An empty or malformed braced list yields
/// an `Unresolved` candidate.
fn scan_tsjs_named_import(rest: &str, line_no: u32, out: &mut Vec<XrefCandidate>) {
    let rest = rest.trim_start();
    let Some(after_brace) = rest.strip_prefix('{') else {
        return;
    };
    let Some(close) = after_brace.find('}') else {
        return;
    };
    let names_part = &after_brace[..close];
    let after_close = &after_brace[close + 1..];
    let Some(from_at) = after_close.find("from") else {
        return;
    };
    let after_from = after_close[from_at + "from".len()..].trim_start();
    let module = extract_quoted_module(after_from);

    match next_identifier(names_part) {
        Some(name) => out.push(candidate(module, name, XrefKind::Import, false, line_no)),
        None => out.push(candidate(
            module,
            names_part.trim().to_string(),
            XrefKind::Unresolved,
            false,
            line_no,
        )),
    }
}

fn scan_tsjs_line(line: &str, line_no: u32, out: &mut Vec<XrefCandidate>) {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix("import ") {
        scan_tsjs_named_import(rest, line_no, out);
        return;
    }
    let stripped = strip_modifiers(trimmed, TSJS_MODIFIERS);
    if is_tsjs_def_line(stripped) {
        return;
    }
    scan_call_sites(line, line_no, out);
}

/// Scans every line of `text` under the given language's matcher,
/// returning every candidate found in file order. Bounded, linear single
/// pass -- no unbounded buffering beyond the already-read file text; no
/// input shape panics.
pub fn scan_file_for_language(text: &str, lang: Language) -> Vec<XrefCandidate> {
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

/// Extensions the resolver accepts as a candidate file for the given
/// language -- mirrors `xref_segment.rs`'s own `language_for_path`
/// mapping BY VALUE (that function is private to its module) so a
/// resolved candidate is only matched against extensions this scanner
/// would itself recognize as belonging to that language.
pub fn candidate_extensions(lang: Language) -> &'static [&'static str] {
    match lang {
        Language::Python => &["py"],
        Language::Rust => &["rs"],
        Language::TsJs => &["ts", "tsx", "js", "jsx", "mjs", "cjs"],
    }
}

/// Outcome of resolving a module path's DIRECTORY-RELATIVE shape (a
/// Python relative-dot import, a Rust `self`/`super`/`crate` path, or a
/// TS/JS `./`/`../` specifier) to a candidate file location, honoring the
/// corpus-root containment guard.
#[derive(Debug, PartialEq, Eq)]
pub enum RelativeResolution {
    /// The module path carries no directory-relative marker this
    /// resolver understands (an absolute dotted Python module, a bare
    /// Rust path with no `self`/`super`/`crate` marker, or a bare TS/JS
    /// package specifier) -- resolution falls back to a whole-batch stem
    /// match.
    NotRelative,
    /// The module path's relative markers would climb above
    /// `corpus_root` -- kept unresolved, never dereferenced outside the
    /// corpus. A crafted escaping import must always land here.
    Escaped,
    /// The extension-less candidate file path this module path names,
    /// relative to the importing file's own directory (or the corpus
    /// root, for Rust's `crate::`).
    Resolved(PathBuf),
}

struct RelativeAnchor {
    /// True only for Rust's `crate::` -- anchored at the corpus root
    /// rather than the importing file's own directory.
    from_corpus_root: bool,
    climb: u32,
    segments: Vec<String>,
}

/// Python: leading dots each count as one climb level past the first
/// (`.mod` = same package, `..mod` = parent package, ...); `dots == 0`
/// is an absolute dotted module with no directory marker.
fn python_relative_anchor(module: &str) -> Option<RelativeAnchor> {
    let dots = module.chars().take_while(|&c| c == '.').count();
    if dots == 0 {
        return None;
    }
    let tail = &module[dots..];
    let segments: Vec<String> = tail
        .split('.')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if segments.is_empty() {
        return None; // "from . import x" -- package-boundary import, not resolved
    }
    Some(RelativeAnchor {
        from_corpus_root: false,
        climb: (dots - 1) as u32,
        segments,
    })
}

/// Rust: `crate::` anchors at the corpus root; each leading `super::`
/// climbs one level; a leading `self::` is a same-directory no-op. A
/// path with none of these markers (an external or sibling-module path)
/// yields `None` -- resolved, if at all, by a whole-batch stem match.
fn rust_relative_anchor(module: &str) -> Option<RelativeAnchor> {
    if module.is_empty() {
        return None;
    }
    let segs: Vec<&str> = module.split("::").collect();
    if segs.first() == Some(&"crate") {
        let remaining: Vec<String> = segs[1..].iter().map(|s| s.to_string()).collect();
        if remaining.is_empty() {
            return None;
        }
        return Some(RelativeAnchor {
            from_corpus_root: true,
            climb: 0,
            segments: remaining,
        });
    }
    let mut i = 0;
    let mut climb = 0u32;
    while segs.get(i) == Some(&"super") {
        climb += 1;
        i += 1;
    }
    if segs.get(i) == Some(&"self") {
        i += 1;
    }
    if i == 0 {
        return None; // no self/super/crate marker -- external/sibling path
    }
    let remaining: Vec<String> = segs[i..].iter().map(|s| s.to_string()).collect();
    if remaining.is_empty() {
        return None;
    }
    Some(RelativeAnchor {
        from_corpus_root: false,
        climb,
        segments: remaining,
    })
}

/// TS/JS: a `./`/`../`-prefixed quoted specifier is a literal relative
/// path -- each leading `..` segment climbs one level. A bare (non-
/// relative) specifier is an npm package name, never a local file.
fn tsjs_relative_anchor(module: &str) -> Option<RelativeAnchor> {
    if !(module.starts_with("./") || module.starts_with("../")) {
        return None;
    }
    let mut climb = 0u32;
    let mut segments: Vec<String> = Vec::new();
    for part in module.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if !segments.is_empty() {
                    // ".." reappearing after a real segment -- an unusual
                    // mid-path shape not interpreted here; bail rather
                    // than guess.
                    return None;
                }
                climb += 1;
            }
            other => segments.push(other.to_string()),
        }
    }
    if segments.is_empty() {
        return None;
    }
    Some(RelativeAnchor {
        from_corpus_root: false,
        climb,
        segments,
    })
}

/// Resolves `module`'s directory-relative shape (per `lang`'s own
/// convention) to a candidate extension-less file path, applying the
/// corpus-root containment guard BEFORE ever returning a path: an import
/// whose relative markers would climb above `corpus_root` comes back
/// `Escaped`, never a path outside the corpus. `from_path` is the
/// importing file itself -- its own directory is the resolution anchor
/// for every shape except Rust's `crate::`, which anchors at
/// `corpus_root` directly.
pub fn resolve_relative_module(
    module: &str,
    lang: Language,
    from_path: &Path,
    corpus_root: &Path,
) -> RelativeResolution {
    let anchor = match lang {
        Language::Python => python_relative_anchor(module),
        Language::Rust => rust_relative_anchor(module),
        Language::TsJs => tsjs_relative_anchor(module),
    };
    let Some(anchor) = anchor else {
        return RelativeResolution::NotRelative;
    };

    let start = if anchor.from_corpus_root {
        Some(corpus_root.to_path_buf())
    } else {
        from_path.parent().map(Path::to_path_buf)
    };
    let Some(mut dir) = start else {
        return RelativeResolution::Escaped;
    };

    for _ in 0..anchor.climb {
        let Some(parent) = dir.parent() else {
            return RelativeResolution::Escaped;
        };
        dir = parent.to_path_buf();
    }

    // Containment guard: the climbed anchor directory must stay at or
    // below the corpus root -- a crafted import that climbs past it is
    // rejected here, before ever building a candidate file path.
    if !dir.starts_with(corpus_root) {
        return RelativeResolution::Escaped;
    }

    let suffix: PathBuf = anchor.segments.iter().collect();
    let resolved = dir.join(&suffix);

    // Belt-and-braces re-check on the full resolved path rather than
    // trusting the anchor check alone.
    if !resolved.starts_with(corpus_root) {
        return RelativeResolution::Escaped;
    }

    RelativeResolution::Resolved(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_python(text: &str) -> Vec<XrefCandidate> {
        scan_file_for_language(text, Language::Python)
    }

    #[test]
    fn from_import_line_yields_module_and_name() {
        let cands = scan_python("from zebracorn_module import zebracorn_helper\n");
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].module.as_deref(), Some("zebracorn_module"));
        assert_eq!(cands[0].name, "zebracorn_helper");
        assert_eq!(cands[0].kind, XrefKind::Import);
        assert!(!cands[0].whole_module);
        assert_eq!(cands[0].line, 0);
    }

    #[test]
    fn relative_import_keeps_leading_dots_in_module() {
        let cands = scan_python("from .zebracorn_module import zebracorn_helper\n");
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].module.as_deref(), Some(".zebracorn_module"));
    }

    #[test]
    fn multi_name_import_yields_only_the_first_name() {
        let cands = scan_python("from zebracorn_module import alpha_name, beta_name\n");
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].name, "alpha_name");
    }

    #[test]
    fn wildcard_import_is_kept_unresolved_with_raw_text() {
        let cands = scan_python("from zebracorn_module import *\n");
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Unresolved);
        assert_eq!(cands[0].name, "*");
        assert_eq!(cands[0].module.as_deref(), Some("zebracorn_module"));
    }

    #[test]
    fn plain_import_statement_yields_a_whole_module_reference() {
        let cands = scan_python("import zebracorn_module\n");
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Import);
        assert!(cands[0].whole_module);
        assert_eq!(cands[0].name, "zebracorn_module");
        assert_eq!(cands[0].module.as_deref(), Some("zebracorn_module"));
    }

    #[test]
    fn mention_only_line_yields_a_call_reference() {
        let cands = scan_python("result = zebracorn_helper()\n");
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Call);
        assert_eq!(cands[0].name, "zebracorn_helper");
    }

    #[test]
    fn def_line_does_not_self_reference_as_a_call() {
        let cands = scan_python("def zebracorn_helper():\n    return 1\n");
        assert!(cands.is_empty());
    }

    #[test]
    fn member_call_is_not_a_bare_call_candidate() {
        let cands = scan_python("result = self.zebracorn_helper()\n");
        assert!(!cands.iter().any(|c| c.name == "zebracorn_helper"));
    }

    #[test]
    fn control_flow_keyword_is_never_a_call_candidate() {
        let cands = scan_python("if (zebracorn_ready):\n");
        assert!(!cands.iter().any(|c| c.name == "if"));
    }

    #[test]
    fn from_keyword_not_first_token_yields_no_candidate() {
        let cands = scan_python("x = 1  # from zebracorn_module import zebracorn_helper\n");
        assert!(cands.is_empty());
    }

    #[test]
    fn from_without_import_marker_yields_no_candidate() {
        let cands = scan_python("from zebracorn_module\n");
        assert!(cands.is_empty());
    }

    #[test]
    fn scan_never_panics_on_malformed_from_line() {
        let cands = scan_python("from    import\n");
        assert!(cands.is_empty());
    }

    #[test]
    fn scan_never_panics_on_adversarial_unicode_and_symbols() {
        let lines = [
            "from  import ***\n",
            "import \n",
            "use ::::;\n",
            "import { } from ''\n",
            "(((((((((\n",
            "\u{0}\u{1}\u{2} zebracorn(((\n",
            "🦄🦄🦄()\n",
        ];
        for line in lines {
            let _ = scan_file_for_language(line, Language::Python);
            let _ = scan_file_for_language(line, Language::Rust);
            let _ = scan_file_for_language(line, Language::TsJs);
        }
    }

    #[test]
    fn rust_use_path_to_name_yields_module_and_name() {
        let cands = scan_file_for_language("use zebracorn_crate::helpers::zebracorn_helper;\n", Language::Rust);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Import);
        assert_eq!(cands[0].name, "zebracorn_helper");
        assert_eq!(cands[0].module.as_deref(), Some("zebracorn_crate::helpers"));
    }

    #[test]
    fn rust_pub_use_is_still_recognized_after_modifier_strip() {
        let cands = scan_file_for_language("pub use zebracorn_crate::zebracorn_helper;\n", Language::Rust);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].name, "zebracorn_helper");
    }

    #[test]
    fn rust_glob_use_is_kept_unresolved() {
        let cands = scan_file_for_language("use zebracorn_crate::*;\n", Language::Rust);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Unresolved);
        assert_eq!(cands[0].name, "*");
    }

    #[test]
    fn rust_grouped_use_yields_only_the_first_name() {
        let cands = scan_file_for_language(
            "use zebracorn_crate::{alpha_name, beta_name};\n",
            Language::Rust,
        );
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].name, "alpha_name");
    }

    #[test]
    fn rust_call_site_yields_a_call_reference() {
        let cands = scan_file_for_language("let x = zebracorn_helper();\n", Language::Rust);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Call);
        assert_eq!(cands[0].name, "zebracorn_helper");
    }

    #[test]
    fn rust_def_line_does_not_self_reference_as_a_call() {
        let cands = scan_file_for_language("pub fn zebracorn_helper() {}\n", Language::Rust);
        assert!(cands.is_empty());
    }

    #[test]
    fn tsjs_named_import_yields_module_and_name() {
        let cands = scan_file_for_language(
            "import { zebracornHelper } from './zebracorn_module';\n",
            Language::TsJs,
        );
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Import);
        assert_eq!(cands[0].name, "zebracornHelper");
        assert_eq!(cands[0].module.as_deref(), Some("./zebracorn_module"));
    }

    #[test]
    fn tsjs_named_import_yields_only_the_first_name() {
        let cands = scan_file_for_language(
            "import { alphaName, betaName } from './zebracorn_module';\n",
            Language::TsJs,
        );
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].name, "alphaName");
    }

    #[test]
    fn tsjs_empty_named_import_is_kept_unresolved() {
        let cands = scan_file_for_language("import {  } from './zebracorn_module';\n", Language::TsJs);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Unresolved);
    }

    #[test]
    fn tsjs_default_import_yields_no_candidate() {
        let cands = scan_file_for_language(
            "import zebracornDefault from './zebracorn_module';\n",
            Language::TsJs,
        );
        assert!(cands.is_empty());
    }

    #[test]
    fn tsjs_call_site_yields_a_call_reference() {
        let cands = scan_file_for_language("const result = zebracornHelper();\n", Language::TsJs);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].kind, XrefKind::Call);
        assert_eq!(cands[0].name, "zebracornHelper");
    }

    #[test]
    fn tsjs_def_line_does_not_self_reference_as_a_call() {
        let cands = scan_file_for_language("export function zebracornHelper() {}\n", Language::TsJs);
        assert!(cands.is_empty());
    }

    #[test]
    fn resolve_relative_module_python_climbs_and_resolves_within_corpus() {
        let corpus_root = Path::new("/corpus");
        let from_path = Path::new("/corpus/pkg/importer.py");
        let result = resolve_relative_module(".sibling", Language::Python, from_path, corpus_root);
        assert_eq!(
            result,
            RelativeResolution::Resolved(PathBuf::from("/corpus/pkg/sibling"))
        );
    }

    #[test]
    fn resolve_relative_module_python_absolute_dotted_module_is_not_relative() {
        let corpus_root = Path::new("/corpus");
        let from_path = Path::new("/corpus/pkg/importer.py");
        let result = resolve_relative_module("pkg.other", Language::Python, from_path, corpus_root);
        assert_eq!(result, RelativeResolution::NotRelative);
    }

    #[test]
    fn resolve_relative_module_python_escaping_climb_is_rejected() {
        let corpus_root = Path::new("/corpus/pkg");
        let from_path = Path::new("/corpus/pkg/importer.py");
        // Four leading dots climbs 3 levels above the importer's own
        // directory, past corpus_root -- must come back Escaped, never a
        // path outside it.
        let result = resolve_relative_module("....secret", Language::Python, from_path, corpus_root);
        assert_eq!(result, RelativeResolution::Escaped);
    }

    #[test]
    fn resolve_relative_module_rust_super_climbs_one_level() {
        let corpus_root = Path::new("/corpus");
        let from_path = Path::new("/corpus/pkg/sub/importer.rs");
        let result = resolve_relative_module("super::helper", Language::Rust, from_path, corpus_root);
        assert_eq!(
            result,
            RelativeResolution::Resolved(PathBuf::from("/corpus/pkg/helper"))
        );
    }

    #[test]
    fn resolve_relative_module_rust_crate_anchors_at_corpus_root() {
        let corpus_root = Path::new("/corpus");
        let from_path = Path::new("/corpus/pkg/sub/importer.rs");
        let result = resolve_relative_module(
            "crate::pkg::helper",
            Language::Rust,
            from_path,
            corpus_root,
        );
        assert_eq!(
            result,
            RelativeResolution::Resolved(PathBuf::from("/corpus/pkg/helper"))
        );
    }

    #[test]
    fn resolve_relative_module_rust_bare_path_is_not_relative() {
        let corpus_root = Path::new("/corpus");
        let from_path = Path::new("/corpus/pkg/importer.rs");
        let result = resolve_relative_module(
            "some_crate::helper",
            Language::Rust,
            from_path,
            corpus_root,
        );
        assert_eq!(result, RelativeResolution::NotRelative);
    }

    #[test]
    fn resolve_relative_module_rust_super_chain_escaping_is_rejected() {
        let corpus_root = Path::new("/corpus/pkg");
        let from_path = Path::new("/corpus/pkg/importer.rs");
        let result = resolve_relative_module(
            "super::super::secret",
            Language::Rust,
            from_path,
            corpus_root,
        );
        assert_eq!(result, RelativeResolution::Escaped);
    }

    #[test]
    fn resolve_relative_module_tsjs_relative_specifier_resolves_within_corpus() {
        let corpus_root = Path::new("/corpus");
        let from_path = Path::new("/corpus/src/importer.ts");
        let result = resolve_relative_module("./bridge", Language::TsJs, from_path, corpus_root);
        assert_eq!(
            result,
            RelativeResolution::Resolved(PathBuf::from("/corpus/src/bridge"))
        );
    }

    #[test]
    fn resolve_relative_module_tsjs_bare_specifier_is_not_relative() {
        let corpus_root = Path::new("/corpus");
        let from_path = Path::new("/corpus/src/importer.ts");
        let result = resolve_relative_module("react", Language::TsJs, from_path, corpus_root);
        assert_eq!(result, RelativeResolution::NotRelative);
    }

    #[test]
    fn resolve_relative_module_tsjs_parent_traversal_escaping_is_rejected() {
        let corpus_root = Path::new("/corpus/src");
        let from_path = Path::new("/corpus/src/importer.ts");
        let result = resolve_relative_module(
            "../../../etc/passwd",
            Language::TsJs,
            from_path,
            corpus_root,
        );
        assert_eq!(result, RelativeResolution::Escaped);
    }
}
