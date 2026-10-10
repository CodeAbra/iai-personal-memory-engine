//! Whole-file token scan: hand-rolled ASCII tokenizer (no regex) plus a
//! deterministic feature-id hash, mirroring rank_core's tokenizer shape.

/// `[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]{3,}` scanned by hand -- no regex crate.
/// Alpha/underscore runs need length >= 2 to emit; digit runs need >= 3.
fn find_raw_tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            let mut j = i + 1;
            while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            if j - start >= 2 {
                out.push(chars[start..j].iter().collect());
                i = j;
                continue;
            }
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            let mut j = i + 1;
            while j < n && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j - start >= 3 {
                out.push(chars[start..j].iter().collect());
                i = j;
                continue;
            }
            i += 1;
            continue;
        }
        i += 1;
    }
    out
}

/// Split before an upper following a lower/digit, and before the last
/// upper of an acronym run.
fn camel_split(raw: &str) -> Vec<String> {
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }
    let mut parts = Vec::new();
    let mut start = 0usize;
    for i in 1..n {
        let prev = chars[i - 1];
        let cur = chars[i];
        let split_before_upper =
            (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase();
        let split_before_acronym_tail = prev.is_ascii_uppercase()
            && cur.is_ascii_uppercase()
            && i + 1 < n
            && chars[i + 1].is_ascii_lowercase();
        if split_before_upper || split_before_acronym_tail {
            parts.push(chars[start..i].iter().collect::<String>());
            start = i;
        }
    }
    parts.push(chars[start..n].iter().collect::<String>());
    parts
}

/// Whole-token plus snake/camel parts, lowercased -- the lowercased whole
/// token always; snake parts only when more than one survives the length
/// filter; camel parts only when more than one survives.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in find_raw_tokens(text) {
        let low = raw.to_lowercase();
        out.push(low.clone());

        let parts: Vec<&str> = low.split('_').filter(|p| p.len() >= 2).collect();
        if parts.len() > 1 {
            out.extend(parts.into_iter().map(str::to_string));
        }

        let camel_parts: Vec<String> = camel_split(&raw)
            .into_iter()
            .filter(|p| p.chars().count() >= 2)
            .map(|p| p.to_lowercase())
            .collect();
        if camel_parts.len() > 1 {
            out.extend(camel_parts);
        }
    }
    out
}

/// Whole raw tokens only, lowercased -- no snake/camel decomposition.
/// Callers needing an EXACT identifier match (never a sub-token standing in
/// for the whole name) must use this instead of `tokenize()`, whose
/// decomposed parts exist only to feed a fuzzy sub-token match.
pub fn tokenize_whole(text: &str) -> Vec<String> {
    find_raw_tokens(text)
        .into_iter()
        .map(|t| t.to_lowercase())
        .collect()
}

/// FNV-1a 32-bit -- deterministic token -> feature_id. Persisted postings
/// carry only this id, never the raw token string.
pub fn feature_id(token: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for b in token.as_bytes() {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_emits_lowercased_whole_token() {
        let toks = tokenize("Hello World");
        assert!(toks.contains(&"hello".to_string()));
        assert!(toks.contains(&"world".to_string()));
    }

    #[test]
    fn feature_id_is_deterministic() {
        assert_eq!(feature_id("zebra"), feature_id("zebra"));
        assert_ne!(feature_id("zebra"), feature_id("giraffe"));
    }

    #[test]
    fn short_alpha_runs_below_threshold_are_dropped() {
        let toks = tokenize("a bb ccc");
        assert!(!toks.contains(&"a".to_string()));
        assert!(toks.contains(&"bb".to_string()));
        assert!(toks.contains(&"ccc".to_string()));
    }

    /// Asserts directly against tokenize()'s current output -- never
    /// reconstruct a shadow predicate here, the two length filters interact
    /// with the enclosing parts-length gate in ways a local copy would miss.
    #[test]
    fn short_token_retune_is_additive_across_identifier_shapes() {
        let cases: &[(&str, &[&str])] = &[
            ("db", &["db"]),
            ("session", &["session"]),
            ("db_conn", &["db", "conn"]),
            ("getDB", &["get", "db"]),
            ("v2Api", &["v2", "api"]),
            ("HTTPRequest", &["http", "request"]),
            ("parse_i18n", &["parse", "i18n"]),
            ("parseUserId_v2", &["parse", "user", "v2", "id_v2"]),
        ];
        for (identifier, expected_subtokens) in cases {
            let toks = tokenize(identifier);
            for sub in *expected_subtokens {
                assert!(
                    toks.contains(&sub.to_string()),
                    "{identifier:?} must resolve via sub-token {sub:?}, got {toks:?}"
                );
            }
        }
    }

    /// Fixed sample, fixed expected sum -- catches a tokenizer change that
    /// emits wildly more or fewer tokens than the current shape.
    #[test]
    fn absolute_token_count_regression_anchor() {
        const REPRESENTATIVE_LINES: &[&str] = &[
            "fn get_user_by_id(user_id: u32) -> Option<User> {",
            "    let dbConn = DbConn::new();",
            "    let parsed = parse_i18n(&raw);",
            "    HTTPRequest::send(&url, &body);",
            "    let v2Api = ApiV2Client::new();",
            "}",
        ];
        let total: usize = REPRESENTATIVE_LINES.iter().map(|l| tokenize(l).len()).sum();
        assert_eq!(
            total, 41,
            "token count over the fixed representative sample regressed, got {total}"
        );
    }

    #[test]
    fn tokenize_whole_never_decomposes_snake_or_camel_parts() {
        let toks = tokenize_whole("IAI_MCP_STORE");
        assert_eq!(toks, vec!["iai_mcp_store".to_string()]);
        assert!(!toks.contains(&"store".to_string()));
        assert!(!toks.contains(&"iai".to_string()));
        assert!(!toks.contains(&"mcp".to_string()));
    }

    #[test]
    fn tokenize_whole_splits_only_on_raw_token_boundaries() {
        let toks = tokenize_whole("where is watch_scan_once defined");
        assert_eq!(
            toks,
            vec![
                "where".to_string(),
                "is".to_string(),
                "watch_scan_once".to_string(),
                "defined".to_string(),
            ]
        );
    }

    #[test]
    fn short_identifier_regression_set_resolves_via_sub_token() {
        let cases: &[(&str, &[&str])] = &[
            ("db", &["db"]),
            ("fs", &["fs"]),
            ("io", &["io"]),
            ("i18n", &["i18n"]),
            ("id", &["id"]),
            ("getDB", &["get", "db"]),
            ("db_conn", &["db", "conn"]),
            ("parse_i18n", &["parse", "i18n"]),
        ];
        for (identifier, expected_subtokens) in cases {
            let toks = tokenize(identifier);
            for sub in *expected_subtokens {
                assert!(
                    toks.contains(&sub.to_string()),
                    "{identifier:?} must resolve via sub-token {sub:?}, got {toks:?}"
                );
            }
        }
    }
}
