//! Recursive-descent parser: SQL string -> statement AST node.
//!
//! Accepts exactly the statement catalogue the host relies on (SELECT, INSERT,
//! UPDATE, DELETE, CREATE TABLE/INDEX, ALTER TABLE, DROP TABLE, PRAGMA) and
//! rejects every out-of-subset construct (JOIN, CTE, window function, subquery,
//! set operation, DISTINCT, multi-key tails) with a descriptive error at parse
//! time, so the executor never evaluates an unsupported statement. The error
//! message text is verbatim parity — callers and tests assert it byte-for-byte.

use lillibrain::Value;

use crate::ast::{
    DdlStmt, DeleteStmt, Expr, GroupSelectItem, InsertStmt, OrderKey, SelectExpr, SelectStmt, Stmt,
    UpdateStmt,
};
use crate::error::{EngineError, Result};
use crate::lexer::{tokenize, Token};

/// Maximum parenthesis nesting in expressions (pathological-input guard).
const MAX_NESTING_DEPTH: usize = 50;
/// Maximum items in an `IN (...)` list.
const MAX_IN_LIST_SIZE: usize = 1000;

/// Keywords that introduce an unsupported join form.
const JOIN_KEYWORDS: &[&str] = &[
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "FULL", "NATURAL",
];

/// Parse a SQL string into its statement AST node.
///
/// Returns [`EngineError::ParseError`] / [`EngineError::UnsupportedStatement`]
/// on any construct outside the supported subset, carrying the verbatim
/// reference message text.
pub fn parse(sql: &str) -> Result<Stmt> {
    let tokens = tokenize(sql)?;
    Parser::new(tokens).parse()
}

/// One select-list item: a plain column name, a non-column expression, or the
/// COUNT(*) sentinel; paired with an optional output alias.
enum SelectItem {
    /// A bare column name.
    Column(String),
    /// A COUNT(*) aggregate item in a grouped list.
    CountStar,
    /// A COALESCE / literal projection expression.
    Expr(Expr),
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            pos: 0,
            depth: 0,
        }
    }

    // -- token navigation ----------------------------------------------------

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn peek_at(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        tok
    }

    /// The display value of a token, matching the reference `Token.value`:
    /// keyword/ident text, the rendered number, the operator string, the raw
    /// quoted string/blob form, `?`, `:name`, or the punctuation character.
    fn token_value(tok: &Token) -> String {
        match tok {
            Token::Keyword(s) | Token::Ident(s) | Token::Op(s) | Token::Str(s) | Token::Blob(s) => {
                s.clone()
            }
            Token::Integer(v) => v.to_string(),
            Token::Real(f) => f.to_string(),
            Token::Placeholder => "?".to_string(),
            Token::NamedParam(name) => format!(":{name}"),
            Token::Punct(c) => c.to_string(),
            Token::Eof => String::new(),
        }
    }

    fn peek_value(&self) -> String {
        Self::token_value(self.peek())
    }

    fn peek_value_upper(&self) -> String {
        self.peek_value().to_ascii_uppercase()
    }

    fn expect(&mut self, value: &str) -> Result<Token> {
        let got = self.peek_value();
        if got.to_ascii_uppercase() != value.to_ascii_uppercase() {
            return Err(EngineError::parse(format!(
                "expected '{value}', got '{got}' at position {}",
                self.pos
            )));
        }
        Ok(self.advance())
    }

    /// Consume and return the current token's name value; accepts IDENT and
    /// KEYWORD so reserved words may be used as identifiers where the grammar
    /// unambiguously expects a name.
    fn expect_name(&mut self) -> Result<String> {
        match self.peek() {
            Token::Ident(_) | Token::Keyword(_) => Ok(Self::token_value(&self.advance())),
            other => Err(EngineError::parse(format!(
                "expected identifier or name, got {} ('{}')",
                token_kind_name(other),
                Self::token_value(other)
            ))),
        }
    }

    fn expect_integer(&mut self) -> Result<i64> {
        match self.peek() {
            Token::Integer(v) => {
                let v = *v;
                self.advance();
                Ok(v)
            }
            other => Err(EngineError::parse(format!(
                "expected token type INTEGER, got {} ('{}')",
                token_kind_name(other),
                Self::token_value(other)
            ))),
        }
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    /// True when the current token (KEYWORD or IDENT) matches any of `values`.
    fn at_keyword(&self, values: &[&str]) -> bool {
        match self.peek() {
            Token::Keyword(_) | Token::Ident(_) => {
                let up = self.peek_value_upper();
                values.iter().any(|v| v.eq_ignore_ascii_case(&up))
            }
            _ => false,
        }
    }

    fn at_punct(&self, value: char) -> bool {
        matches!(self.peek(), Token::Punct(c) if *c == value)
    }

    // -- top-level dispatch --------------------------------------------------

    fn parse(&mut self) -> Result<Stmt> {
        if self.at_eof() {
            return Err(EngineError::parse("empty SQL statement"));
        }

        let upper = self.peek_value_upper();

        if upper == "WITH" {
            return Err(EngineError::parse(
                "CTEs (WITH ... AS) are not supported in this SQL subset",
            ));
        }

        if upper == "EXPLAIN" {
            self.advance(); // EXPLAIN
            if self.at_keyword(&["QUERY"]) {
                self.advance();
                if self.at_keyword(&["PLAN"]) {
                    self.advance();
                }
            }
            let mut raw_parts: Vec<String> = Vec::new();
            while !self.at_eof() {
                raw_parts.push(Self::token_value(&self.advance()));
            }
            return Ok(Stmt::Ddl(DdlStmt {
                kind: "pragma".to_string(),
                raw: raw_parts.join(" "),
            }));
        }

        let stmt = match upper.as_str() {
            "SELECT" => Stmt::Select(self.parse_select()?),
            "INSERT" => Stmt::Insert(self.parse_insert()?),
            "UPDATE" => Stmt::Update(self.parse_update()?),
            "DELETE" => Stmt::Delete(self.parse_delete()?),
            "CREATE" | "ALTER" | "DROP" | "PRAGMA" => Stmt::Ddl(self.parse_ddl()),
            _ => {
                return Err(EngineError::parse(format!(
                    "unsupported statement type: '{}'",
                    self.peek_value()
                )));
            }
        };

        // Reject set operations after a parsed SELECT.
        if self.at_keyword(&["UNION", "INTERSECT", "EXCEPT"]) {
            let kw = self.peek_value_upper();
            return Err(EngineError::parse(format!(
                "{kw} is not supported; set operations are outside the supported grammar"
            )));
        }

        // Any token left after a fully parsed statement is an unsupported tail.
        if !self.at_eof() {
            return Err(EngineError::parse(format!(
                "unexpected trailing tokens after statement: '{}' at position {}",
                self.peek_value(),
                self.pos
            )));
        }

        Ok(stmt)
    }

    // -- SELECT --------------------------------------------------------------

    fn parse_select(&mut self) -> Result<SelectStmt> {
        self.expect("SELECT")?;

        if self.at_keyword(&["DISTINCT"]) {
            return Err(EngineError::parse(
                "DISTINCT is not supported in this SQL subset",
            ));
        }

        let mut s = SelectStmt::default();

        if self.at_punct('*') {
            self.advance();
            s.columns = vec!["*".to_string()];
        } else if self.at_keyword(&["COUNT"]) && self.count_star_is_sole_select() {
            self.advance(); // COUNT
            self.expect("(")?;
            self.expect("*")?;
            self.expect(")")?;
            s.aggregate = Some("count".to_string());
            s.count_alias = self.parse_optional_alias()?;
        } else if self.at_keyword(&["SUM"]) {
            self.advance();
            self.expect("(")?;
            s.aggregate_column = Some(self.expect_name()?);
            self.expect(")")?;
            s.aggregate = Some("sum".to_string());
        } else if self.at_keyword(&["MIN", "MAX"]) {
            let agg = self.peek_value().to_ascii_lowercase();
            self.advance();
            self.expect("(")?;
            s.aggregate_column = Some(self.expect_name()?);
            self.expect(")")?;
            s.aggregate = Some(agg);
        } else {
            let mut items: Vec<(SelectItem, Option<String>)> = vec![self.parse_select_item()?];
            while self.at_punct(',') {
                self.advance();
                items.push(self.parse_select_item()?);
            }

            let has_count = items
                .iter()
                .any(|(n, _)| matches!(n, SelectItem::CountStar));
            if has_count {
                for (node, alias) in items {
                    match node {
                        SelectItem::CountStar => {
                            let label = alias.unwrap_or_else(|| "COUNT(*)".to_string());
                            s.group_count_alias = Some(label.clone());
                            s.group_select.push(GroupSelectItem {
                                kind: "count".to_string(),
                                output_name: label,
                                source_name: String::new(),
                            });
                        }
                        SelectItem::Column(name) => {
                            let out = alias.unwrap_or_else(|| name.clone());
                            s.group_select.push(GroupSelectItem {
                                kind: "col".to_string(),
                                output_name: out,
                                source_name: name,
                            });
                        }
                        SelectItem::Expr(_) => {
                            return Err(EngineError::parse(
                                "a grouped COUNT(*) select list supports only plain \
                                 columns beside the aggregate",
                            ));
                        }
                    }
                }
            } else if items
                .iter()
                .all(|(n, a)| matches!(n, SelectItem::Column(_)) && a.is_none())
            {
                s.columns = items
                    .into_iter()
                    .map(|(n, _)| match n {
                        SelectItem::Column(name) => name,
                        _ => unreachable!(),
                    })
                    .collect();
            } else {
                for (node, alias) in items {
                    match node {
                        SelectItem::Column(name) => {
                            let out = alias.clone().unwrap_or_else(|| name.clone());
                            s.select_exprs.push(SelectExpr {
                                expr: Expr::Column(name),
                                output_name: out,
                            });
                        }
                        SelectItem::Expr(expr) => match alias {
                            Some(a) => s.select_exprs.push(SelectExpr {
                                expr,
                                output_name: a,
                            }),
                            None => match &expr {
                                Expr::Literal(v) => {
                                    let name = literal_output_name(v);
                                    s.select_exprs.push(SelectExpr {
                                        expr,
                                        output_name: name,
                                    });
                                }
                                _ => {
                                    return Err(EngineError::parse(
                                        "a COALESCE projection requires an output name \
                                         (use AS <name>)",
                                    ));
                                }
                            },
                        },
                        SelectItem::CountStar => unreachable!(),
                    }
                }
            }
        }

        // FROM <table>
        self.expect("FROM")?;
        match self.peek() {
            Token::Ident(_) | Token::Keyword(_) => {}
            _ => {
                return Err(EngineError::parse(format!(
                    "expected table name, got '{}'",
                    self.peek_value()
                )));
            }
        }
        s.table = Self::token_value(&self.advance());

        self.reject_join()?;

        s.where_clause = self.parse_where()?;

        // Optional GROUP BY <col> [HAVING COUNT(*) <op> <int>].
        if self.at_keyword(&["GROUP"]) {
            self.advance();
            self.expect("BY")?;
            s.group_by_col = Some(self.expect_name()?);
            if self.at_keyword(&["HAVING"]) {
                self.advance();
                self.expect("COUNT")?;
                self.expect("(")?;
                self.expect("*")?;
                self.expect(")")?;
                let op = match self.peek() {
                    Token::Op(o)
                        if matches!(o.as_str(), ">=" | ">" | "<=" | "<" | "=" | "!=" | "<>") =>
                    {
                        o.clone()
                    }
                    other => {
                        return Err(EngineError::parse(format!(
                            "HAVING COUNT(*) expects a comparison operator, got '{}'",
                            Self::token_value(other)
                        )));
                    }
                };
                self.advance();
                let op = if op == "<>" { "!=".to_string() } else { op };
                s.having_op = Some(op);
                s.having_value = Some(self.expect_integer()?);
            }
        } else if self.at_keyword(&["HAVING"]) {
            return Err(EngineError::parse(
                "HAVING without GROUP BY is not supported",
            ));
        }

        // Optional ORDER BY <col> [ASC|DESC] [, <col> [ASC|DESC]] — up to two keys.
        if self.at_keyword(&["ORDER"]) {
            self.advance();
            self.expect("BY")?;
            let (col, dt) = self.parse_order_by_key()?;
            let mut desc = false;
            if self.at_keyword(&["DESC"]) {
                self.advance();
                desc = true;
            } else if self.at_keyword(&["ASC"]) {
                self.advance();
            }
            s.order_by = Some(OrderKey {
                col,
                desc,
                datetime: dt,
            });
            if self.at_punct(',') {
                self.advance();
                let (col2, dt2) = self.parse_order_by_key()?;
                let mut desc2 = false;
                if self.at_keyword(&["DESC"]) {
                    self.advance();
                    desc2 = true;
                } else if self.at_keyword(&["ASC"]) {
                    self.advance();
                }
                s.order_by2 = Some(OrderKey {
                    col: col2,
                    desc: desc2,
                    datetime: dt2,
                });
            }
        }

        // Optional LIMIT [OFFSET], integer literal or '?' placeholder.
        if self.at_keyword(&["LIMIT"]) {
            self.advance();
            if matches!(self.peek(), Token::Placeholder) {
                self.advance();
                s.limit_is_param = true;
            } else {
                s.limit = Some(self.expect_integer()?);
            }
            if self.at_keyword(&["OFFSET"]) {
                self.advance();
                if matches!(self.peek(), Token::Placeholder) {
                    self.advance();
                    s.offset_is_param = true;
                } else {
                    s.offset = Some(self.expect_integer()?);
                }
            }
        }

        Ok(s)
    }

    /// Read one ORDER BY key, returning `(column_name, is_datetime_wrapped)`.
    fn parse_order_by_key(&mut self) -> Result<(String, bool)> {
        if self.at_keyword(&["DATETIME"])
            && self
                .peek_at(1)
                .is_some_and(|t| matches!(t, Token::Punct('(')))
        {
            self.advance(); // DATETIME
            self.expect("(")?;
            let col = self.expect_name()?;
            self.expect(")")?;
            return Ok((col, true));
        }
        Ok((self.expect_name()?, false))
    }

    fn parse_select_item(&mut self) -> Result<(SelectItem, Option<String>)> {
        // COUNT(*) aggregate item inside a grouped select list.
        if self.at_keyword(&["COUNT"]) && self.peek_next_is_open_paren() {
            self.advance();
            self.expect("(")?;
            self.expect("*")?;
            self.expect(")")?;
            let alias = self.parse_optional_alias()?;
            return Ok((SelectItem::CountStar, alias));
        }

        // COALESCE(<expr>, <fallback>) projection.
        if self.at_keyword(&["COALESCE"]) && self.peek_next_is_open_paren() {
            self.advance();
            self.expect("(")?;
            let mut args: Vec<Expr> = vec![self.parse_primary()?];
            while self.at_punct(',') {
                self.advance();
                args.push(self.parse_primary()?);
            }
            self.expect(")")?;
            if args.len() != 2 {
                return Err(EngineError::parse(format!(
                    "COALESCE expects exactly 2 arguments, got {}",
                    args.len()
                )));
            }
            let alias = self.parse_optional_alias()?;
            return Ok((SelectItem::Expr(Expr::Coalesce(args)), alias));
        }

        // Scalar literal in the select list.
        if matches!(
            self.peek(),
            Token::Integer(_)
                | Token::Real(_)
                | Token::Str(_)
                | Token::Blob(_)
                | Token::Placeholder
                | Token::NamedParam(_)
        ) {
            let lit = self.parse_primary()?;
            let alias = self.parse_optional_alias()?;
            return Ok((SelectItem::Expr(lit), alias));
        }
        if matches!(self.peek(), Token::Keyword(k) if matches!(k.to_ascii_uppercase().as_str(), "NULL" | "TRUE" | "FALSE"))
        {
            let lit = self.parse_primary()?;
            let alias = self.parse_optional_alias()?;
            return Ok((SelectItem::Expr(lit), alias));
        }

        // Plain column name.
        let name = self.expect_name()?;
        // Reject a function-call '(' after a non-COALESCE identifier.
        if self.at_punct('(') {
            self.advance();
            let mut depth = 1;
            while !self.at_eof() && depth > 0 {
                if self.at_punct('(') {
                    depth += 1;
                } else if self.at_punct(')') {
                    depth -= 1;
                }
                self.advance();
            }
            if self.at_keyword(&["OVER"]) {
                return Err(EngineError::parse(
                    "window functions (OVER) are not supported in this SQL subset",
                ));
            }
            return Err(EngineError::parse(
                "function calls are not supported in the SELECT list; \
                 use COUNT(*) or SUM(col) for aggregates, or COALESCE(col, lit)",
            ));
        }
        let alias = self.parse_optional_alias()?;
        Ok((SelectItem::Column(name), alias))
    }

    fn peek_next_is_open_paren(&self) -> bool {
        self.peek_at(1)
            .is_some_and(|t| matches!(t, Token::Punct('(')))
    }

    /// True when COUNT(*) at the current position is the sole select item.
    fn count_star_is_sole_select(&self) -> bool {
        let p = self.pos;
        let toks = &self.tokens;
        let ok = p + 3 < toks.len()
            && matches!(toks[p + 1], Token::Punct('('))
            && matches!(toks[p + 2], Token::Punct('*'))
            && matches!(toks[p + 3], Token::Punct(')'));
        if !ok {
            return false;
        }
        let mut after = p + 4;
        if after >= toks.len() {
            return true;
        }
        if matches!(toks[after], Token::Punct(',')) {
            return false;
        }
        let nxt_upper = Self::token_value(&toks[after]).to_ascii_uppercase();
        if nxt_upper == "AS" && after + 1 < toks.len() {
            after += 2;
        } else if matches!(toks[after], Token::Ident(_) | Token::Keyword(_)) && nxt_upper != "FROM"
        {
            after += 1;
        }
        if after < toks.len() && matches!(toks[after], Token::Punct(',')) {
            return false;
        }
        true
    }

    /// Consume an `AS <name>` (or a bare alias) after a select item.
    fn parse_optional_alias(&mut self) -> Result<Option<String>> {
        if self.at_keyword(&["AS"]) {
            self.advance();
            return Ok(Some(self.expect_name()?));
        }
        if matches!(self.peek(), Token::Ident(_) | Token::Keyword(_)) && !self.at_keyword(&["FROM"])
        {
            return Ok(Some(Self::token_value(&self.advance())));
        }
        Ok(None)
    }

    // -- INSERT --------------------------------------------------------------

    fn parse_insert(&mut self) -> Result<InsertStmt> {
        self.expect("INSERT")?;

        let mut i = InsertStmt::default();
        if self.at_keyword(&["OR"]) {
            self.advance();
            if self.at_keyword(&["IGNORE"]) {
                self.advance();
                i.or_ignore = true;
            } else if self.at_keyword(&["REPLACE"]) {
                self.advance();
                i.or_replace = true;
            } else {
                return Err(EngineError::parse(format!(
                    "expected IGNORE or REPLACE after OR, got '{}' at position {}",
                    self.peek_value(),
                    self.pos
                )));
            }
        }

        self.expect("INTO")?;
        i.table = self.expect_name()?;

        self.expect("(")?;
        i.columns.push(self.expect_name()?);
        while self.at_punct(',') {
            self.advance();
            i.columns.push(self.expect_name()?);
        }
        self.expect(")")?;

        self.expect("VALUES")?;

        self.expect("(")?;
        i.values.push(self.parse_primary()?);
        while self.at_punct(',') {
            self.advance();
            i.values.push(self.parse_primary()?);
        }
        self.expect(")")?;

        if self.at_keyword(&["ON"]) {
            self.advance();
            self.expect("CONFLICT")?;
            self.expect("(")?;
            i.conflict_keys.push(self.expect_name()?);
            while self.at_punct(',') {
                self.advance();
                i.conflict_keys.push(self.expect_name()?);
            }
            self.expect(")")?;
            self.expect("DO")?;

            if self.at_keyword(&["NOTHING"]) {
                self.advance();
                i.conflict_action = Some("nothing".to_string());
            } else if self.at_keyword(&["UPDATE"]) {
                self.advance();
                i.conflict_action = Some("update".to_string());
                self.expect("SET")?;
                i.set_clauses = self.parse_set_clauses(true)?;
            } else {
                return Err(EngineError::parse(format!(
                    "expected NOTHING or UPDATE after DO, got '{}'",
                    self.peek_value()
                )));
            }
        }

        Ok(i)
    }

    // -- UPDATE / DELETE -----------------------------------------------------

    fn parse_update(&mut self) -> Result<UpdateStmt> {
        self.expect("UPDATE")?;
        let table = self.expect_name()?;
        self.expect("SET")?;
        let set_clauses = self.parse_set_clauses(false)?;
        let where_clause = self.parse_where()?;
        Ok(UpdateStmt {
            table,
            set_clauses,
            where_clause,
        })
    }

    fn parse_delete(&mut self) -> Result<DeleteStmt> {
        self.expect("DELETE")?;
        self.expect("FROM")?;
        let table = self.expect_name()?;
        let where_clause = self.parse_where()?;
        Ok(DeleteStmt {
            table,
            where_clause,
        })
    }

    // -- DDL / PRAGMA --------------------------------------------------------

    fn parse_ddl(&mut self) -> DdlStmt {
        let first = self.peek_value_upper();
        let mut raw_parts: Vec<String> = Vec::new();
        while !self.at_eof() {
            raw_parts.push(Self::token_value(&self.advance()));
        }
        let raw = raw_parts.join(" ");

        let kind = match first.as_str() {
            "CREATE" => {
                let words: Vec<&str> = raw.split(' ').collect();
                let second = words
                    .get(1)
                    .map(|w| w.to_ascii_uppercase())
                    .unwrap_or_default();
                let third = words
                    .get(2)
                    .map(|w| w.to_ascii_uppercase())
                    .unwrap_or_default();
                if second == "INDEX" || (second == "UNIQUE" && third == "INDEX") {
                    "create_index"
                } else {
                    "create_table"
                }
            }
            "ALTER" => {
                let upper = raw.to_ascii_uppercase();
                if upper.contains(" RENAME ") {
                    "alter_rename_table"
                } else if upper.contains(" DROP ") {
                    "alter_drop_column"
                } else {
                    "alter_add_column"
                }
            }
            "DROP" => "drop_table",
            _ => "pragma",
        };

        DdlStmt {
            kind: kind.to_string(),
            raw,
        }
    }

    // -- SET clause ----------------------------------------------------------

    fn parse_set_clauses(&mut self, allow_excluded: bool) -> Result<Vec<(String, Expr)>> {
        let mut clauses = Vec::new();
        let col = self.expect_name()?;
        self.expect("=")?;
        let val = self.parse_set_rhs(allow_excluded)?;
        clauses.push((col, val));
        while self.at_punct(',') {
            self.advance();
            let col = self.expect_name()?;
            self.expect("=")?;
            let val = self.parse_set_rhs(allow_excluded)?;
            clauses.push((col, val));
        }
        Ok(clauses)
    }

    fn parse_set_rhs(&mut self, allow_excluded: bool) -> Result<Expr> {
        if allow_excluded
            && matches!(self.peek(), Token::Keyword(k) if k.eq_ignore_ascii_case("EXCLUDED"))
        {
            self.advance(); // EXCLUDED
            self.expect(".")?;
            let col = self.expect_name()?;
            return Ok(Expr::Excluded(col));
        }
        self.parse_primary()
    }

    // -- WHERE / expression grammar ------------------------------------------

    fn parse_where(&mut self) -> Result<Option<Expr>> {
        if !self.at_keyword(&["WHERE"]) {
            return Ok(None);
        }
        self.advance();
        Ok(Some(self.parse_expr()?))
    }

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr> {
        let mut left = self.parse_and()?;
        while self.at_keyword(&["OR"]) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::BinOp {
                op: "OR".to_string(),
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr> {
        let mut left = self.parse_not()?;
        while self.at_keyword(&["AND"]) {
            self.advance();
            let right = self.parse_not()?;
            left = Expr::BinOp {
                op: "AND".to_string(),
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr> {
        if self.at_keyword(&["NOT"]) {
            self.advance();
            let operand = self.parse_comparison()?;
            return Ok(Expr::UnaryOp {
                op: "NOT".to_string(),
                operand: Box::new(operand),
            });
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let left = self.parse_primary()?;

        if self.at_keyword(&["OVER"]) {
            return Err(EngineError::parse(
                "window functions (OVER) are not supported in this SQL subset",
            ));
        }

        // IS [NOT] NULL
        if self.at_keyword(&["IS"]) {
            self.advance();
            let mut negated = false;
            if self.at_keyword(&["NOT"]) {
                self.advance();
                negated = true;
            }
            self.expect("NULL")?;
            return Ok(Expr::IsNull {
                operand: Box::new(left),
                negated,
            });
        }

        // IN (...)
        if self.at_keyword(&["IN"]) {
            self.advance();
            self.expect("(")?;
            if self.at_keyword(&["SELECT"]) {
                return Err(EngineError::parse(
                    "subquery in IN (...) is not supported; use parameterized IN (?, ...) instead",
                ));
            }
            let items = self.parse_in_items()?;
            self.expect(")")?;
            return Ok(Expr::InList {
                operand: Box::new(left),
                items,
                text_set: None,
            });
        }

        // NOT IN (...)
        if self.at_keyword(&["NOT"])
            && self
                .peek_at(1)
                .is_some_and(|t| Self::token_value(t).eq_ignore_ascii_case("IN"))
        {
            self.advance(); // NOT
            self.advance(); // IN
            self.expect("(")?;
            if self.at_keyword(&["SELECT"]) {
                return Err(EngineError::parse(
                    "subquery in IN (...) is not supported; use parameterized IN (?, ...) instead",
                ));
            }
            let items = self.parse_in_items()?;
            self.expect(")")?;
            return Ok(Expr::UnaryOp {
                op: "NOT".to_string(),
                operand: Box::new(Expr::InList {
                    operand: Box::new(left),
                    items,
                    text_set: None,
                }),
            });
        }

        // LIKE
        if self.at_keyword(&["LIKE"]) {
            self.advance();
            let right = self.parse_primary()?;
            return Ok(Expr::BinOp {
                op: "LIKE".to_string(),
                left: Box::new(left),
                right: Box::new(right),
            });
        }

        // NOT LIKE
        if self.at_keyword(&["NOT"])
            && self
                .peek_at(1)
                .is_some_and(|t| Self::token_value(t).eq_ignore_ascii_case("LIKE"))
        {
            self.advance(); // NOT
            self.advance(); // LIKE
            let right = self.parse_primary()?;
            return Ok(Expr::BinOp {
                op: "NOT LIKE".to_string(),
                left: Box::new(left),
                right: Box::new(right),
            });
        }

        // Binary comparison operator
        if let Token::Op(op) = self.peek() {
            let op = op.clone();
            self.advance();
            let right = self.parse_primary()?;
            return Ok(Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }

        Ok(left)
    }

    fn parse_in_items(&mut self) -> Result<Vec<Expr>> {
        let mut items = vec![self.parse_primary()?];
        while self.at_punct(',') {
            self.advance();
            if self.at_keyword(&["SELECT"]) {
                return Err(EngineError::parse(
                    "subquery in IN (...) is not supported; use parameterized IN (?, ...) instead",
                ));
            }
            items.push(self.parse_primary()?);
            if items.len() > MAX_IN_LIST_SIZE {
                return Err(EngineError::parse(format!(
                    "IN list exceeds maximum allowed size of {MAX_IN_LIST_SIZE}; \
                     reduce the number of items or use a temporary table"
                )));
            }
        }
        Ok(items)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        // Parenthesised expression.
        if self.at_punct('(') {
            self.advance();
            self.depth += 1;
            if self.depth > MAX_NESTING_DEPTH {
                return Err(EngineError::parse(format!(
                    "expression nesting depth exceeds maximum of {MAX_NESTING_DEPTH}; \
                     simplify the predicate"
                )));
            }
            if self.at_keyword(&["SELECT"]) {
                return Err(EngineError::parse(
                    "subquery is not supported in this SQL subset",
                ));
            }
            let expr = self.parse_expr()?;
            self.expect(")")?;
            self.depth -= 1;
            return Ok(expr);
        }

        // Placeholder.
        if matches!(self.peek(), Token::Placeholder) {
            self.advance();
            return Ok(Expr::Placeholder);
        }

        // Named placeholder.
        if let Token::NamedParam(name) = self.peek() {
            let name = name.clone();
            self.advance();
            return Ok(Expr::NamedParam(name));
        }

        // String literal — strip the surrounding quotes.
        if let Token::Str(raw) = self.peek() {
            let raw = raw.clone();
            self.advance();
            let body = strip_outer_quotes(&raw);
            return Ok(Expr::Literal(Value::Text(body)));
        }

        // Hex-BLOB literal: x'<hex>' -> bytes.
        if let Token::Blob(raw) = self.peek() {
            let raw = raw.clone();
            self.advance();
            let hex = &raw[2..raw.len() - 1];
            let bytes = decode_hex(hex)?;
            return Ok(Expr::Literal(Value::Blob(bytes)));
        }

        // Integer literal.
        if let Token::Integer(v) = self.peek() {
            let v = *v;
            self.advance();
            return Ok(Expr::Literal(Value::Int(v)));
        }

        // Real literal.
        if let Token::Real(f) = self.peek() {
            let f = *f;
            self.advance();
            return Ok(Expr::Literal(Value::Float(f)));
        }

        // Keyword literals.
        if let Token::Keyword(k) = self.peek() {
            match k.to_ascii_uppercase().as_str() {
                "NULL" => {
                    self.advance();
                    return Ok(Expr::Literal(Value::Null));
                }
                "TRUE" => {
                    self.advance();
                    return Ok(Expr::Literal(Value::Int(1)));
                }
                "FALSE" => {
                    self.advance();
                    return Ok(Expr::Literal(Value::Int(0)));
                }
                _ => {}
            }
        }

        // COALESCE(<expr>, <fallback>).
        if self.matches_call("COALESCE") {
            self.advance(); // COALESCE
            self.expect("(")?;
            let mut args = vec![self.parse_primary()?];
            while self.at_punct(',') {
                self.advance();
                args.push(self.parse_primary()?);
            }
            self.expect(")")?;
            if args.len() != 2 {
                return Err(EngineError::parse(format!(
                    "COALESCE expects exactly 2 arguments, got {}",
                    args.len()
                )));
            }
            return Ok(Expr::Coalesce(args));
        }

        // datetime(<expr>).
        if self.matches_call("DATETIME") {
            self.advance(); // DATETIME
            self.depth += 1;
            if self.depth > MAX_NESTING_DEPTH {
                return Err(EngineError::parse(format!(
                    "expression nesting depth exceeds maximum of {MAX_NESTING_DEPTH}; \
                     simplify the predicate"
                )));
            }
            self.expect("(")?;
            let mut args = vec![self.parse_primary()?];
            while self.at_punct(',') {
                self.advance();
                args.push(self.parse_primary()?);
            }
            self.expect(")")?;
            self.depth -= 1;
            if args.is_empty() {
                return Err(EngineError::parse(
                    "datetime() expects at least 1 argument, got 0".to_string(),
                ));
            }
            return Ok(Expr::FuncCall {
                name: "datetime".to_string(),
                args,
            });
        }

        // EXCLUDED in expression context is invalid.
        if matches!(self.peek(), Token::Keyword(k) if k.eq_ignore_ascii_case("EXCLUDED")) {
            return Err(EngineError::parse(
                "unexpected keyword EXCLUDED in expression context",
            ));
        }

        // Identifier — column reference, possibly dotted.
        if matches!(self.peek(), Token::Ident(_) | Token::Keyword(_)) {
            let name = Self::token_value(&self.advance());
            if self.at_punct('.') {
                self.advance();
                let col = self.expect_name()?;
                return Ok(Expr::Column(format!("{name}.{col}")));
            }
            if self.at_keyword(&["OVER"]) {
                return Err(EngineError::parse(
                    "window functions (OVER) are not supported in this SQL subset",
                ));
            }
            return Ok(Expr::Column(name));
        }

        Err(EngineError::parse(format!(
            "unexpected token '{}' in expression",
            self.peek_value()
        )))
    }

    /// True when the current token is `name` (IDENT or KEYWORD) immediately
    /// followed by an opening parenthesis — a call form.
    fn matches_call(&self, name: &str) -> bool {
        matches!(self.peek(), Token::Ident(_) | Token::Keyword(_))
            && self.peek_value_upper() == name
            && self.peek_next_is_open_paren()
    }

    fn reject_join(&self) -> Result<()> {
        if self.at_keyword(JOIN_KEYWORDS) {
            let tok_val = self.peek_value();
            return Err(EngineError::parse(format!(
                "JOIN ({tok_val}) is not supported; only single-table queries are allowed"
            )));
        }
        Ok(())
    }
}

/// The display name of a token's kind, for error messages.
fn token_kind_name(tok: &Token) -> &'static str {
    match tok {
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
    }
}

/// Strip a single pair of surrounding single quotes, matching the reference's
/// `raw[1:-1]` (no doubled-quote unescaping — the lexer never produces one).
fn strip_outer_quotes(raw: &str) -> String {
    if raw.len() >= 2 && raw.starts_with('\'') && raw.ends_with('\'') {
        raw[1..raw.len() - 1].to_string()
    } else {
        raw.to_string()
    }
}

/// The output column name for an unaliased literal projection (sqlite3 parity:
/// `SELECT 1` -> column "1"; `SELECT NULL` -> column "NULL").
fn literal_output_name(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Text(s) => s.clone(),
        Value::Blob(_) => "NULL".to_string(),
    }
}

/// Decode a hex string to bytes; an odd-length or non-hex body fails loud,
/// matching the reference's `bytes.fromhex` ValueError.
fn decode_hex(hex: &str) -> Result<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return Err(EngineError::parse(format!(
            "non-hexadecimal number found in fromhex() arg at position {}",
            hex.len()
        )));
    }
    let bytes = hex.as_bytes();
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_val(bytes[i])?;
        let lo = hex_val(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_val(b: u8) -> Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(EngineError::parse(
            "non-hexadecimal number found in fromhex() arg",
        )),
    }
}
