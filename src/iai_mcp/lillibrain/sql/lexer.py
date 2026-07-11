"""SQL lexer: single-pass tokenizer using re.Scanner.

All functions are pure (no I/O, no module-level state beyond the compiled
_SCANNER constant). The remainder check after _SCANNER.scan() is the only
guard against re.Scanner silently stopping at an unrecognised character.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Token type enumeration
# ---------------------------------------------------------------------------


class TT(Enum):
    """Token type for the SQL lexer."""

    KEYWORD = auto()
    IDENT = auto()
    INTEGER = auto()
    REAL = auto()
    STRING = auto()
    BLOB = auto()
    PLACEHOLDER = auto()
    NAMED_PARAM = auto()
    OP = auto()
    PUNCT = auto()
    EOF = auto()


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """Single lexer token."""

    type: TT
    value: str


# ---------------------------------------------------------------------------
# Keyword set and scanner
# ---------------------------------------------------------------------------

_KEYWORDS: frozenset[str] = frozenset({
    "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE", "AND", "OR", "NOT",
    "IS", "NULL", "IN", "LIKE", "LIMIT", "INTO", "VALUES", "SET", "ON", "CONFLICT",
    "DO", "NOTHING", "EXCLUDED", "CREATE", "TABLE", "IF", "EXISTS", "INDEX",
    "ALTER", "ADD", "COLUMN", "PRAGMA", "BEGIN", "COMMIT", "ROLLBACK",
    "INTEGER", "TEXT", "REAL", "BLOB", "PRIMARY", "KEY", "AUTOINCREMENT",
    "UNIQUE", "DEFAULT", "ORDER", "BY", "ASC", "DESC", "COUNT", "SUM",
    "MIN", "MAX", "IGNORE", "REPLACE", "TRUE", "FALSE",
})

_SCANNER: re.Scanner = re.Scanner([  # type: ignore[attr-defined]
    (r"--[^\n]*",               lambda s, t: None),           # comment — discard
    (r"\s+",                    lambda s, t: None),           # whitespace — discard
    (r"==|!=|<>|<=|>=|[=<>]",  lambda s, t: (TT.OP, "=" if t == "==" else t)),
    (r"\?",                     lambda s, t: (TT.PLACEHOLDER, t)),
    (r":[A-Za-z_][A-Za-z0-9_]*", lambda s, t: (TT.NAMED_PARAM, t)),
    (r"[(),;.*]",               lambda s, t: (TT.PUNCT, t)),
    (r"[0-9]+\.[0-9]+",        lambda s, t: (TT.REAL, t)),
    (r"[0-9]+",                 lambda s, t: (TT.INTEGER, t)),
    (r"'(?:[^'\\]|\\.)*'",     lambda s, t: (TT.STRING, t)),
    (r"[xX]'[0-9a-fA-F]*'",     lambda s, t: (TT.BLOB, t)),
    (r"\[[^\]]+\]",             lambda s, t: (TT.IDENT, t[1:-1])),
    (r"[A-Za-z_][A-Za-z0-9_]*", lambda s, t: (
        TT.KEYWORD if t.upper() in _KEYWORDS else TT.IDENT, t
    )),
])


# ---------------------------------------------------------------------------
# Public tokenize function
# ---------------------------------------------------------------------------


def tokenize(sql: str) -> list[Token]:
    """Tokenize a SQL string into a list of Token objects ending with EOF.

    Discards whitespace and -- comments in a single pass. Raises SyntaxError
    when any character in the input cannot be matched by the scanner patterns.
    Always appends an EOF token as the final element.
    """
    tokens_raw, remainder = _SCANNER.scan(sql)
    if remainder.strip():
        raise SyntaxError(f"unexpected token near: {remainder[:20]!r}")
    return [Token(tt, v) for tt, v in tokens_raw if tt is not None] + [Token(TT.EOF, "")]
