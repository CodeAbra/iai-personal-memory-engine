"""The engine's plain-INSERT UNIQUE/PK enforcement backs its uniqueness probe
with one conflict index per INSERT that records no key-set identity, so it can
enforce at most one key-set. This locks the schema-level precondition against
the LIVE table DDLs (the source of truth in `hippo/_table.py`, not a mirror):
every table presents at most one enforceable (non-auto-injected) key-set. Add a
second column-level UNIQUE / PRIMARY KEY to any table and this turns RED -- the
signal to generalize the conflict cache before the engine can enforce it.

Table-level ``UNIQUE(...)`` is intentionally excluded: the engine tracks only
column-level uniqueness (its enforcement is column-level), so a table-level
UNIQUE is not an enforceable key-set on that path and cannot break the
single-conflict-index assumption.
"""
from __future__ import annotations

import re

import iai_mcp.hippo._table as table_mod

_TABLE_NAME_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE
)


def _discover_table_ddls() -> dict[str, str]:
    """Every live CREATE TABLE DDL constant, keyed by table name, discovered
    dynamically so a newly added table is covered automatically."""
    out: dict[str, str] = {}
    for name in dir(table_mod):
        if not name.startswith("_DDL_") or name.endswith("_INDEXES"):
            continue
        val = getattr(table_mod, name)
        if isinstance(val, str) and "CREATE TABLE" in val.upper():
            m = _TABLE_NAME_RE.search(val)
            if m:
                out[m.group(1)] = val
    return out


def _split_top_level(body: str) -> list[str]:
    """Split a CREATE TABLE body on commas that are not inside parentheses, so
    a composite ``PRIMARY KEY (a, b, c)`` stays one item."""
    items: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        items.append(tail)
    return items


def _enforceable_keyset_count(ddl: str) -> int:
    """Count the key-sets the engine's plain-INSERT path would enforce: each
    column-level UNIQUE, plus the PRIMARY KEY, excluding a lone AUTOINCREMENT
    ``vec_label`` PK (injected fresh per row, never collides) and any table-level
    ``UNIQUE(...)`` (not tracked by the column-level enforcement path)."""
    open_paren = ddl.index("(")
    close_paren = ddl.rindex(")")
    body = ddl[open_paren + 1 : close_paren]

    count = 0
    for item in _split_top_level(body):
        upper = item.upper()
        if upper.startswith("PRIMARY KEY"):
            count += 1  # table-level (possibly composite) PK
            continue
        if upper.startswith("UNIQUE"):
            continue  # table-level UNIQUE: not column-level, not enforced here
        # A column definition: "<name> <type> <constraints...>".
        name = item.split()[0]
        tokens = upper.split()
        is_pk = "PRIMARY" in tokens and "KEY" in tokens
        is_unique = "UNIQUE" in tokens
        if is_pk:
            if name.lower() == "vec_label" and "AUTOINCREMENT" in upper:
                continue  # the injected rowid alias can never collide
            count += 1
        elif is_unique:
            count += 1
    return count


def test_discovery_covers_every_registered_table() -> None:
    # Guard against silent coverage loss: every table the module registers in
    # `_TABLE_SQL` must have a discoverable DDL, so a renamed/off-convention
    # constant turns this RED instead of quietly narrowing the invariant check.
    discovered = set(_discover_table_ddls())
    registered = set(table_mod._TABLE_SQL)
    missing = registered - discovered
    assert not missing, f"tables with no discoverable DDL (coverage gap): {sorted(missing)}"


def test_every_live_table_presents_at_most_one_enforceable_keyset() -> None:
    ddls = _discover_table_ddls()
    assert ddls, "no live table DDL constants were discovered"
    for name, ddl in ddls.items():
        n = _enforceable_keyset_count(ddl)
        assert n <= 1, (
            f"{name} presents {n} enforceable key-sets; the engine's plain-INSERT "
            f"path backs one conflict index per INSERT and enforces only one -- "
            f"generalize the conflict cache to a per-key-set map before adding a "
            f"second UNIQUE / PRIMARY KEY.\nDDL:\n{ddl}"
        )


def test_counter_detects_a_second_enforceable_keyset() -> None:
    # Positive control: a PRIMARY KEY plus a separate column-level UNIQUE is two
    # enforceable key-sets, so the assertion above is not vacuously green.
    ddl = "CREATE TABLE two ( a INTEGER PRIMARY KEY , b TEXT UNIQUE )"
    assert _enforceable_keyset_count(ddl) == 2
    # A lone AUTOINCREMENT vec_label PK plus one UNIQUE stays at one.
    records_shaped = (
        "CREATE TABLE r ( vec_label INTEGER PRIMARY KEY AUTOINCREMENT , "
        "id TEXT NOT NULL UNIQUE )"
    )
    assert _enforceable_keyset_count(records_shaped) == 1
    # A table-level UNIQUE is not counted (column-level enforcement only).
    tag_shaped = (
        "CREATE TABLE t ( record_id TEXT NOT NULL , tag TEXT NOT NULL , "
        "UNIQUE(record_id, tag) )"
    )
    assert _enforceable_keyset_count(tag_shaped) == 0
