"""Column-schema and row-batch plumbing for the Hippo storage shim.

Replaces the columnar-library objects the shim used for interchange: a
schema here is a typed column list, a batch is a dict of equal-length
columns, and a table is rows. Nothing in this module touches storage —
encode/decode and SQL stay in the table layer.

Dtypes are interned string tokens (``"string"``, ``"int64"``,
``"list<float32:384>"`` …). Token equality IS dtype equality, and the
``is_*`` predicates parse the token prefix.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator


# ---------------------------------------------------------------------------
# Dtype tokens
# ---------------------------------------------------------------------------

class _Token(str):
    """Dtype token: a string whose equality IS dtype equality, plus the
    ``equals`` method callers ported from the columnar library still use."""

    def equals(self, other: object) -> bool:
        return str(self) == str(other)


#: Dtypes ARE their tokens; the alias keeps annotations readable.
DataType = str


def string() -> _Token:
    return _Token("string")


def binary() -> _Token:
    return _Token("binary")


def bool_() -> _Token:
    return _Token("bool")


def int64() -> _Token:
    return _Token("int64")


def float32() -> _Token:
    return _Token("float32")


def float64() -> _Token:
    return _Token("float64")


def timestamp(unit: str = "us", tz: str | None = None) -> _Token:
    return _Token(
        f"timestamp[{unit}]" if tz is None else f"timestamp[{unit},{tz}]"
    )


class _ListToken(_Token):
    """List dtype token that also answers the structural attribute reads the
    columnar library supported (``.list_size``, ``.value_type``)."""

    list_size: int
    value_type: str

    def __new__(cls, item: str, size: int) -> "_ListToken":
        tok = super().__new__(cls, f"list<{item}:{size}>")
        tok.list_size = size
        tok.value_type = item
        return tok


def list_(item: str, size: int = -1) -> str:
    return _ListToken(item, size)


def int32() -> _Token:
    return _Token("int32")


class types:

    @staticmethod
    def is_integer(t: str) -> bool:
        return t in ("int32", "int64")

    @staticmethod
    def is_floating(t: str) -> bool:
        return t in ("float32", "float64")

    @staticmethod
    def is_boolean(t: str) -> bool:
        return t == "bool"

    @staticmethod
    def is_list(t: str) -> bool:
        return t.startswith("list<")

    @staticmethod
    def is_large_list(t: str) -> bool:
        return False

    @staticmethod
    def is_timestamp(t: str) -> bool:
        return t.startswith("timestamp")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class Field:

    __slots__ = ("name", "type", "nullable")

    def __init__(self, name: str, type: str, nullable: bool = True) -> None:  # noqa: A002
        self.name = name
        self.type = type
        self.nullable = nullable

    def __repr__(self) -> str:  # pragma: no cover
        return f"Field({self.name!r}, {self.type!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Field)
            and self.name == other.name
            and self.type == other.type
        )


def field(name: str, type: str, nullable: bool = True) -> Field:  # noqa: A002
    return Field(name, type, nullable)


class Schema:

    def __init__(self, fields: Iterable[Field]) -> None:
        self._fields: list[Field] = list(fields)

    def __iter__(self) -> Iterator[Field]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self._fields]

    def field(self, name: str) -> Field:
        for f in self._fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def get_field_index(self, name: str) -> int:
        for i, f in enumerate(self._fields):
            if f.name == name:
                return i
        return -1

    def set(self, index: int, new_field: Field) -> "Schema":
        fields = list(self._fields)
        fields[index] = new_field
        return Schema(fields)

    def __repr__(self) -> str:
        # Stable content-derived repr: schema-identity assertions compare it.
        cols = ", ".join(f"{f.name}:{f.type}" for f in self._fields)
        return f"Schema({cols})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Schema) and self._fields == other._fields


def schema(fields: Iterable[Field]) -> Schema:
    return Schema(fields)


# ---------------------------------------------------------------------------
# Batches and tables
# ---------------------------------------------------------------------------


class _Column:

    __slots__ = ("_values",)

    def __init__(self, values: list) -> None:
        self._values = values

    def to_pylist(self) -> list:
        return list(self._values)


class RecordBatch:
    """Equal-length named columns; rows are materialized on demand."""

    def __init__(self, columns: dict[str, list]) -> None:
        self._columns = {k: list(v) for k, v in columns.items()}
        lengths = {len(v) for v in self._columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"ragged batch: column lengths {sorted(lengths)}")

    @property
    def schema(self) -> Schema:
        return Schema([Field(n, "string") for n in self._columns])

    @property
    def num_rows(self) -> int:
        if not self._columns:
            return 0
        return len(next(iter(self._columns.values())))

    def column(self, name: str) -> _Column:
        return _Column(self._columns[name])

    def take(self, indices: list[int]) -> "RecordBatch":
        return RecordBatch(
            {k: [v[i] for i in indices] for k, v in self._columns.items()}
        )

    def to_pylist(self) -> list[dict[str, Any]]:
        names = list(self._columns)
        n = self.num_rows
        return [{k: self._columns[k][i] for k in names} for i in range(n)]


def record_batch(columns: dict[str, list]) -> RecordBatch:
    return RecordBatch(columns)


def array(values: list, type: str | None = None) -> list:  # noqa: A002
    del type
    return list(values)


class Table(RecordBatch):

    @staticmethod
    def from_pylist(rows: list[dict[str, Any]], schema: "Schema | None" = None) -> "Table":
        del schema  # rows are untyped carriers; encode happens in the table layer
        if not rows:
            return Table({})
        names: list[str] = []
        for row in rows:
            for k in row:
                if k not in names:
                    names.append(k)
        return Table({k: [row.get(k) for row in rows] for k in names})


def table(columns: dict[str, list]) -> Table:
    return Table(columns)
