"""Storage exception hierarchy (PEP 249 shaped).

The native engine raises these classes directly, and the legacy stdlib
driver's connection wrapper translates its exceptions into them, so every
runtime `except` matches ONE hierarchy regardless of driver. The names
follow the DB-API because they name the same contract — a parse/transient
fault is Operational, disk damage is Database, a bind fault is Programming,
a constraint violation is Integrity.

Catching discipline: `except OperationalError` must NOT swallow disk
damage, so corruption raises the bare `DatabaseError` parent — mirror that
split in new code.
"""

from __future__ import annotations


class Error(Exception):
    pass


class DatabaseError(Error):
    pass


class OperationalError(DatabaseError):
    pass


class IntegrityError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


class DataError(DatabaseError):
    pass


class InterfaceError(Error):
    pass
