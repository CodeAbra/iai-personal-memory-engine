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


#: JSON-RPC error code for embedder-selection refusals (foreign vector
#: space, misconfigured model, identity mismatch). Shared wire contract:
#: the daemon socket classifies the refusal under this code, and the
#: recall CLI and the MCP wrapper key on it first; the message-prose
#: hint survives only as a fallback for a daemon predating the code.
ERR_EMBEDDER_REFUSAL = -32011
