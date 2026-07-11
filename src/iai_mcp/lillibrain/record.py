"""Varint codec and SQLite serial-type record encoder/decoder.

Implements the SQLite variable-length integer (varint) format used by
the B+tree leaf cell payloads. Compatible with standard SQLite tooling
for debugging.

All functions are pure (no I/O, no module-level state, no logger).
"""
from __future__ import annotations

import struct

# ---------------------------------------------------------------------------
# Varint encode / decode
# ---------------------------------------------------------------------------


def encode_varint(value: int) -> bytes:
    """Encode an integer as a SQLite-compatible varint (1–9 bytes).

    Negative values are masked to 64-bit unsigned before encoding.
    Bytes 1–8 each carry 7 bits of payload (MSB first) with the high
    bit set as a continuation flag. The 9th byte uses all 8 bits and
    carries no continuation flag (the 9-byte case encodes bits 56-63
    in bytes 1-8 as 7-bit groups and bits 7-0 in byte 9).
    """
    if value < 0:
        value &= (1 << 64) - 1

    # Determine byte count
    if value <= 0x7F:
        n = 1
    elif value <= 0x3FFF:
        n = 2
    elif value <= 0x1FFFFF:
        n = 3
    elif value <= 0xFFFFFFF:
        n = 4
    elif value <= 0x7FFFFFFFF:
        n = 5
    elif value <= 0x3FFFFFFFFFF:
        n = 6
    elif value <= 0x1FFFFFFFFFFFF:
        n = 7
    elif value <= 0xFFFFFFFFFFFFFF:
        n = 8
    else:
        n = 9

    buf = bytearray(n)
    if n < 9:
        # N bytes: extract 7-bit groups from LSB up, store MSB-first with continuation bits.
        for i in range(n - 1, -1, -1):
            buf[i] = (value & 0x7F) | (0x80 if i < n - 1 else 0)
            value >>= 7
    else:
        # 9-byte case: last byte gets bits 7-0 (full 8 bits, no continuation).
        buf[8] = value & 0xFF
        value >>= 8
        # Bytes 0-7 each carry 7 bits (MSB first) with continuation bit set.
        for i in range(7, -1, -1):
            buf[i] = (value & 0x7F) | 0x80
            value >>= 7
    return bytes(buf)


def decode_varint(
    data: bytes | bytearray | memoryview,
    offset: int,
) -> tuple[int, int]:
    """Decode a SQLite varint starting at data[offset].

    Returns (value, bytes_consumed). Accepts memoryview for zero-copy use.
    Raises ValueError if the input is too short to contain a valid varint
    (i.e. the buffer ends before the continuation-bit sequence terminates).
    """
    value = 0
    try:
        for i in range(9):
            byte = data[offset + i]
            if i < 8:
                value = (value << 7) | (byte & 0x7F)
                if not (byte & 0x80):
                    return value, i + 1
            else:
                # 9th byte: all 8 bits are data (no continuation bit concept here).
                value = (value << 8) | byte
                return value, 9
    except IndexError as exc:
        raise ValueError("varint: impossible state — buffer too short") from exc
    raise ValueError("varint: impossible state")


# ---------------------------------------------------------------------------
# SQLite serial-type record encoder / decoder
# ---------------------------------------------------------------------------
#
# Record layout (SQLite fileformat2.html §2.1):
#   varint(header_size)                   — total bytes in the header
#   varint(serial_type_1) ...             — one serial-type code per column
#   <body_1> <body_2> ...                 — column data, tightly packed
#
# Serial type codes used here:
#   0          → NULL  (zero body bytes; decodes to SqlNull sentinel)
#   1          → 8-bit integer (body = 1 byte)
#   2          → 16-bit big-endian integer (body = 2 bytes)
#   4          → 32-bit big-endian integer (body = 4 bytes)
#   6          → 64-bit big-endian integer (body = 8 bytes)
#   8          → integer 0 (body = 0 bytes)
#   9          → integer 1 (body = 0 bytes)
#   N≥12, even → TEXT of length (N-12)//2 bytes (UTF-8)
#   N≥13, odd  → blob of length (N-13)//2 bytes


def _serial_type_for(value: object) -> int:
    """Return the SQLite serial-type code for value."""
    # bool is a subclass of int — check before int to avoid encoding True/False
    # as the special 0/1 integer forms (serial types 8/9), since those decode as
    # Python int, not bool. Storing bools as 8-bit integers preserves round-trip
    # identity without any loss of information.
    if isinstance(value, bool):
        return 1  # 8-bit integer: True → 1, False → 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        n = len(value)
        return 13 + 2 * n  # blob serial type
    if isinstance(value, float):
        return 7  # IEEE-754 8-byte big-endian double (serial type 7)
    if isinstance(value, int):
        if value == 0:
            return 8
        if value == 1:
            return 9
        # Serial-type widths are selected by signed magnitude so that the
        # two's-complement bytes written by _body_for and read back with
        # signed=True are byte-identical to SQLite's own representation.
        if -(1 << 7) <= value < (1 << 7):
            return 1   # signed 8-bit  (-128 .. 127)
        if -(1 << 15) <= value < (1 << 15):
            return 2   # signed 16-bit
        if -(1 << 31) <= value < (1 << 31):
            return 4   # signed 32-bit
        return 6       # signed 64-bit
    from iai_mcp.lillibrain.sql.types import SqlNull  # noqa: PLC0415
    if value is SqlNull:
        return 0  # NULL: zero body bytes
    if isinstance(value, str):
        n = len(value.encode())
        return 12 + 2 * n  # TEXT: even serial type >= 12
    import datetime  # noqa: PLC0415
    if isinstance(value, datetime.datetime):
        # A datetime is stored space-form (a space separator at index 10) to
        # match the sqlite3 default adapter, so raw lexicographic TEXT timestamp
        # comparisons agree across the two storage drivers. Must encode the same
        # string as the body site below (sizing == body).
        iso = value.isoformat(sep=" ")
        n = len(iso.encode())
        return 12 + 2 * n  # TEXT
    if isinstance(value, datetime.date):
        # A plain date has no time component, hence no separator: the bare
        # YYYY-MM-DD form, matching the sqlite3 adapter for a date.
        iso = value.isoformat()
        n = len(iso.encode())
        return 12 + 2 * n  # TEXT
    raise ValueError(f"encode_record: unsupported value type {type(value).__name__!r}")


def _body_for(value: object, serial_type: int) -> bytes:
    """Return the body bytes for a value with the given serial_type."""
    if serial_type == 0:
        return b""  # NULL has no body
    if serial_type in (8, 9):
        return b""
    if serial_type == 7:
        assert isinstance(value, float)
        return struct.pack(">d", value)  # IEEE-754 big-endian double
    # datetime → ISO TEXT (same path as str below). A datetime is space-form
    # (space separator at index 10, matching the sqlite3 default adapter); a
    # plain date has no separator. The string MUST match the sizing site above.
    import datetime  # noqa: PLC0415
    if isinstance(value, datetime.datetime):
        return value.isoformat(sep=" ").encode()
    if isinstance(value, datetime.date):
        return value.isoformat().encode()
    if isinstance(value, str):
        return value.encode()  # TEXT: raw UTF-8 bytes
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    # bool is a subclass of int — coerce before the int path
    if isinstance(value, bool):
        v = 1 if value else 0
        # bool never reaches the signed branches (True→1, False→0 are handled
        # by serial types 8/9 above), but write unsigned for safety.
        if serial_type == 1:
            return v.to_bytes(1, "big")
        if serial_type == 2:
            return v.to_bytes(2, "big")
        if serial_type == 4:
            return v.to_bytes(4, "big")
        return v.to_bytes(8, "big")
    assert isinstance(value, int)
    # SQLite serial-type integers are two's-complement big-endian signed.
    # Write signed bytes so that a signed decode recovers the original value.
    if serial_type == 1:
        return value.to_bytes(1, "big", signed=True)
    if serial_type == 2:
        return value.to_bytes(2, "big", signed=True)
    if serial_type == 4:
        return value.to_bytes(4, "big", signed=True)
    # serial_type == 6
    return value.to_bytes(8, "big", signed=True)


def _body_size(serial_type: int) -> int:
    """Return the number of body bytes for a serial_type code."""
    if serial_type == 0:
        return 0  # NULL consumes zero body bytes
    if serial_type in (8, 9):
        return 0
    if serial_type == 1:
        return 1
    if serial_type == 2:
        return 2
    if serial_type == 4:
        return 4
    if serial_type in (6, 7):
        return 8  # 64-bit integer or IEEE-754 double
    if serial_type >= 13 and serial_type % 2 == 1:
        return (serial_type - 13) // 2
    if serial_type >= 12 and serial_type % 2 == 0:
        return (serial_type - 12) // 2
    raise ValueError(f"decode_record: unknown serial_type {serial_type}")


def encode_record(values: list) -> bytes:
    """Encode a list of Python values as a SQLite serial-type record.

    Supported value types: int, bytes. Returns the full record bytes including
    the varint header_size prefix and all serial-type codes.
    """
    serial_types = [_serial_type_for(v) for v in values]

    # Build the header: varint(header_size) + varint(serial_type) * N
    # header_size includes itself.
    type_bytes = b"".join(encode_varint(st) for st in serial_types)
    # header_size field itself: start with a placeholder, then compute.
    # We need to know the length of the header_size varint itself.
    # header_size = len(header_size_varint) + len(type_bytes)
    # Iterate to fixpoint (usually one pass is enough).
    for _ in range(3):
        header_size_varint = encode_varint(len(encode_varint(0)) + len(type_bytes))
        candidate_size = len(header_size_varint) + len(type_bytes)
        header_size_varint = encode_varint(candidate_size)
        if len(header_size_varint) + len(type_bytes) == candidate_size:
            break

    header = header_size_varint + type_bytes
    body = b"".join(_body_for(v, st) for v, st in zip(values, serial_types))
    return header + body


def decode_record_views(
    data: bytes | bytearray | memoryview,
    offset: int,
) -> tuple[list, int]:
    """Decode a SQLite serial-type record, returning BLOB cells as zero-copy slices.

    Identical header/body walk as decode_record, but for BLOB cells the returned
    value is a memoryview slice into *data* when *data* is itself a memoryview,
    avoiding a bytes() copy.  When *data* is plain bytes or bytearray, BLOB cells
    fall back to bytes() copies (so the function is total regardless of input type).

    TEXT cells always decode to str (the bytes→str conversion cannot be avoided).
    Scalar cells (int, float, NULL) are identical to decode_record.

    Return shape: (values, bytes_consumed) — a drop-in replacement for callers that
    want to opt into the zero-copy path.

    Lifetime contract: a returned memoryview slice aliases the input buffer directly.
    The view is valid only as long as that buffer remains live and unchanged.  When
    the buffer is a mapped page slice (e.g. from Pager.read_page with the mapped
    read path ON), the next commit, checkpoint, or file-grow invalidates it because
    the mapping is refreshed.  The consumer must materialise the view (e.g. pass it
    to np.frombuffer or copy it into an index) synchronously before any mutation.
    """
    start = offset

    # Read header_size varint
    header_size, n = decode_varint(data, offset)
    offset += n

    # Read serial-type codes until we've consumed header_size bytes total
    header_end = start + header_size
    serial_types: list[int] = []
    while offset < header_end:
        st, n = decode_varint(data, offset)
        offset += n
        serial_types.append(st)

    # Decode body values
    body_offset = header_end
    values: list = []
    is_mv = isinstance(data, memoryview)
    for st in serial_types:
        sz = _body_size(st)
        if st == 0:
            # NULL column — sz is 0, so body_offset does not advance.
            # Fall through to body_offset += sz (no-op) to keep total_consumed
            # identical to decode_record and decode_record_projected.
            from iai_mcp.lillibrain.sql.types import SqlNull  # noqa: PLC0415
            values.append(SqlNull)
        elif st == 8:
            values.append(0)
        elif st == 9:
            values.append(1)
        elif st == 7:
            # IEEE-754 big-endian double
            (fval,) = struct.unpack_from(">d", data, body_offset)
            values.append(fval)
        elif isinstance(st, int) and st >= 13 and st % 2 == 1:
            # BLOB: return a zero-copy slice when the input is a memoryview;
            # fall back to a bytes copy otherwise so the function is always total.
            if is_mv:
                values.append(data[body_offset : body_offset + sz])
            else:
                values.append(bytes(data[body_offset : body_offset + sz]))
        elif isinstance(st, int) and st >= 12 and st % 2 == 0:
            # TEXT: even serial type >= 12; decode UTF-8 to str
            raw = bytes(data[body_offset : body_offset + sz])
            values.append(raw.decode())
        else:
            # integer: SQLite serial-type integers are two's-complement signed
            raw = bytes(data[body_offset : body_offset + sz])
            values.append(int.from_bytes(raw, "big", signed=True))
        body_offset += sz

    total_consumed = body_offset - start
    return values, total_consumed


def decode_record_projected(
    data: bytes | bytearray | memoryview,
    offset: int,
    col_indices: "set[int] | frozenset[int]",
) -> tuple[dict[int, object], int]:
    """Decode a record, materializing only the columns at the given indices.

    Walks the header exactly as decode_record (varint header size, then the
    serial-type list), then advances body_offset by _body_size(st) for every
    cell, but only materializes cells whose position is in col_indices.
    Unrequested cells are skipped without their body bytes being read or
    decoded — a corrupted unrequested cell does not affect the result.

    Return shape: (values_by_index, bytes_consumed) where values_by_index is
    a dict mapping each requested column index to its decoded value.

    For requested BLOB cells: returns a memoryview slice when data is a
    memoryview (zero-copy), otherwise a bytes copy.
    For requested TEXT/scalar cells: identical to decode_record.

    The function is pure (no I/O, no module state).

    Lifetime contract for returned memoryview slices: identical to
    decode_record_views — the view aliases the input buffer and is valid only
    while that buffer is live and unchanged.
    """
    start = offset

    # Read header_size varint
    header_size, n = decode_varint(data, offset)
    offset += n

    # Read serial-type codes until the full header is consumed
    header_end = start + header_size
    serial_types: list[int] = []
    while offset < header_end:
        st, n = decode_varint(data, offset)
        offset += n
        serial_types.append(st)

    # Walk the body; materialize only the requested column indices.
    body_offset = header_end
    values: dict[int, object] = {}
    is_mv = isinstance(data, memoryview)
    for col_idx, st in enumerate(serial_types):
        sz = _body_size(st)
        if col_idx in col_indices:
            if st == 0:
                from iai_mcp.lillibrain.sql.types import SqlNull  # noqa: PLC0415
                values[col_idx] = SqlNull
            elif st == 8:
                values[col_idx] = 0
            elif st == 9:
                values[col_idx] = 1
            elif st == 7:
                (fval,) = struct.unpack_from(">d", data, body_offset)
                values[col_idx] = fval
            elif st >= 13 and st % 2 == 1:
                # BLOB: zero-copy slice when data is a memoryview
                if is_mv:
                    values[col_idx] = data[body_offset : body_offset + sz]
                else:
                    values[col_idx] = bytes(data[body_offset : body_offset + sz])
            elif st >= 12 and st % 2 == 0:
                # TEXT
                raw = bytes(data[body_offset : body_offset + sz])
                values[col_idx] = raw.decode()
            else:
                # integer: SQLite serial-type integers are two's-complement signed
                raw = bytes(data[body_offset : body_offset + sz])
                values[col_idx] = int.from_bytes(raw, "big", signed=True)
        # Advance past this cell's body bytes regardless of whether it was requested.
        # For NULL (st==0) and zero-body integer types (8,9), sz is 0 and body_offset
        # does not change — which is correct.
        body_offset += sz

    total_consumed = body_offset - start
    return values, total_consumed


def decode_record(
    data: bytes | bytearray | memoryview,
    offset: int,
) -> tuple[list, int]:
    """Decode a SQLite serial-type record starting at data[offset].

    Returns (values, bytes_consumed). Accepts memoryview for zero-copy use.
    """
    start = offset

    # Read header_size varint
    header_size, n = decode_varint(data, offset)
    offset += n

    # Read serial-type codes until we've consumed header_size bytes total
    header_end = start + header_size
    serial_types: list[int] = []
    while offset < header_end:
        st, n = decode_varint(data, offset)
        offset += n
        serial_types.append(st)

    # Decode body values
    body_offset = header_end
    values: list = []
    for st in serial_types:
        sz = _body_size(st)
        if st == 0:
            # NULL column — no body bytes consumed; return the SqlNull sentinel.
            from iai_mcp.lillibrain.sql.types import SqlNull  # noqa: PLC0415
            values.append(SqlNull)
            continue
        if st == 8:
            values.append(0)
        elif st == 9:
            values.append(1)
        elif st == 7:
            # IEEE-754 big-endian double
            (fval,) = struct.unpack_from(">d", data, body_offset)
            values.append(fval)
        elif isinstance(st, int) and st >= 13 and st % 2 == 1:
            # blob
            values.append(bytes(data[body_offset : body_offset + sz]))
        elif isinstance(st, int) and st >= 12 and st % 2 == 0:
            # TEXT: even serial type >= 12; decode UTF-8 to str
            raw = bytes(data[body_offset : body_offset + sz])
            values.append(raw.decode())
        else:
            # integer: SQLite serial-type integers are two's-complement signed
            raw = bytes(data[body_offset : body_offset + sz])
            values.append(int.from_bytes(raw, "big", signed=True))
        body_offset += sz

    total_consumed = body_offset - start
    return values, total_consumed
