"""Hermetic integration tests for the file-ingestion CLI subcommand.

Covers parser registration, structural anti-pattern guards (the upload
handler MUST NOT route through the daemon socket because the daemon RPC
drops source_uuid, and it MUST NOT touch store.insert directly because
the single capture spine is the only sanctioned write path), clean
error surfaces for unsupported / missing-dependency extensions, and
end-to-end functional behaviour (round-trip, idempotency, boilerplate
dedup, episodic tier, provenance round-trip, on-disk envelope shape).
"""
from __future__ import annotations

import argparse
import ast
import io
import json as _json
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_IAI_CLI_PATH = _REPO_ROOT / "src" / "iai_mcp" / "iai_cli.py"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"


def _make_args(
    path: str,
    *,
    session_id: str | None = None,
    use_json: bool = False,
) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.path = path
    ns.session_id = session_id
    ns.json = use_json
    return ns


def _cmd_upload_source() -> str:
    """Return the source body of the cmd_upload function as a string.

    Slices from the `def cmd_upload` line to the next top-level `def` so
    the textual / regex guards inspect ONLY the upload handler body.
    """
    src = _IAI_CLI_PATH.read_text(encoding="utf-8")
    start = src.find("\ndef cmd_upload(")
    if start == -1:
        return ""
    # Find the next top-level def after the start.
    after = src.find("\ndef ", start + 1)
    return src[start : after if after != -1 else len(src)]


# ---------------------------------------------------------------------------
# Parser & dependency declaration
# ---------------------------------------------------------------------------

def test_cmd_upload_registered():
    """`iai upload <path>` must be a registered subparser whose default
    `func` resolves to the cmd_upload callable, with --json + --session-id
    defaults wired to (False, None) so callers can omit them.
    """
    from iai_mcp.iai_cli import _build_parser

    if not hasattr(__import__("iai_mcp.iai_cli", fromlist=["cmd_upload"]), "cmd_upload"):
        pytest.fail("cmd_upload not yet implemented")

    from iai_mcp.iai_cli import cmd_upload  # noqa: F401 (imported for identity check below)

    parser = _build_parser()
    ns = parser.parse_args(["upload", "x.txt"])
    assert getattr(ns, "func", None) is cmd_upload, (
        "upload subcommand must wire func=cmd_upload"
    )
    assert ns.path == "x.txt"
    assert getattr(ns, "session_id", "missing") is None
    assert getattr(ns, "json", "missing") is False


def test_pypdf_declared_in_pyproject():
    """`pypdf>=4.0` must live in the runtime `[project] dependencies`
    array (NOT in an optional-extras table), because the file-ingestion
    .pdf branch lazy-imports it at runtime.
    """
    text = _PYPROJECT_PATH.read_text(encoding="utf-8")
    # Find the [project] dependencies array. A bare `dependencies =` key
    # exists only in the [project] table (optional-dependencies tables use
    # named extras), so anchoring on the key alone stays correct even when
    # earlier [project] keys hold inline tables/arrays (authors = [{...}]).
    m = re.search(
        r"^dependencies\s*=\s*\[(?P<arr>.*?)^\]",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert m is not None, "[project] dependencies array not found in pyproject.toml"
    arr = m.group("arr")
    assert "pypdf>=4.0" in arr, (
        "pypdf>=4.0 must be declared in [project] dependencies "
        "(not an optional extras table)"
    )


# ---------------------------------------------------------------------------
# Structural anti-pattern guards
# ---------------------------------------------------------------------------

def test_no_parallel_write_path():
    """The upload handler must NOT call store.insert directly — the only
    sanctioned write path is through the capture spine (capture_turn).
    A direct insert would bypass the dedup / idempotency gate and the
    shared provenance shape.

    Enforced by BOTH an AST walk and a textual regex; either signal is
    enough to fail the build.
    """
    body = _cmd_upload_source()
    if not body:
        pytest.fail("cmd_upload not yet implemented")

    # Textual fallback first — catches `store.insert(` regardless of
    # whether the local symbol is `store` or aliased.
    textual_hits = re.findall(r"\bstore\.insert\s*\(", body)
    assert not textual_hits, (
        f"cmd_upload must not call store.insert directly "
        f"(found {len(textual_hits)} hit(s) in body)"
    )

    # AST walk — parse the function body and reject any Call whose
    # function attribute ends in `.insert`.
    tree = ast.parse(body.lstrip())
    upload_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cmd_upload":
            upload_def = node
            break
    assert upload_def is not None, "cmd_upload FunctionDef not found in body slice"

    bad: list[str] = []
    for node in ast.walk(upload_def):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "insert":
                bad.append(ast.unparse(node))
    assert not bad, (
        f"cmd_upload must not invoke `<obj>.insert(...)`; offenders: {bad!r}"
    )


def test_no_daemon_rpc_capture():
    """The upload handler must NOT route chunks through the daemon RPC
    `memory_capture` — that RPC handler at `core/__init__.py` drops
    `source_uuid`, defeating the content-hash idempotency contract.
    Upload is in-process by construction.
    """
    body = _cmd_upload_source()
    if not body:
        pytest.fail("cmd_upload not yet implemented")

    forbidden_double = '_send_jsonrpc_request("memory_capture"'
    forbidden_single = "_send_jsonrpc_request('memory_capture'"
    assert forbidden_double not in body, (
        "cmd_upload must not call _send_jsonrpc_request for memory_capture "
        "(double-quoted variant) — daemon RPC drops source_uuid"
    )
    assert forbidden_single not in body, (
        "cmd_upload must not call _send_jsonrpc_request for memory_capture "
        "(single-quoted variant) — daemon RPC drops source_uuid"
    )


# ---------------------------------------------------------------------------
# Clean-defer and missing-dependency error surfaces
# ---------------------------------------------------------------------------

def test_unsupported_suffix_clean_exit(tmp_path):
    """`.docx` upload exits 1 with a clear stderr message naming the
    deferred suffix and the v1-supported extensions.
    """
    from iai_mcp import iai_cli

    if not hasattr(iai_cli, "cmd_upload"):
        pytest.fail("cmd_upload not yet implemented")

    fixture = tmp_path / "doc.docx"
    fixture.write_bytes(b"PK\x03\x04 stub bytes - a truncated, unparsable container")

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))

    assert rc == 1
    msg = err.getvalue()
    assert "could not parse" in msg.lower()
    assert ".docx" in msg


def test_pypdf_missing_clean_error(monkeypatch, tmp_path):
    """When pypdf is unimportable, a `.pdf` upload exits 1 with stderr
    pointing at the missing module and the pip install hint, instead of
    propagating an opaque ImportError to the user.
    """
    from iai_mcp import iai_cli

    if not hasattr(iai_cli, "cmd_upload"):
        pytest.fail("cmd_upload not yet implemented")

    fixture = tmp_path / "doc.pdf"
    fixture.write_bytes(b"%PDF-1.4\n%stub for suffix dispatch - never parsed")

    # Force the lazy `from pypdf import PdfReader` branch to ImportError.
    monkeypatch.setitem(sys.modules, "pypdf", None)

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))

    assert rc == 1
    msg = err.getvalue()
    assert "pypdf" in msg
    assert "pip install" in msg


def test_empty_file_clean_exit(tmp_path):
    """An empty .txt file exits 1 with a clean "no extractable text" or
    equivalent message — the upload handler must not crash with an
    opaque chunker error or commit zero records silently.
    """
    from iai_mcp import iai_cli

    if not hasattr(iai_cli, "cmd_upload"):
        pytest.fail("cmd_upload not yet implemented")

    fixture = tmp_path / "empty.txt"
    fixture.write_text("", encoding="utf-8")

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))

    assert rc == 1
    msg = err.getvalue().lower()
    # Acceptable phrasings: "no extractable text", "too short", "no chunks"
    assert any(phrase in msg for phrase in (
        "no extractable text", "too short", "no chunks",
    )), f"expected a clean empty-input message in stderr, got: {msg!r}"


def test_pdf_malformed_clean_exit(tmp_path):
    """A structurally-invalid PDF exits 1 with a clean stderr message
    instead of propagating a raw pypdf traceback to the user.
    Mirrors test_empty_file_clean_exit but exercises the PDF parse path.
    """
    pytest.importorskip("pypdf")

    from iai_mcp import iai_cli

    if not hasattr(iai_cli, "cmd_upload"):
        pytest.fail("cmd_upload not yet implemented")

    # Header-only PDF: the bare %PDF-1.7 prefix is syntactically invalid;
    # pypdf raises PyPdfError which the parser now wraps as ValueError.
    fixture = tmp_path / "bad.pdf"
    fixture.write_bytes(b"%PDF-1.7\n")

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))

    assert rc == 1
    msg = err.getvalue()
    assert "upload failed" in msg.lower(), (
        f"expected 'upload failed' in stderr, got: {msg!r}"
    )
    # Must NOT be a raw pypdf traceback (no PdfStreamError / PyPdfError class name).
    assert "PdfStreamError" not in msg and "PyPdfError" not in msg and "PdfReadError" not in msg, (
        f"raw pypdf traceback leaked to user: {msg!r}"
    )


def test_upload_extracted_chars_cap(monkeypatch, tmp_path):
    """When extracted text exceeds IAI_UPLOAD_MAX_CHARS the upload must exit 1
    with a clear 'too large' message and produce zero store writes.
    The cap is set to a low value via env so the test does not need a huge file.
    """
    from iai_mcp import iai_cli

    if not hasattr(iai_cli, "cmd_upload"):
        pytest.fail("cmd_upload not yet implemented")

    # Set a tiny cap so a small file triggers it.
    monkeypatch.setenv("IAI_UPLOAD_MAX_CHARS", "10")

    fixture = tmp_path / "large.txt"
    # 100 chars — over the 10-char cap.
    fixture.write_text("a" * 100, encoding="utf-8")

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))

    assert rc == 1
    msg = err.getvalue().lower()
    assert "too large" in msg, (
        f"expected 'too large' in stderr for oversized extracted text, got: {msg!r}"
    )


def test_upload_chunk_count_cap(monkeypatch, tmp_path):
    """When the number of chunks exceeds IAI_UPLOAD_MAX_CHUNKS the upload must
    exit 1 with a clear 'exceeds' / cap message and produce zero store writes.
    The cap is set to 1 via env so any multi-chunk file triggers it.
    """
    from iai_mcp import iai_cli

    if not hasattr(iai_cli, "cmd_upload"):
        pytest.fail("cmd_upload not yet implemented")

    # A 600-word file produces ~3 chunks with default settings.
    # Cap at 1 to force the limit to fire.
    monkeypatch.setenv("IAI_UPLOAD_MAX_CHUNKS", "1")

    fixture = tmp_path / "many_chunks.txt"
    fixture.write_text("word " * 600, encoding="utf-8")

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))

    assert rc == 1
    msg = err.getvalue().lower()
    assert "cap" in msg or "exceeds" in msg, (
        f"expected cap/exceeds message in stderr, got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Functional integration: round-trip, idempotency, dedup, tier, provenance,
# on-disk envelope inheritance from the shared encrypt chokepoint.
# ---------------------------------------------------------------------------

def _open_store_for_read():
    """Open a MemoryStore over the autouse hermetic root.

    The autouse `_hermetic_default_paths` fixture patches `HOME` and
    `DEFAULT_STORAGE_PATH` to a tmp root; opening with no arg picks it up.
    """
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _read_all_records(store):
    """Return every record in the store via the public `all_records` helper."""
    return list(store.all_records())


def _drop_color(stdout: str) -> str:
    """Strip ANSI color codes so substring asserts work in a TTY-less buf."""
    return re.sub(r"\x1b\[[0-9;]*m", "", stdout)


def test_upload_txt_round_trip(monkeypatch, tmp_path):
    """End-to-end happy path: a small .txt produces non-zero inserts and the
    stored records carry `source:"upload"` provenance reachable via the
    standard store read path.
    """
    from iai_mcp import iai_cli

    fixture = tmp_path / "sample.txt"
    fixture.write_text("the export format is JSONL. " * 100, encoding="utf-8")

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))

    assert rc == 0
    out = _drop_color(buf.getvalue()).lower()
    assert "uploaded" in out
    assert "inserted=" in out
    assert "chunks=" in out

    store = _open_store_for_read()
    try:
        records = _read_all_records(store)
    finally:
        store.close()
    upload_records = [
        r for r in records
        if any(
            isinstance(entry, dict) and entry.get("source") == "upload"
            for entry in (r.provenance or [])
        )
    ]
    assert len(upload_records) >= 1, (
        f"expected >=1 uploaded record in store, got {len(upload_records)}"
    )


def test_re_upload_is_idempotent(monkeypatch, tmp_path):
    """Uploading the same file twice — the second run must insert ZERO new
    records and reinforce every chunk that the first run inserted, courtesy
    of the deterministic session_id + content-hash source_uuid.
    """
    from iai_mcp import iai_cli

    fixture = tmp_path / "idem.txt"
    # Chunks must be mutually DISSIMILAR (distinct topics), not merely
    # SHA-unique: near-identical numbered sentences sail past exact dedup but
    # trip the semantic near-dup gate (cos >= 0.95), which legitimately
    # reinforces instead of inserting — and the cross-run count equality
    # below only holds when every first-run chunk actually inserts.
    topics = [
        "Radio telescopes map hydrogen clouds across the spiral arms of "
        "distant galaxies while astronomers catalog pulsar timing anomalies.",
        "Slow-fermented sourdough depends on hydration ratios, ambient "
        "temperature, and the wild yeast culture living in the starter.",
        "Clay-court tennis rewards topspin and patient rally construction, "
        "with sliding footwork that grass surfaces never allow.",
        "Register allocation in optimizing compilers graph-colors virtual "
        "registers onto a finite set of machine registers between spills.",
        "Drip irrigation delivers water directly to tomato roots, cutting "
        "evaporation losses that overhead sprinklers cannot avoid.",
        "Humpback whales migrate thousands of kilometers between polar "
        "feeding grounds and tropical calving lagoons every single year.",
        "Violin bows are rehaired with horsehair and tensioned by a screw "
        "mechanism hidden inside the ebony frog at the player's hand.",
        "Municipal bond ladders stagger maturities so a retiree harvests "
        "predictable principal repayments without timing interest rates.",
    ]
    elaborations = [
        "The survey teams cross-check interference logs nightly.",
        "Bakers weigh flour and water on calibrated scales at dawn.",
        "Coaches drill drop shots to exploit deep court positioning.",
        "Spill heuristics prefer values with the longest live ranges.",
        "Mulch layers keep the soil temperature stable through July.",
        "Acoustic tags record dive depth along the migration route.",
        "Luthiers test the wedge fit before locking the tension screw.",
        "Brokers quote the yield curve before each rung is purchased.",
    ]
    fixture.write_text(
        " ".join(
            f"Section {i}: {topic} {extra} This concludes section {i}."
            for i, (topic, extra) in enumerate(zip(topics, elaborations))
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf1 = io.StringIO()
    with redirect_stdout(buf1):
        rc1 = iai_cli.cmd_upload(_make_args(str(fixture), use_json=True))
    assert rc1 == 0
    p1 = _json.loads(_drop_color(buf1.getvalue()))
    assert p1["inserted"] > 0, f"first run must insert chunks, got payload={p1!r}"

    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        rc2 = iai_cli.cmd_upload(_make_args(str(fixture), use_json=True))
    assert rc2 == 0
    p2 = _json.loads(_drop_color(buf2.getvalue()))
    assert p2["inserted"] == 0, (
        f"re-upload must NOT insert new chunks, got {p2['inserted']}"
    )
    assert p2["reinforced"] == p1["inserted"], (
        f"re-upload must reinforce every chunk; "
        f"first inserted={p1['inserted']}, second reinforced={p2['reinforced']}"
    )


def test_boilerplate_dedup(monkeypatch, tmp_path):
    """Two files sharing a substantial boilerplate header dedup the
    shared chunks (reinforce on second upload) while inserting the
    unique body chunks. Mirrors the real-world "company header on top of
    every document" use case.
    """
    from iai_mcp import iai_cli

    # Header sized to span at least the first full ~256-token chunk plus
    # the overlap window of the second chunk, so chunk 0 is guaranteed
    # byte-identical across both files regardless of body content.
    header = "COMPANY HEADER. " * 400
    file_a = tmp_path / "doc_a.txt"
    file_a.write_text(header + "first body unique to A. " * 80, encoding="utf-8")
    file_b = tmp_path / "doc_b.txt"
    file_b.write_text(header + "second body unique to B. " * 80, encoding="utf-8")

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf_a = io.StringIO()
    with redirect_stdout(buf_a):
        rc_a = iai_cli.cmd_upload(_make_args(str(file_a), use_json=True))
    assert rc_a == 0
    p_a = _json.loads(_drop_color(buf_a.getvalue()))
    assert p_a["inserted"] > 0

    buf_b = io.StringIO()
    with redirect_stdout(buf_b):
        rc_b = iai_cli.cmd_upload(_make_args(str(file_b), use_json=True))
    assert rc_b == 0
    p_b = _json.loads(_drop_color(buf_b.getvalue()))
    assert p_b["reinforced"] > 0, (
        f"file B must reinforce at least the shared header chunk; "
        f"payload={p_b!r}"
    )
    assert p_b["inserted"] > 0, (
        f"file B must insert at least one unique-body chunk; payload={p_b!r}"
    )


def test_provenance_round_trip(monkeypatch, tmp_path):
    """The encrypted provenance column round-trips: the second entry on
    every uploaded chunk's stored provenance is the `{source, filename,
    chunk_index}` dict supplied by the upload handler.
    """
    from iai_mcp import iai_cli

    fixture = tmp_path / "prov.txt"
    fixture.write_text("the export format is JSONL. " * 100, encoding="utf-8")

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))
    assert rc == 0

    store = _open_store_for_read()
    try:
        records = _read_all_records(store)
    finally:
        store.close()

    upload_records = [
        r for r in records
        if any(
            isinstance(entry, dict) and entry.get("source") == "upload"
            for entry in (r.provenance or [])
        )
    ]
    assert upload_records, "expected at least one record with source=upload"

    for rec in upload_records:
        prov = rec.provenance
        assert isinstance(prov, list)
        assert len(prov) == 2, f"expected 2-entry provenance, got {len(prov)}: {prov!r}"
        spine, upload_tag = prov[0], prov[1]
        assert "ts" in spine and "session_id" in spine and "role" in spine, (
            f"spine entry missing required keys: {spine!r}"
        )
        assert upload_tag.get("source") == "upload"
        assert upload_tag.get("filename") == fixture.name
        assert isinstance(upload_tag.get("chunk_index"), int)
        assert upload_tag["chunk_index"] >= 0


def test_tier_is_episodic(monkeypatch, tmp_path):
    """Every uploaded chunk lands as `tier="episodic"`. Filter by the
    upload provenance tag to scope to records this test created.
    """
    from iai_mcp import iai_cli

    fixture = tmp_path / "tier.txt"
    fixture.write_text("the export format is JSONL. " * 100, encoding="utf-8")

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))
    assert rc == 0

    store = _open_store_for_read()
    try:
        records = _read_all_records(store)
    finally:
        store.close()

    upload_records = [
        r for r in records
        if any(
            isinstance(entry, dict) and entry.get("source") == "upload"
            for entry in (r.provenance or [])
        )
    ]
    assert upload_records
    for rec in upload_records:
        assert rec.tier == "episodic", (
            f"uploaded record has tier={rec.tier!r}, expected 'episodic'"
        )


def test_uploaded_chunks_are_zstd_compressed(monkeypatch, tmp_path):
    """The shared crypto.encrypt_field chokepoint wraps every fresh write
    with a zstd version-byte envelope. Verify that an uploaded chunk's
    on-disk `provenance_json` column carries the outer wire prefix and
    that decrypt yields a JSON list (the inner version byte routing is
    handled by decrypt_field itself).
    """
    from tests._store_raw import open_store_raw

    try:
        import zstandard  # noqa: F401
    except ImportError:
        pytest.skip("zstandard not installed — codec test inapplicable")

    from iai_mcp import iai_cli
    from iai_mcp.crypto import CIPHERTEXT_PREFIX, decrypt_field

    fixture = tmp_path / "zstd.txt"
    fixture.write_text("the export format is JSONL. " * 100, encoding="utf-8")

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_upload(_make_args(str(fixture)))
    assert rc == 0

    store = _open_store_for_read()
    try:
        # Locate one uploaded record id (in-process).
        records = _read_all_records(store)
        upload_records = [
            r for r in records
            if any(
                isinstance(entry, dict) and entry.get("source") == "upload"
                for entry in (r.provenance or [])
            )
        ]
        assert upload_records
        target_id = str(upload_records[0].id).lower()
        key = store._key()
        db_path = store.db._hippo_dir / "brain.sqlite3"
    finally:
        store.close()

    # Read the raw provenance_json column directly from the store, bypassing
    # the store's decrypt path so we can assert the on-disk wire prefix.
    conn = open_store_raw(db_path)
    try:
        row = conn.execute(
            "SELECT id, provenance_json FROM records WHERE id = ?",
            (target_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"record {target_id} not found on disk"
    rec_id, prov_ct = row
    assert isinstance(prov_ct, str)
    assert prov_ct.startswith(CIPHERTEXT_PREFIX), (
        f"on-disk provenance_json must carry the {CIPHERTEXT_PREFIX!r} wire "
        f"prefix; got: {prov_ct[:40]!r}"
    )

    # Decrypt round-trips through the codec; the result is a JSON list of
    # provenance entries. (decrypt_field handles the inner version byte.)
    plain = decrypt_field(prov_ct, key, associated_data=str(rec_id).lower().encode("ascii"))
    assert plain.startswith("["), (
        f"decrypted provenance must be a JSON list, got: {plain[:40]!r}"
    )
    parsed = _json.loads(plain)
    assert isinstance(parsed, list)
    assert any(
        isinstance(e, dict) and e.get("source") == "upload" for e in parsed
    )
