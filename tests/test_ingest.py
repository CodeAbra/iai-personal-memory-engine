"""Hermetic regression tests for the file ingestion pipeline.

Verifies extract_text dispatches by suffix, chunk_text emits ~256-token
windows with ~48 overlap, short-tail merges sub-12-char tails into the
previous chunk, and the tiktoken cl100k encoder is module-cached.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths",
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# Package import + barrel surface
# ---------------------------------------------------------------------------

def test_ingest_package_imports():
    from iai_mcp.ingest import chunk_text, extract_text

    assert callable(extract_text)
    assert callable(chunk_text)


def test_ingest_module_attributes_present():
    m = importlib.import_module("iai_mcp.ingest")
    assert hasattr(m, "extract_text")
    assert hasattr(m, "chunk_text")
    assert set(m.__all__) == {"extract_text", "chunk_text"}


# ---------------------------------------------------------------------------
# pyproject.toml declarations
# ---------------------------------------------------------------------------

def test_pypdf_declared_in_runtime_dependencies():
    """pypdf>=4.0 lives in [project].dependencies (runtime), not an extra."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert '"pypdf>=4.0"' in text
    runtime_block = text.split("[project.optional-dependencies]")[0]
    assert '"pypdf>=4.0"' in runtime_block, (
        "pypdf must be in [project].dependencies (runtime), "
        "not in any optional-dependencies table"
    )


def test_reportlab_declared_in_dev_dependencies():
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert '"reportlab>=4.0"' in text
    dev_block = text.split("[project.optional-dependencies]")[1]
    assert '"reportlab>=4.0"' in dev_block.split("migration")[0], (
        "reportlab must live in [project.optional-dependencies].dev"
    )


# ---------------------------------------------------------------------------
# extract_text — suffix dispatch
# ---------------------------------------------------------------------------

def test_extract_text_unsupported_suffix_raises_value_error(tmp_path):
    from iai_mcp.ingest import extract_text

    with pytest.raises(ValueError, match="unsupported file suffix"):
        extract_text(tmp_path / "foo.unknown")


def test_extract_text_txt_round_trip(tmp_path):
    from iai_mcp.ingest import extract_text

    fixture = tmp_path / "sample.txt"
    content = "The export format is JSONL.\nNo trailing changes.\n"
    fixture.write_text(content, encoding="utf-8")
    assert extract_text(fixture) == content


def test_extract_text_md_round_trip(tmp_path):
    from iai_mcp.ingest import extract_text

    fixture = tmp_path / "notes.md"
    content = "# Heading\n\nbody paragraph.\n"
    fixture.write_text(content, encoding="utf-8")
    assert extract_text(fixture) == content


def test_extract_text_csv_parses_cells(tmp_path):
    from iai_mcp.ingest import extract_text

    fixture = tmp_path / "data.csv"
    fixture.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert extract_text(fixture) == "a,b,c\n1,2,3"


def test_extract_text_pdf_lazy_import_error(tmp_path, monkeypatch):
    """When pypdf is unimportable inside the .pdf branch, raise RuntimeError
    with an actionable pip-install hint.
    """
    from iai_mcp.ingest import extract_text

    monkeypatch.setitem(sys.modules, "pypdf", None)

    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    with pytest.raises(RuntimeError) as exc_info:
        extract_text(pdf_path)
    msg = str(exc_info.value)
    assert "pypdf" in msg
    assert "pip install" in msg


def test_extract_text_pdf_happy_path(_pdf_fixture):
    pytest.importorskip("pypdf")
    from iai_mcp.ingest import extract_text

    out = extract_text(_pdf_fixture)
    assert "This is a test PDF body" in out


def test_extract_text_pdf_malformed_raises_value_error(tmp_path):
    """A malformed or truncated PDF must raise a clean ValueError with a
    descriptive message, not leak a raw pypdf traceback to the caller.
    The CLI handler already catches ValueError; this test verifies the
    parser-level wrapping so that contract holds for all pypdf error types.
    """
    pytest.importorskip("pypdf")
    from iai_mcp.ingest import extract_text

    # Header-only PDF: syntactically invalid, triggers pypdf.errors.PyPdfError.
    malformed = tmp_path / "malformed.pdf"
    malformed.write_bytes(b"%PDF-1.7\n")

    with pytest.raises(ValueError) as exc_info:
        extract_text(malformed)
    msg = str(exc_info.value)
    assert "could not parse PDF" in msg
    assert "malformed.pdf" in msg


def test_extract_text_docx_deferred(tmp_path):
    from iai_mcp.ingest import extract_text

    fixture = tmp_path / "file.docx"
    fixture.write_bytes(b"PK\x03\x04")
    with pytest.raises(NotImplementedError) as exc_info:
        extract_text(fixture)
    msg = str(exc_info.value)
    assert ".docx" in msg
    assert "python-docx" in msg


def test_extract_text_xlsx_deferred(tmp_path):
    from iai_mcp.ingest import extract_text

    fixture = tmp_path / "sheet.xlsx"
    fixture.write_bytes(b"PK\x03\x04")
    with pytest.raises(NotImplementedError) as exc_info:
        extract_text(fixture)
    msg = str(exc_info.value)
    assert ".xlsx" in msg
    assert "openpyxl" in msg


# ---------------------------------------------------------------------------
# chunk_text — invariants
# ---------------------------------------------------------------------------

def test_chunk_text_empty_input_returns_empty_list():
    from iai_mcp.ingest import chunk_text

    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_whole_doc_under_min_returns_empty():
    """Sub-12-char whole doc would silently drop at capture_turn → return []."""
    from iai_mcp.ingest import chunk_text

    assert chunk_text("hi") == []


def test_chunk_text_short_doc_single_chunk():
    from iai_mcp.ingest import chunk_text

    text = "the export format is JSONL."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].startswith("the export format")


def test_chunk_text_window_size_respected():
    """Each chunk's decoded form fits in ~target_tokens (small slack for boundary decode)."""
    import tiktoken

    from iai_mcp.ingest import chunk_text

    enc = tiktoken.get_encoding("cl100k_base")
    text = "word " * 400
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    for ch in chunks:
        assert len(enc.encode(ch)) <= 256 + 5  # cl100k boundary decode allowance


def test_chunk_text_overlap_shares_tokens():
    """Consecutive chunks share tokens (sliding window overlap)."""
    import tiktoken

    from iai_mcp.ingest import chunk_text

    enc = tiktoken.get_encoding("cl100k_base")
    text = "word " * 600
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    ids0 = enc.encode(chunks[0])
    ids1 = enc.encode(chunks[1])
    tail0 = set(ids0[-48:])
    head1 = set(ids1[:48])
    assert tail0 & head1, (
        "Consecutive chunks must share overlap tokens to preserve local context."
    )


def test_chunk_text_short_tail_merge_or_clip():
    """Verbatim/lossless: the last chunk is never both present AND <12 chars."""
    from iai_mcp.ingest import chunk_text

    text = "word " * 260
    chunks = chunk_text(text)
    if len(chunks) >= 2:
        assert len(chunks[-1]) >= 12, (
            "Sub-12-char tail must be merged into the prior chunk."
        )


def test_chunk_text_encoder_module_cached(monkeypatch):
    """tiktoken.get_encoding('cl100k_base') is invoked at most once across N
    chunk_text calls; the encoder is held in module-private state.
    """
    import tiktoken

    import iai_mcp.ingest._chunker as _ck

    monkeypatch.setattr(_ck, "_CL100K", None, raising=True)

    counter = {"n": 0}
    real_get = tiktoken.get_encoding

    def counted(name):
        counter["n"] += 1
        return real_get(name)

    monkeypatch.setattr(tiktoken, "get_encoding", counted)

    _ck.chunk_text("first call text the export format is JSONL." * 4)
    _ck.chunk_text("second call text more content here for chunking.")
    _ck.chunk_text("third call another body of text for the chunker.")

    assert counter["n"] == 1, (
        f"cl100k encoder must be module-cached singleton; "
        f"observed {counter['n']} get_encoding calls across 3 chunk_text invocations."
    )


def test_chunk_text_zero_token_loss_after_sentence_snap():
    """Every token in the source text must appear in at least one chunk window.

    Adversarial fixture: a sentence boundary lands just past the character-
    midpoint of window 0, placing a boundary-free unique-word gap between the
    snap point and the next window start.  The old code (``i += stride``
    unconditionally) dropped tokens in that gap; the fixed code
    (``step = min(base_stride, max(1, consumed - overlap))``) resumes from
    the snap point so the gap is covered by the next window.
    """
    from iai_mcp.ingest._chunker import chunk_text, _encoder

    enc = _encoder()
    # 158 × "aa " pads the first window to ~158 tokens before the boundary.
    # The ". " then satisfies _SENTENCE_BOUNDARY past the char-midpoint of the
    # 256-token window, triggering the snap.  The UNIQ words are unique and
    # boundary-free, so they fall in the gap region when stride is applied
    # unconditionally.
    uniq_words = [f"UNIQ{j}" for j in range(60)]
    text = ("aa " * 158) + ". " + " ".join(uniq_words) + " " + ("bb " * 300)
    chunks = chunk_text(text)
    assert chunks, "expected non-empty chunk list for this input"
    all_chunk_text = " ".join(chunks)
    dropped = [w for w in uniq_words if w not in all_chunk_text]
    assert not dropped, (
        f"chunk_text dropped {len(dropped)} unique word(s) after sentence snap; "
        f"first dropped: {dropped[:5]}. "
        "The sliding window must resume from the snap point, not the fixed stride."
    )
