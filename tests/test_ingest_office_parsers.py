from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from iai_mcp.ingest._parsers import (
    OFFICE_SUFFIXES,
    PLAINTEXT_SUFFIXES,
    SUPPORTED_SUFFIXES,
    extract_text,
)

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _write_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def _docx(path: Path, paragraphs: list[list[str]]) -> Path:
    body = "".join(
        "<w:p>" + "".join(f"<w:r><w:t>{run}</w:t></w:r>" for run in para) + "</w:p>"
        for para in paragraphs
    )
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{_W_NS}"><w:body>{body}</w:body></w:document>'
    )
    return _write_zip(path, {"word/document.xml": xml})


def test_docx_paragraphs_and_split_runs(tmp_path: Path) -> None:
    p = _docx(
        tmp_path / "doc.docx",
        [["Hello ", "world"], ["Second paragraph"]],
    )
    assert extract_text(p) == "Hello world\nSecond paragraph"


def test_pptx_slides_in_numeric_order(tmp_path: Path) -> None:
    def slide(text: str) -> str:
        return (
            f'<p:sld xmlns:a="{_A_NS}" '
            f'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:sld>"
        )

    p = _write_zip(
        tmp_path / "deck.pptx",
        {
            "ppt/slides/slide10.xml": slide("tenth"),
            "ppt/slides/slide2.xml": slide("second"),
            "ppt/slides/slide1.xml": slide("first"),
        },
    )
    assert extract_text(p) == "first\nsecond\ntenth"


def test_xlsx_shared_strings(tmp_path: Path) -> None:
    xml = (
        f'<sst xmlns="{_S_NS}">'
        "<si><t>alpha</t></si>"
        "<si><r><t>be</t></r><r><t>ta</t></r></si>"
        "</sst>"
    )
    p = _write_zip(tmp_path / "book.xlsx", {"xl/sharedStrings.xml": xml})
    assert extract_text(p) == "alpha\nbeta"


def test_xlsx_without_shared_strings_is_empty(tmp_path: Path) -> None:
    p = _write_zip(tmp_path / "numbers.xlsx", {"xl/workbook.xml": "<workbook/>"})
    assert extract_text(p) == ""


def test_epub_html_chapters_stripped(tmp_path: Path) -> None:
    chapter = (
        "<html><head><style>p { color: red }</style>"
        "<script>alert(1)</script></head>"
        "<body><h1>Title</h1><p>One sentence.</p><p>Two.</p></body></html>"
    )
    p = _write_zip(
        tmp_path / "book.epub",
        {"OEBPS/ch1.xhtml": chapter, "mimetype": "application/epub+zip"},
    )
    assert extract_text(p) == "Title\nOne sentence.\nTwo."


def test_rtf_controls_escapes_and_destinations(tmp_path: Path) -> None:
    rtf = (
        r"{\rtf1\ansi{\fonttbl{\f0 Calibri;}}"
        r"\f0 Caf\'e9 costs \u8364? 5\par second line\par}"
    )
    p = tmp_path / "note.rtf"
    p.write_text(rtf, encoding="latin-1")
    assert extract_text(p) == "Café costs € 5\nsecond line"


def test_tex_treated_as_plaintext(tmp_path: Path) -> None:
    p = tmp_path / "paper.tex"
    p.write_text(r"\section{Intro} body text", encoding="utf-8")
    assert "body text" in extract_text(p)


def test_dtd_carrying_xml_is_refused(tmp_path: Path) -> None:
    xml = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY a "b">]>'
        f'<w:document xmlns:w="{_W_NS}"><w:body>'
        "<w:p><w:r><w:t>&a;</w:t></w:r></w:p></w:body></w:document>"
    )
    p = _write_zip(tmp_path / "evil.docx", {"word/document.xml": xml})
    with pytest.raises(ValueError, match="DTD"):
        extract_text(p)


def test_oversize_member_is_refused(tmp_path: Path, monkeypatch) -> None:
    import iai_mcp.ingest._parsers as parsers

    monkeypatch.setattr(parsers, "_ZIP_MEMBER_CAP", 64)
    big = "x" * 200
    p = _write_zip(
        tmp_path / "bomb.docx",
        {"word/document.xml": f"<doc>{big}</doc>"},
    )
    with pytest.raises(ValueError, match="size cap"):
        extract_text(p)


def test_corrupt_container_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "broken.docx"
    p.write_bytes(b"this is not a zip archive")
    with pytest.raises(ValueError):
        extract_text(p)


def test_supported_suffixes_cover_office_and_core() -> None:
    assert OFFICE_SUFFIXES <= SUPPORTED_SUFFIXES
    assert PLAINTEXT_SUFFIXES <= SUPPORTED_SUFFIXES
    assert {".csv", ".pdf", ".tex", ".bib"} <= SUPPORTED_SUFFIXES
