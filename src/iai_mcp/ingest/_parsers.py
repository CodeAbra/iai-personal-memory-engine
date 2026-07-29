"""Pure-function text extraction dispatch by file suffix.

Stdlib-only at module-top; heavy / optional parsers (currently pypdf for .pdf)
are lazy-imported inside their respective branches so a slimmed install can
still ingest .txt / .md / .csv without pulling the full parser stack.
Office containers (.docx / .pptx / .xlsx / .epub) are zip+XML and parsed with
the stdlib alone.
"""
from __future__ import annotations

import csv
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree


PLAINTEXT_SUFFIXES: "frozenset[str]" = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".tex", ".bib",
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs",
    ".sh", ".bash", ".zsh", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".sql",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
    ".html", ".css", ".scss", ".xml", ".proto",
})

OFFICE_SUFFIXES: "frozenset[str]" = frozenset({
    ".docx", ".pptx", ".xlsx", ".rtf", ".epub",
})

SUPPORTED_SUFFIXES: "frozenset[str]" = (
    PLAINTEXT_SUFFIXES | OFFICE_SUFFIXES | frozenset({".csv", ".pdf"})
)

# Decompressed-size cap per archive member: the central directory's declared
# size can lie, so the guard reads through the stream, not the header.
_ZIP_MEMBER_CAP = 64 * 1024 * 1024


def _zip_member_text(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name) as fh:
        data = fh.read(_ZIP_MEMBER_CAP + 1)
        if len(data) > _ZIP_MEMBER_CAP:
            raise ValueError(f"archive member {name} exceeds the size cap")
    return data.decode("utf-8", errors="replace")


_DTD_MARKER_RE = re.compile(r"<!(?:DOCTYPE|ENTITY)", re.IGNORECASE)


def _safe_xml_fromstring(xml: str) -> "ElementTree.Element":
    # Legitimate OOXML never declares a DTD; refusing <!DOCTYPE / <!ENTITY
    # outright removes the XXE and entity-expansion attack surface that
    # stdlib XML parsers otherwise carry.
    if _DTD_MARKER_RE.search(xml):
        raise ValueError("XML carrying DTD declarations is refused")
    return ElementTree.fromstring(xml)


def _xml_paragraph_text(xml: str) -> str:
    """Paragraph-per-line text of a WordprocessingML / DrawingML body:
    ``<w:p>`` / ``<a:p>`` group runs, ``<w:t>`` / ``<a:t>`` carry text."""
    root = _safe_xml_fromstring(xml)
    paragraphs: list[str] = []
    for node in root.iter():
        if not node.tag.endswith("}p"):
            continue
        runs = [
            t.text for t in node.iter()
            if t.tag.endswith("}t") and t.text
        ]
        if runs:
            paragraphs.append("".join(runs))
    return "\n".join(paragraphs)


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        return _xml_paragraph_text(_zip_member_text(zf, "word/document.xml"))


def _pptx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        slides = sorted(
            (
                n for n in zf.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)
            ),
            key=lambda n: int(re.search(r"slide(\d+)\.xml", n).group(1)),
        )
        parts = [
            _xml_paragraph_text(_zip_member_text(zf, name)) for name in slides
        ]
    return "\n".join(p for p in parts if p)


def _xlsx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        if "xl/sharedStrings.xml" not in zf.namelist():
            return ""
        xml = _zip_member_text(zf, "xl/sharedStrings.xml")
    root = _safe_xml_fromstring(xml)
    lines: list[str] = []
    for si in root.iter():
        if not si.tag.endswith("}si"):
            continue
        runs = [
            t.text for t in si.iter()
            if t.tag.endswith("}t") and t.text
        ]
        if runs:
            lines.append("".join(runs))
    return "\n".join(lines)


class _HTMLTextCollector(HTMLParser):
    _BREAK_TAGS = frozenset({
        "p", "div", "br", "li", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppress = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._suppress += 1
        if tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._suppress:
            self._suppress -= 1
        if tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._suppress:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [ln.strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _epub_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        docs = sorted(
            n for n in zf.namelist()
            if n.lower().endswith((".xhtml", ".html", ".htm"))
        )
        chunks: list[str] = []
        for name in docs:
            collector = _HTMLTextCollector()
            collector.feed(_zip_member_text(zf, name))
            text = collector.text()
            if text:
                chunks.append(text)
    return "\n".join(chunks)


_RTF_SKIP_DESTINATIONS = frozenset({
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "themedata",
    "generator", "xmlnstbl", "filetbl", "listtable", "listoverridetable",
})

_RTF_CONTROL_RE = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])")


def _rtf_text(data: str) -> str:
    out: list[str] = []
    i = 0
    n = len(data)
    uc_skip = 1
    while i < n:
        ch = data[i]
        if ch == "{" or ch == "}":
            if ch == "{":
                m = re.match(r"\{\\\*?\\?([a-zA-Z]+)", data[i:])
                if m and m.group(1) in _RTF_SKIP_DESTINATIONS:
                    depth = 0
                    while i < n:
                        if data[i] == "{":
                            depth += 1
                        elif data[i] == "}":
                            depth -= 1
                            if depth == 0:
                                break
                        i += 1
            i += 1
            continue
        if ch == "\\":
            m = _RTF_CONTROL_RE.match(data, i)
            if m is None:
                i += 1
                continue
            word, param, hexcode, escaped = m.groups()
            if hexcode is not None:
                out.append(bytes([int(hexcode, 16)]).decode("cp1252", errors="replace"))
            elif escaped is not None:
                if escaped in ("\\", "{", "}"):
                    out.append(escaped)
                elif escaped == "~":
                    out.append("\u00a0")
            elif word in ("par", "line", "sect", "page"):
                out.append("\n")
            elif word == "tab":
                out.append("\t")
            elif word == "uc" and param is not None:
                uc_skip = max(0, int(param))
            elif word == "u" and param is not None:
                code = int(param)
                out.append(chr(code + 0x10000 if code < 0 else code))
                skip = uc_skip
                j = m.end()
                while skip > 0 and j < n:
                    if data[j] == "\\" and j + 3 < n and data[j + 1] == "'":
                        j += 4
                    else:
                        j += 1
                    skip -= 1
                i = j
                continue
            i = m.end()
            continue
        if ch not in ("\r", "\n"):
            out.append(ch)
        i += 1
    text = "".join(out)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in PLAINTEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            return "\n".join(",".join(row) for row in csv.reader(f))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is required for .pdf upload. "
                "Install with: pip install 'pypdf>=4.0'"
            ) from exc
        try:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except PyPdfError as exc:
            raise ValueError(f"could not parse PDF {path.name}: {exc}") from exc
    if suffix == ".rtf":
        return _rtf_text(path.read_text(encoding="latin-1", errors="replace"))
    if suffix in (".docx", ".pptx", ".xlsx", ".epub"):
        parser = {
            ".docx": _docx_text,
            ".pptx": _pptx_text,
            ".xlsx": _xlsx_text,
            ".epub": _epub_text,
        }[suffix]
        try:
            return parser(path)
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
            raise ValueError(
                f"could not parse {suffix} {path.name}: {exc}"
            ) from exc
    raise ValueError(f"unsupported file suffix: {suffix}")
