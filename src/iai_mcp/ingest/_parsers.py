"""Pure-function text extraction dispatch by file suffix.

Stdlib-only at module-top; heavy / optional parsers (currently pypdf for .pdf)
are lazy-imported inside their respective branches so a slimmed install can
still ingest .txt / .md / .csv without pulling the full parser stack.
"""
from __future__ import annotations

import csv
from pathlib import Path


PLAINTEXT_SUFFIXES: "frozenset[str]" = frozenset({
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs",
    ".sh", ".bash", ".zsh", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".sql",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
    ".html", ".css", ".scss", ".xml", ".proto",
})


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
    if suffix in (".docx", ".xlsx"):
        raise NotImplementedError(
            f"{suffix} upload is deferred. v1 supports .txt, .md, .csv, .pdf. "
            f".docx needs python-docx, .xlsx needs openpyxl (not in v1 deps)."
        )
    raise ValueError(f"unsupported file suffix: {suffix}")
