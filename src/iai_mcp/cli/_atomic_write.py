"""Shared atomic text-file write for the host config installers.

Readers and a mid-write crash must only ever see the old content or the
new content, never a partial write -- tmp file in the same dir + os.replace.
"""
from __future__ import annotations

import os
from pathlib import Path


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.parent / f"{path.name}.tmp{os.getpid()}"
    try:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            try:
                os.chmod(tmp, os.stat(path).st_mode & 0o777)
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
