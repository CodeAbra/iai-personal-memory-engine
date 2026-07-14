from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = os.path.expanduser("~/.iai-mcp")


def _store_path() -> Path:
    return Path(os.environ.get("IAI_MCP_STORE", DEFAULT_STORE_PATH))


def export_jsonl(output: Path | None = None) -> Path:
    from iai_mcp.store import MemoryStore

    store_dir = _store_path()
    store = MemoryStore(str(store_dir))

    if output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = store_dir / f"export-{ts}.jsonl"

    records = store.all_records()
    count = 0
    with open(output, "w") as f:
        for rec in records:
            entry = {
                "id": str(rec.id),
                "tier": rec.tier,
                "literal_surface": rec.literal_surface,
                "aaak_index": rec.aaak_index,
                "community_id": rec.community_id,
                "centrality": rec.centrality,
                "detail_level": rec.detail_level,
                "pinned": rec.pinned,
                "stability": rec.stability,
                "difficulty": rec.difficulty,
                "created_at": rec.created_at.isoformat() if rec.created_at else None,
                "updated_at": rec.updated_at.isoformat() if rec.updated_at else None,
                "last_reviewed": rec.last_reviewed.isoformat() if rec.last_reviewed else None,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d records to %s", count, output)
    return output


def backup(output: Path | None = None) -> Path:
    store_dir = _store_path()

    if output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = store_dir / f"brain-backup-{ts}.tar.gz"

    items_to_include: "list[tuple[Path, str]]" = []

    for name in [
        ".crypto.key",
        "config.json",
        "lifecycle_state.json",
        ".daemon-state.json",
    ]:
        p = store_dir / name
        if p.exists():
            items_to_include.append((p, name))

    # The memory database itself: everything under hippo/ except what is
    # rebuilt from the DB at boot (RO snapshot, ANN index, column index),
    # transient locks, and temp litter. The DB travels together with its
    # WAL/SHM so a live-store snapshot stays crash-consistent — recovery
    # replays it exactly like a power loss. Globbing the directory instead
    # of hardcoding file names means an engine file rename can never again
    # silently drop the database from the backup set.
    hippo_dir = store_dir / "hippo"
    if hippo_dir.exists():
        for p in sorted(hippo_dir.iterdir()):
            name = p.name
            if name == ".lock" or name.startswith("tmp"):
                continue
            if name.endswith((".ro", ".hnsw", ".colindex")):
                continue
            items_to_include.append((p, f"hippo/{name}"))

    for bank_name in ("bank", ".memory-bank"):
        bank_dir = store_dir / bank_name
        if bank_dir.exists():
            items_to_include.append((bank_dir, bank_name))

    if not any(arc.startswith("hippo/brain.") for _, arc in items_to_include):
        logger.warning(
            "backup: no database file found under %s — the archive holds "
            "config/key material only",
            hippo_dir,
        )

    with tarfile.open(str(output), "w:gz") as tar:
        for full_path, arcname in items_to_include:
            tar.add(str(full_path), arcname=arcname)

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("Backup created: %s (%.1f MB, %d items)", output, size_mb, len(items_to_include))
    return output


def restore(archive: Path, target: Path | None = None) -> Path:
    if target is None:
        target = _store_path()

    if target.exists() and any(target.iterdir()):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        pre_restore = target.parent / f".pre-restore-{ts}"
        shutil.move(str(target), str(pre_restore))
        logger.info("Existing data moved to %s", pre_restore)

    target.mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(archive), "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target / member.name).resolve()
            if not str(member_path).startswith(str(target.resolve())):
                raise ValueError(f"Path traversal detected in archive: {member.name}")
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(member_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    logger.info("Restored brain from %s to %s", archive, target)
    return target
