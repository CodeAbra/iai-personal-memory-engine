from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID


import iai_mcp.concurrency
import iai_mcp.daemon_state
import iai_mcp.hippo
import iai_mcp.store
from iai_mcp.store import MemoryStore

from _recall_helpers import _deterministic_vec, _make_gold_record


def _under(child: Path, ancestor: Path) -> bool:
    child = Path(child).resolve()
    ancestor = Path(ancestor).resolve()
    return child == ancestor or ancestor in child.parents


def test_fixture_does_not_set_iai_mcp_store() -> None:
    assert os.environ.get("IAI_MCP_STORE") is None


def test_home_and_socket_under_tmp() -> None:
    home = Path(os.environ["HOME"])
    real_home = iai_mcp.hippo._REAL_IAI_ROOT.parent
    assert home != real_home
    sock = Path(os.environ["IAI_DAEMON_SOCKET_PATH"])
    assert _under(sock, home)


def test_four_default_constants_redirected_to_tmp() -> None:
    real_root = iai_mcp.hippo._REAL_IAI_ROOT
    fake_root = Path(os.environ["HOME"]) / ".iai-mcp"

    assert iai_mcp.hippo._DEFAULT_IAI_ROOT == fake_root
    assert iai_mcp.hippo._DEFAULT_IAI_ROOT != real_root

    assert iai_mcp.store.DEFAULT_STORAGE_PATH == fake_root
    assert iai_mcp.store.DEFAULT_STORAGE_PATH != real_root

    assert _under(iai_mcp.concurrency.SOCKET_PATH, fake_root)
    assert _under(iai_mcp.daemon_state.STATE_PATH, fake_root)


def test_bare_store_resolves_under_tmp_default() -> None:
    fake_root = Path(os.environ["HOME"]) / ".iai-mcp"
    store = MemoryStore()
    assert _under(store.root, fake_root)
    assert store.root != iai_mcp.hippo._REAL_IAI_ROOT

    rec = _make_gold_record(1, _deterministic_vec(1))
    store.insert(rec)
    got = store.get(UUID(int=1))
    assert got is not None
    assert got.id == UUID(int=1)


def test_path_store_is_consistent(tmp_path: Path) -> None:
    store = MemoryStore(path=tmp_path)
    assert store.root == tmp_path
    assert store.db._store_root == tmp_path
    assert _under(store.db._hippo_dir, tmp_path)
