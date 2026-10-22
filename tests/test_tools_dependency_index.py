"""End-to-end coverage for package-scoped dependency indexing across the
Python, Node, and Rust ecosystems: resolving one installed package
(subprocessing the TARGET interpreter for Python, a local filesystem walk
for Node/Rust), indexing it into a store separate from the project source
store, and searching that store.

`code_search`/`iai-tools` resolve their default store under `Path.home()`,
bound at IMPORT time -- an in-process `store_root=` override cannot reach
that path. The isolation test below runs the real CLI as a subprocess
with `HOME` pointed at a fresh temp directory, mirroring
tests/test_tools_xref_reverse_query.py's hermetic pattern. Tests that only
exercise an explicit `root=`/`store_root=` override (never the default
store resolution) run in-process for speed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
import venv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# CLI subprocess helpers (mirrors test_tools_xref_reverse_query.py)
# --------------------------------------------------------------------------


def _cli_env(home_dir: Path) -> dict:
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(home_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = _cli_env(home_dir)
    return subprocess.run(
        [sys.executable, "-m", "iai_mcp.tools_cli", *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_cli_json(home_dir: Path, *args: str) -> dict:
    proc = _run_cli(home_dir, *args, "--json")
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _run_iai_mcp_cli(home_dir: Path, *args: str) -> subprocess.CompletedProcess:
    env = _cli_env(home_dir)
    return subprocess.run(
        [sys.executable, "-m", "iai_mcp.cli", *args],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


# --------------------------------------------------------------------------
# Synthetic Python project + venv helpers
# --------------------------------------------------------------------------


def _venv_site_packages(venv_dir: Path) -> Path:
    candidates = list(venv_dir.glob("lib/python*/site-packages"))
    assert candidates, f"no site-packages directory found under {venv_dir}"
    return candidates[0]


def _make_project_with_venv(tmp_path: Path) -> Path:
    project_dir = tmp_path / f"project-{uuid.uuid4().hex[:8]}"
    project_dir.mkdir()
    venv.EnvBuilder(with_pip=False, symlinks=True).create(str(project_dir / ".venv"))
    return project_dir


def _add_package(
    project_dir: Path, pkg_name: str, extra_files: "dict[str, str] | None" = None
) -> Path:
    site_packages = _venv_site_packages(project_dir / ".venv")
    pkg_dir = site_packages / pkg_name
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    for rel, content in (extra_files or {}).items():
        path = pkg_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return pkg_dir


# -- Isolation: dependency indexing must never touch the source store -----


def test_dependency_index_does_not_touch_source_store(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Single-token markers with no shared substrings once tokenized: an
    # underscore-joined name like SOURCE_MARKER_TOKEN/DEP_MARKER_TOKEN would
    # split into shared sub-tokens ("marker", "token") and falsely appear
    # to leak across stores.
    source_marker = "srcmarker" + uuid.uuid4().hex
    dep_marker = "depmarker" + uuid.uuid4().hex
    (corpus / "widget.py").write_text(f"{source_marker} = 1\n")

    proc = _run_cli(home_dir, "index", str(corpus))
    assert proc.returncode == 0, proc.stderr

    before = _run_cli_json(home_dir, "search", source_marker)
    assert before["hits"], "the source store must find its own marker before any dependency index"

    project_dir = _make_project_with_venv(tmp_path)
    _add_package(project_dir, "alicelib", {"core.py": f"{dep_marker} = 1\n"})

    dep_proc = _run_cli(
        home_dir,
        "index",
        "--ecosystem", "python",
        "--package", "alicelib",
        "--project-dir", str(project_dir),
    )
    assert dep_proc.returncode == 0, dep_proc.stderr

    after = _run_cli_json(home_dir, "search", source_marker)
    assert after["hits"] == before["hits"], (
        "source-store hits must be byte-identical before vs after a dependency index"
    )
    assert after["scope"] == before["scope"], (
        "source-store manifest scope (n_files, generation) must be unchanged -- "
        "a changed generation means the dependency index touched the wrong store"
    )

    dep_leak = _run_cli_json(
        home_dir, "search", source_marker,
        "--store", "dependency", "--ecosystem", "python", "--package", "alicelib",
    )
    assert dep_leak["hits"] == [], "the dependency store must never contain the project source marker"


# -- Resolution: must run in the TARGET interpreter, not this process's ---


def test_resolve_python_package_in_target_interpreter(tmp_path: Path) -> None:
    from iai_mcp.tools.dependency import resolve_package_location

    project_dir = _make_project_with_venv(tmp_path)
    pkg_dir = _add_package(project_dir, "alicelib")

    resolved = resolve_package_location("python", "alicelib", project_dir)
    assert resolved is not None
    assert resolved.resolve() == pkg_dir.resolve()

    missing = resolve_package_location(
        "python", "definitely_not_installed_anywhere_xyz", project_dir
    )
    assert missing is None, (
        "a package absent from the target env must return a clean not-found, "
        "never a wrong-env hit resolved from this test process's own interpreter"
    )


# -- Package-scoped search: one store never leaks another package's hits --


def test_dependency_search_scoped_to_one_package(tmp_path: Path) -> None:
    from iai_mcp.tools import dependency_store_root_for, search
    from iai_mcp.tools.dependency import index_dependency

    alice_marker = "alicemarker" + uuid.uuid4().hex
    bob_marker = "bobmarker" + uuid.uuid4().hex

    project_dir = _make_project_with_venv(tmp_path)
    _add_package(project_dir, "alicelib", {"core.py": f"{alice_marker} = 1\n"})
    _add_package(project_dir, "boblib", {"core.py": f"{bob_marker} = 1\n"})

    deps_root = tmp_path / "deps"
    report_a = index_dependency("python", "alicelib", project_dir, root=deps_root)
    report_b = index_dependency("python", "boblib", project_dir, root=deps_root)
    assert report_a["resolved"] and report_b["resolved"]

    alice_store = dependency_store_root_for("python", "alicelib", root=deps_root)
    bob_store = dependency_store_root_for("python", "boblib", root=deps_root)

    alice_hits = search(alice_marker, store_root=alice_store)
    assert alice_hits, "alicelib's own store must return its own marker"

    cross_hits = search(bob_marker, store_root=alice_store)
    assert cross_hits == [], "alicelib's store must never return boblib's content"

    bob_hits = search(bob_marker, store_root=bob_store)
    assert bob_hits, "boblib's own store must return its own marker"


# -- Not-indexed signal: distinguishable from a genuine zero-hit search ---


def test_dependency_search_reports_not_indexed(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    result = _run_cli_json(
        home_dir, "search", "anything",
        "--store", "dependency", "--ecosystem", "python", "--package", "neverindexed",
    )
    assert result["status"] == "not_indexed"
    assert result["hits"] == []


# -- Truncation surfaced: cap breach reported, never silently dropped ------


def test_dependency_index_reports_truncated_when_cap_reached(
    tmp_path: Path, monkeypatch
) -> None:
    from iai_mcp.tools.dependency import DEPENDENCY_INDEX_MAX_FILES_ENV, index_dependency

    project_dir = _make_project_with_venv(tmp_path)
    _add_package(
        project_dir,
        "alicelib",
        {f"mod_{i}.py": f"VALUE_{i} = {i}\n" for i in range(5)},
    )
    monkeypatch.setenv(DEPENDENCY_INDEX_MAX_FILES_ENV, "3")

    report = index_dependency("python", "alicelib", project_dir, root=tmp_path / "deps")

    assert report["resolved"] is True
    assert report["truncated"] is True, "6 eligible files over a cap of 3 must report truncated"
    assert report["indexed"] <= 3


# -- Index-target XOR: exactly one of {path} or {ecosystem+package+dir} ---


def test_index_target_rejects_both_path_and_dependency_spec(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    both = _run_cli(
        home_dir, "index", str(corpus),
        "--ecosystem", "python", "--package", "alicelib", "--project-dir", str(corpus),
    )
    assert both.returncode != 0
    assert "not both" in both.stderr

    neither = _run_cli(home_dir, "index")
    assert neither.returncode != 0
    assert "supply either" in neither.stderr


# -- Cross-surface parity: iai-tools and iai-mcp tools share one body -----


def test_both_cli_surfaces_return_identical_dependency_hits(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    marker = "crossmarker" + uuid.uuid4().hex

    project_dir = _make_project_with_venv(tmp_path)
    _add_package(project_dir, "alicelib", {"core.py": f"{marker} = 1\n"})

    index_proc = _run_cli(
        home_dir, "index",
        "--ecosystem", "python", "--package", "alicelib", "--project-dir", str(project_dir),
    )
    assert index_proc.returncode == 0, index_proc.stderr

    iai_tools_search = _run_cli(
        home_dir, "search", marker,
        "--store", "dependency", "--ecosystem", "python", "--package", "alicelib",
    )
    assert iai_tools_search.returncode == 0, iai_tools_search.stderr

    iai_mcp_search = _run_iai_mcp_cli(
        home_dir, "tools", "search", marker,
        "--store", "dependency", "--ecosystem", "python", "--package", "alicelib",
    )
    assert iai_mcp_search.returncode == 0, iai_mcp_search.stderr

    assert iai_tools_search.stdout == iai_mcp_search.stdout, (
        "both CLI surfaces must return identical hits for the same indexed dependency"
    )
    assert iai_tools_search.stdout.strip(), "the search must actually find the indexed marker"


# -- Freshness gate: unchanged lockfile hash skips, changed hash reindexes -


def test_dependency_freshness_gate_skips_unchanged_and_reindexes_changed(tmp_path: Path) -> None:
    from iai_mcp.tools import dependency_store_root_for, manifest_scope
    from iai_mcp.tools.dependency import index_dependency_gated

    project_dir = _make_project_with_venv(tmp_path)
    _add_package(project_dir, "alicelib", {"core.py": "VALUE = 1\n"})
    (project_dir / "requirements.txt").write_text("alicelib==1.0\n")

    deps_root = tmp_path / "deps"

    first = index_dependency_gated("python", "alicelib", project_dir, root=deps_root)
    assert first["resolved"] is True
    assert first["reused"] is False

    store_root = dependency_store_root_for("python", "alicelib", root=deps_root)
    generation_after_first = manifest_scope(store_root)["generation"]

    second = index_dependency_gated("python", "alicelib", project_dir, root=deps_root)
    assert second["reused"] is True, "an unchanged lockfile hash must skip the re-index"
    generation_after_second = manifest_scope(store_root)["generation"]
    assert generation_after_second == generation_after_first, (
        "a skipped re-index must never advance the dependency store generation"
    )

    (project_dir / "requirements.txt").write_text("alicelib==2.0\n")
    # Content also changes -- the store's generation only advances when the
    # native index sees genuinely different content, verified empirically:
    # re-indexing byte-identical content is a no-op at the generation level
    # even on an unconditional (ungated) call.
    _venv_site_packages(project_dir / ".venv").joinpath("alicelib", "core.py").write_text(
        "VALUE = 2\n"
    )

    third = index_dependency_gated("python", "alicelib", project_dir, root=deps_root)
    assert third["reused"] is False, "a changed lockfile hash must re-index"
    generation_after_third = manifest_scope(store_root)["generation"]
    assert generation_after_third > generation_after_first, (
        "a real re-index must advance the dependency store generation"
    )


def test_dependency_freshness_lock_prevents_concurrent_reindex(tmp_path: Path) -> None:
    from iai_mcp.tools import dependency_store_root_for
    from iai_mcp.tools.dependency import dependency_lock_path, index_dependency_gated

    deps_root = tmp_path / "deps"
    lock_dir = dependency_lock_path("python", "alicelib", root=deps_root)
    lock_dir.mkdir(parents=True)

    result = index_dependency_gated("python", "alicelib", tmp_path, root=deps_root)

    assert result["reused"] is True
    assert result["resolved"] is False, "a held lock must skip the caller's own re-index attempt"
    store_root = dependency_store_root_for("python", "alicelib", root=deps_root)
    assert not store_root.exists(), "a held lock must prevent any indexing side effect"


# -- Node/Rust resolvers + cross-ecosystem path containment ---------------


_CARGO_INDEX_DIRNAME = "index.crates.io-testfixture"


def _make_cargo_home(tmp_path: Path) -> Path:
    cargo_home = tmp_path / "cargo-home"
    (cargo_home / "registry" / "src" / _CARGO_INDEX_DIRNAME).mkdir(parents=True)
    return cargo_home


def test_resolve_package_location(tmp_path: Path, monkeypatch) -> None:
    from iai_mcp.tools.dependency import resolve_package_location

    # Node: upward walk finds node_modules/<package> from a nested cwd.
    project_root = tmp_path / "monorepo"
    nested = project_root / "packages" / "app"
    nested.mkdir(parents=True)
    node_pkg_dir = project_root / "node_modules" / "alicejs"
    node_pkg_dir.mkdir(parents=True)
    (node_pkg_dir / "index.js").write_text("module.exports = {};\n")

    resolved_node = resolve_package_location("node", "alicejs", nested)
    assert resolved_node is not None
    assert resolved_node.resolve() == node_pkg_dir.resolve()

    # Node: absent at every level is a clean not-found.
    empty_project = tmp_path / "empty-project"
    empty_project.mkdir()
    assert resolve_package_location("node", "neverinstalled", empty_project) is None

    # Rust: exactly one matching version dir resolves to it.
    cargo_home = _make_cargo_home(tmp_path)
    crate_dir = cargo_home / "registry" / "src" / _CARGO_INDEX_DIRNAME / "alicecrate-1.0.0"
    crate_dir.mkdir()
    monkeypatch.setenv("CARGO_HOME", str(cargo_home))

    resolved_rust = resolve_package_location("rust", "alicecrate", tmp_path)
    assert resolved_rust is not None
    assert resolved_rust.resolve() == crate_dir.resolve()


def test_rust_multi_version_refuses_with_signal(tmp_path: Path, monkeypatch) -> None:
    from iai_mcp.tools.dependency import (
        AmbiguousDependencyVersionError,
        resolve_package_location,
    )

    cargo_home = _make_cargo_home(tmp_path)
    v1 = cargo_home / "registry" / "src" / _CARGO_INDEX_DIRNAME / "alicecrate-1.0.0"
    v2 = cargo_home / "registry" / "src" / _CARGO_INDEX_DIRNAME / "alicecrate-2.0.0"
    v1.mkdir()
    v2.mkdir()
    monkeypatch.setenv("CARGO_HOME", str(cargo_home))

    with pytest.raises(AmbiguousDependencyVersionError) as exc_info:
        resolve_package_location("rust", "alicecrate", tmp_path)

    assert set(exc_info.value.matched_versions) == {"alicecrate-1.0.0", "alicecrate-2.0.0"}

    from iai_mcp.tools.dependency import index_dependency

    report = index_dependency("rust", "alicecrate", tmp_path, root=tmp_path / "deps")
    assert report["resolved"] is False
    assert report["ambiguous"] is True
    assert set(report["matched_versions"]) == {"alicecrate-1.0.0", "alicecrate-2.0.0"}
    assert not (tmp_path / "deps").exists(), "an ambiguous match must index nothing"


def test_resolve_rejects_path_escape(tmp_path: Path) -> None:
    from iai_mcp.tools.dependency import resolve_package_location

    for ecosystem in ("python", "node", "rust"):
        with pytest.raises(ValueError):
            resolve_package_location(ecosystem, "../../etc/passwd", tmp_path)


# -- Structural scope fence: no bulk indexing, no network fetch -----------


_DEPENDENCY_MODULE_PATH = REPO / "src" / "iai_mcp" / "tools" / "dependency.py"

# Fragment-built so this guard's own source never spells a forbidden
# shape contiguously -- a network-transport or package-install token
# assembled here at runtime, never as a whole literal.
_URLLIB_TOKEN = "urllib" + "." + "request"
_HTTP_CLIENT_TOKEN = "http" + "." + "client"
_SOCKET_TOKEN = "socket" + "." + "socket("
_REQUESTS_TOKEN = "requests" + "."
_HTTPX_TOKEN = "httpx" + "."
_AIOHTTP_TOKEN = "aiohttp" + "."

_NETWORK_TRANSPORT_TOKENS = (
    _URLLIB_TOKEN,
    _HTTP_CLIENT_TOKEN,
    _SOCKET_TOKEN,
    _REQUESTS_TOKEN,
    _HTTPX_TOKEN,
    _AIOHTTP_TOKEN,
)

_PKG_MANAGERS = ("pip", "pip3", "npm", "cargo", "yarn", "poetry")
_INSTALL_VERBS = ("install", "fetch", "add")
_PKG_MANAGER_VERB_TOKENS = tuple(
    f"{manager} {verb}" for manager in _PKG_MANAGERS for verb in _INSTALL_VERBS
)

_FORBIDDEN_TOKENS = _NETWORK_TRANSPORT_TOKENS + _PKG_MANAGER_VERB_TOKENS

# Network-capable CLI tools a subprocess call could shell out to instead of
# importing a transport library -- the guard above only catches the latter
# shape, leaving this the more likely accidental-introduction vector in a
# module that already spawns subprocess.run.
_NETWORK_CLI_TOOLS = ("curl", "wget", "scp", "ssh", "rsync")


def _subprocess_call_argv_text(source: str) -> str:
    """Every string literal appearing inside a subprocess.run/Popen/call/
    check_call/check_output or os.system/os.popen call anywhere in
    `source`, joined in source order -- the argv-string surface a shell-
    executed network tool would have to name. AST-aware so a future
    `subprocess.run(["curl", "-O", url])` is caught structurally, not by
    a source-substring guess a renamed variable could dodge.
    """
    import ast

    tree = ast.parse(source)
    consts: "list[ast.Constant]" = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_subprocess_call = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
            and func.attr in ("run", "Popen", "call", "check_call", "check_output")
        )
        is_os_call = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and func.attr in ("system", "popen")
        )
        if not (is_subprocess_call or is_os_call):
            continue
        for arg_node in ast.walk(node):
            if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                consts.append(arg_node)
    consts.sort(key=lambda n: (n.lineno, n.col_offset))
    return " ".join(n.value for n in consts)


def test_no_network_fetch_verbs_in_dependency_module() -> None:
    source = _DEPENDENCY_MODULE_PATH.read_text(encoding="utf-8")
    hits = [token for token in _FORBIDDEN_TOKENS if token in source]
    assert not hits, (
        "dependency.py must never contain a network-transport or "
        f"package-install/fetch call shape: {hits}"
    )

    argv_text = _subprocess_call_argv_text(source)
    cli_hits = [tool for tool in _NETWORK_CLI_TOOLS if tool in argv_text]
    if "git" in argv_text and "clone" in argv_text:
        cli_hits.append("git clone")
    assert not cli_hits, (
        "dependency.py must never invoke a network-capable subprocess tool "
        f"(curl/wget/scp/ssh/rsync/git clone) via its own argv: {cli_hits}"
    )


def test_only_single_named_package_index_entry() -> None:
    import ast

    source = _DEPENDENCY_MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: [arg.arg for arg in node.args.args]
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    forbidden_name_fragments = ("bulk", "recursive", "wildcard", "all_packages", "every_package")
    offending = [
        name
        for name in functions
        if any(fragment in name.lower() for fragment in forbidden_name_fragments)
    ]
    assert not offending, (
        f"a bulk/recursive/wildcard indexing entry point must not exist: {offending}"
    )

    for entry_point in ("index_dependency", "index_dependency_gated"):
        assert entry_point in functions, f"expected index entry point {entry_point!r} not found"
        params = functions[entry_point]
        assert "package" in params, f"{entry_point} must accept a single named 'package' parameter"
        assert "packages" not in params, f"{entry_point} must not accept a plural 'packages' collection"


# -- Never fall back to a bare PATH-resolved interpreter -------------------


def test_python_resolution_without_venv_returns_clean_not_found(tmp_path: Path) -> None:
    """A project dir with no known venv shape must never resolve via the
    calling process's own inherited PATH interpreter -- reproduced with
    `pip`, a package virtually guaranteed importable in whatever
    interpreter is running this test, so a regression would silently
    resolve a wrong-environment hit instead of a clean not-found.
    """
    from iai_mcp.tools.dependency import _python_interpreter_for, resolve_package_location

    project_dir = tmp_path / "no-venv-project"
    project_dir.mkdir()

    assert _python_interpreter_for(project_dir) is None

    resolved = resolve_package_location("python", "pip", project_dir)
    assert resolved is None, (
        "a project with no known venv shape must resolve to a clean not-found, "
        "never a wrong-environment hit from the calling process's own interpreter"
    )


# -- An editable (pip install -e) install resolves, not not-found ---------


def test_resolve_python_package_editable_install_outside_site_packages(tmp_path: Path) -> None:
    from iai_mcp.tools.dependency import resolve_package_location

    project_dir = _make_project_with_venv(tmp_path)
    site_packages = _venv_site_packages(project_dir / ".venv")

    src_dir = project_dir / "src"
    pkg_dir = src_dir / "editablepkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")

    # A .pth-redirect is one of the two documented editable-install shapes
    # (PEP 660 finder-based, and this older .pth-redirect style) -- both put
    # the resolved location outside every site-packages root.
    (site_packages / "editable-redirect.pth").write_text(str(src_dir) + "\n")

    resolved = resolve_package_location("python", "editablepkg", project_dir)
    assert resolved is not None, (
        "an editable install's source-tree location must resolve, not fall "
        "back to a false not-found just because it is outside site-packages"
    )
    assert resolved.resolve() == pkg_dir.resolve()


# -- Exactly-cap eligible files must not report truncated ------------------


def test_dependency_index_not_truncated_when_files_exactly_equal_cap(
    tmp_path: Path, monkeypatch
) -> None:
    from iai_mcp.tools.dependency import DEPENDENCY_INDEX_MAX_FILES_ENV, index_dependency

    project_dir = _make_project_with_venv(tmp_path)
    # __init__.py + 2 extra modules == 3 eligible files, exactly the cap.
    _add_package(
        project_dir,
        "alicelib",
        {f"mod_{i}.py": f"VALUE_{i} = {i}\n" for i in range(2)},
    )
    monkeypatch.setenv(DEPENDENCY_INDEX_MAX_FILES_ENV, "3")

    report = index_dependency("python", "alicelib", project_dir, root=tmp_path / "deps")

    assert report["resolved"] is True
    assert report["indexed"] == 3
    assert report["truncated"] is False, (
        "exactly-cap eligible files with nothing dropped must not report truncated"
    )


# -- A package name containing '/' must never crash ------------------------


def test_index_dependency_scoped_npm_package_clean_refusal(tmp_path: Path) -> None:
    from iai_mcp.tools.dependency import index_dependency

    report = index_dependency("node", "@babel/core", tmp_path, root=tmp_path / "deps")
    assert report["resolved"] is False
    assert report["indexed"] == 0
    assert not (tmp_path / "deps").exists()


def test_cli_index_scoped_npm_package_returns_clean_error_not_traceback(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    proc = _run_cli(
        home_dir, "index",
        "--ecosystem", "node", "--package", "@babel/core", "--project-dir", str(project_dir),
    )
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr, f"CLI must never surface a raw traceback: {proc.stderr}"
    assert "not found" in proc.stderr


# -- The freshness gate is wired into the real CLI entry point -------------


def test_cli_index_reuses_on_unchanged_lockfile_and_reindexes_on_change(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    project_dir = _make_project_with_venv(tmp_path)
    _add_package(project_dir, "alicelib", {"core.py": "VALUE = 1\n"})
    (project_dir / "requirements.txt").write_text("alicelib==1.0\n")

    first = _run_cli(
        home_dir, "index",
        "--ecosystem", "python", "--package", "alicelib", "--project-dir", str(project_dir),
    )
    assert first.returncode == 0, first.stderr
    assert "reused" not in first.stdout, "the first real index must not be reported as reused"

    second = _run_cli(
        home_dir, "index",
        "--ecosystem", "python", "--package", "alicelib", "--project-dir", str(project_dir),
    )
    assert second.returncode == 0, second.stderr
    assert "reused" in second.stdout and "skipped_reason=fresh" in second.stdout, (
        "the real CLI index path must skip re-indexing when the lockfile hash is unchanged: "
        f"{second.stdout!r}"
    )

    (project_dir / "requirements.txt").write_text("alicelib==2.0\n")
    third = _run_cli(
        home_dir, "index",
        "--ecosystem", "python", "--package", "alicelib", "--project-dir", str(project_dir),
    )
    assert third.returncode == 0, third.stderr
    assert "reused" not in third.stdout, "a changed lockfile hash must re-index, not skip"


def test_dependency_freshness_gate_never_seeds_watermark_on_failed_resolution(
    tmp_path: Path,
) -> None:
    """A failed resolution (not-found, ambiguous, malformed segment) must
    never write the freshness watermark -- otherwise a second identical
    call with an unchanged lockfile would report `reused, resolved: True`
    for a package that was never actually indexed.
    """
    from iai_mcp.tools.dependency import index_dependency_gated

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "requirements.txt").write_text("neverinstalled==1.0\n")

    first = index_dependency_gated(
        "python", "neverinstalled", project_dir, root=tmp_path / "deps"
    )
    assert first["resolved"] is False
    assert first["reused"] is False

    second = index_dependency_gated(
        "python", "neverinstalled", project_dir, root=tmp_path / "deps"
    )
    assert second["resolved"] is False, (
        "a not-found package must stay not-found on a second call with the same "
        "lockfile, never a false 'reused, resolved: True' from a wrongly-seeded watermark"
    )


# -- An ambiguous match must never render as a genuine not-found -----------


def test_cli_index_ambiguous_rust_version_message_differs_from_not_found(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    cargo_home = _make_cargo_home(tmp_path)
    v1 = cargo_home / "registry" / "src" / _CARGO_INDEX_DIRNAME / "alicecrate-1.0.0"
    v2 = cargo_home / "registry" / "src" / _CARGO_INDEX_DIRNAME / "alicecrate-2.0.0"
    v1.mkdir()
    v2.mkdir()

    env = _cli_env(home_dir)
    env["CARGO_HOME"] = str(cargo_home)
    proc = subprocess.run(
        [
            sys.executable, "-m", "iai_mcp.tools_cli", "index",
            "--ecosystem", "rust", "--package", "alicecrate", "--project-dir", str(tmp_path),
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1
    assert "ambiguous" in proc.stderr
    assert "not found" not in proc.stderr, (
        "an ambiguous multi-version match must never render identically to a genuine not-found: "
        f"{proc.stderr!r}"
    )


# -- Lock stale-timeout is configurable via env, not hardcoded -------------


def test_dependency_lock_stale_seconds_configurable_via_env(tmp_path: Path, monkeypatch) -> None:
    import time

    from iai_mcp.tools.dependency import (
        DEPENDENCY_LOCK_STALE_SECONDS_ENV,
        _acquire_dependency_lock,
    )

    lock_dir = tmp_path / "lock"
    lock_dir.mkdir()
    old_mtime = time.time() - 5
    os.utime(lock_dir, (old_mtime, old_mtime))

    # Default stale window (60s) must not consider a 5s-old lock stale.
    assert _acquire_dependency_lock(lock_dir) is None

    monkeypatch.setenv(DEPENDENCY_LOCK_STALE_SECONDS_ENV, "1")
    nonce = _acquire_dependency_lock(lock_dir)
    assert nonce is not None, "a shorter configured stale window must let a 5s-old lock be stolen"
