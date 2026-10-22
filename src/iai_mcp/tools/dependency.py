"""Shared body for package-scoped dependency indexing across the Python,
Node, and Rust ecosystems: resolve one installed package's on-disk
location, index just that package's files into a store separate from the
project source store, and search that store.

Module-level imports are stdlib only. `iai_mcp.tools`/`iai_mcp.study` are
imported at call time, so a plain import of this module stays native-
extension-free and brain-free, mirroring `git_query.py`.

`validate_index_target` is the single authoritative choke point both CLI
surfaces' index handlers call for the index-target XOR -- no caller
surface can bypass it; the MCP layer only mirrors it pre-spawn.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

DEPENDENCY_INDEX_MAX_FILES_ENV = "IAI_MCP_DEPENDENCY_INDEX_MAX_FILES"
DEPENDENCY_INDEX_MAX_FILES_DEFAULT = 20000

# Resolves ONE package's install directory in the interpreter this script
# runs under. The package name arrives as a discrete argv element
# (sys.argv[1]) -- never string-formatted into this snippet -- so a
# malicious name cannot break out of the -c script text.
_PYTHON_RESOLVE_SNIPPET = """
import importlib.util as _u
import site as _site
import sys as _sys

_name = _sys.argv[1]
_spec = _u.find_spec(_name)
if _spec is None:
    raise SystemExit(1)
if _spec.submodule_search_locations:
    _location = list(_spec.submodule_search_locations)[0]
elif _spec.origin:
    _location = _spec.origin
else:
    raise SystemExit(1)
_roots = []
try:
    _roots.extend(_site.getsitepackages())
except Exception:
    pass
try:
    _roots.append(_site.getusersitepackages())
except Exception:
    pass
print(_location)
for _root in _roots:
    print(_root)
"""


class DependencyRequestError(ValueError):
    """Raised for a malformed dependency index/search request: both or
    neither of {path} and the (ecosystem, package, project_dir) triple
    supplied to index, or a dependency search missing ecosystem/package.
    """


class AmbiguousDependencyVersionError(ValueError):
    """Raised when more than one cached version directory matches a
    requested package -- refuse-with-signal rather than guessing a
    version. `matched_versions` holds the ambiguous version-dir names."""

    def __init__(self, package: str, matched_versions: "list[str]") -> None:
        self.package = package
        self.matched_versions = matched_versions
        super().__init__(
            f"ambiguous package {package!r}: {len(matched_versions)} versions matched"
        )


def validate_index_target(
    *,
    path: "str | None",
    ecosystem: "str | None",
    package: "str | None",
    project_dir: "str | None",
) -> None:
    """The single authoritative choke point for the index-target XOR: a
    caller supplies EITHER `path` OR the full (ecosystem, package,
    project_dir) triple, never both, never neither, and never a partial
    triple. Both CLI surfaces' index handlers call this before dispatch --
    no caller surface can bypass it.
    """
    dep_fields = (ecosystem, package, project_dir)
    has_dep_any = any(field is not None for field in dep_fields)
    has_dep_all = all(field is not None for field in dep_fields)
    if has_dep_any and not has_dep_all:
        raise DependencyRequestError(
            "index: --ecosystem/--package/--project-dir must all be supplied together"
        )
    has_path = path is not None
    if has_path and has_dep_all:
        raise DependencyRequestError(
            "index: supply either a path or --ecosystem/--package/--project-dir, not both"
        )
    if not has_path and not has_dep_all:
        raise DependencyRequestError(
            "index: supply either a path or --ecosystem/--package/--project-dir"
        )


_PYTHON_VENV_CANDIDATES = (
    Path(".venv") / "bin" / "python3",
    Path("venv") / "bin" / "python3",
    Path(".env") / "bin" / "python3",
    Path("env") / "bin" / "python3",
)


def _python_interpreter_for(project_dir: Path) -> "str | None":
    """Probes a small set of known venv shapes under `project_dir`.
    Never falls back to a bare, PATH-resolved interpreter name -- that
    would resolve via the CALLING process's own inherited PATH rather
    than the target project's environment, reintroducing the interpreter-
    mismatch bug one call-frame later inside the spawned subprocess.
    Returns None when no known venv shape is found; the caller must
    treat that as a clean not-found, never substitute its own PATH.
    """
    for candidate in _PYTHON_VENV_CANDIDATES:
        venv_python = project_dir / candidate
        if venv_python.is_file():
            return str(venv_python)
    return None


_PYTHON_LOCATION_DENYLIST = (
    Path("/tmp"),
    Path("/var/tmp"),
    Path("/etc"),
    Path("/dev"),
    Path("/proc"),
    Path("/sys"),
)


def _is_denied_python_location(location: Path) -> bool:
    """True for a resolved location clearly unrelated to any package
    install (system scratch/config dirs) -- the narrow deny-list an
    editable install's out-of-site-packages source tree must still pass
    through, rather than the blanket site-packages-only containment that
    rejected every editable install."""
    for denied in _PYTHON_LOCATION_DENYLIST:
        try:
            location.relative_to(denied.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def _resolve_python_package(package: str, project_dir: Path) -> "Path | None":
    interpreter = _python_interpreter_for(project_dir)
    if interpreter is None:
        return None
    try:
        proc = subprocess.run(
            [interpreter, "-c", _PYTHON_RESOLVE_SNIPPET, package],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None

    lines = [line for line in proc.stdout.splitlines() if line]
    if not lines:
        return None
    location = Path(lines[0]).resolve()
    for root in lines[1:]:
        try:
            location.relative_to(Path(root).resolve())
            return location
        except (ValueError, OSError):
            continue
    # Not under a site-packages root -- an editable install's source tree
    # resolves here by construction (PEP 660, and the older .pth-redirect
    # style alike). The TARGET interpreter's own find_spec already vouches
    # for this exact location; only reject one clearly unrelated to any
    # package install.
    if _is_denied_python_location(location):
        return None
    return location


def _resolve_node_package(package: str, project_dir: Path) -> "Path | None":
    """Standard Node module-resolution algorithm: walk `node_modules/
    <package>` upward from `project_dir` to the filesystem root. A
    candidate whose resolved path escapes the `node_modules` it was found
    under (e.g. a symlink pointing outside) is treated as not-found rather
    than followed.
    """
    current = project_dir.resolve()
    while True:
        node_modules = current / "node_modules"
        candidate = node_modules / package
        if candidate.is_dir():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(node_modules.resolve())
            except (ValueError, OSError):
                return None
            return resolved
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _cargo_home() -> Path:
    override = os.environ.get("CARGO_HOME")
    if override:
        return Path(override)
    return Path.home() / ".cargo"


# A matched registry dir name is "<package>-<version>"; the version suffix
# must start with a digit so "foo-bar" never matches package "foo" (would
# otherwise collide with an unrelated crate sharing the name as a prefix).
_RUST_VERSION_SUFFIX_RE = re.compile(r"^\d")


def _resolve_rust_package(package: str) -> "Path | None":
    """Globs `$CARGO_HOME/registry/src/*/<package>-<version>` (default
    `~/.cargo`). Zero matches is a clean not-found. Exactly one match
    resolves it. More than one match -- the registry cache is global and
    accumulates every version ever fetched across every project on the
    machine -- raises `AmbiguousDependencyVersionError` rather than
    guessing a version.
    """
    registry_src = _cargo_home() / "registry" / "src"
    if not registry_src.is_dir():
        return None

    prefix = f"{package}-"
    matches: "list[Path]" = []
    for index_dir in sorted(registry_src.iterdir()):
        if not index_dir.is_dir():
            continue
        for candidate in sorted(index_dir.iterdir()):
            if not candidate.is_dir() or not candidate.name.startswith(prefix):
                continue
            suffix = candidate.name[len(prefix):]
            if _RUST_VERSION_SUFFIX_RE.match(suffix):
                matches.append(candidate)

    if not matches:
        return None
    if len(matches) > 1:
        raise AmbiguousDependencyVersionError(package, sorted(m.name for m in matches))

    resolved = matches[0].resolve()
    try:
        resolved.relative_to(registry_src.resolve())
    except (ValueError, OSError):
        return None
    return resolved


def resolve_package_location(
    ecosystem: str, package: str, project_dir: "str | Path"
) -> "Path | None":
    """Resolves ONE package's on-disk install directory across all three
    supported ecosystems. Python resolution subprocesses the TARGET
    project's own interpreter -- never the calling process's -- so an
    environment mismatch never resolves the wrong package. Node walks
    `node_modules/<package>` upward from `project_dir` to the filesystem
    root. Rust globs the cargo registry cache and raises
    `AmbiguousDependencyVersionError` on more than one matching version.
    Every branch validates ecosystem+package as one safe path segment
    BEFORE any path-join or subprocess argv, whether the escape attempt
    is a malformed package name or (node/rust) a resolved candidate that
    escapes its ecosystem root. Returns None on a clean not-found; never
    raises for that case.
    """
    from iai_mcp.tools import validate_dependency_segment

    validate_dependency_segment(ecosystem, label="ecosystem")
    validate_dependency_segment(package, label="package")

    if ecosystem == "python":
        return _resolve_python_package(package, Path(project_dir))
    if ecosystem == "node":
        return _resolve_node_package(package, Path(project_dir))
    if ecosystem == "rust":
        return _resolve_rust_package(package)
    raise ValueError(f"unsupported ecosystem: {ecosystem!r}")


def _dependency_index_max_files() -> int:
    try:
        return int(
            os.environ.get(
                DEPENDENCY_INDEX_MAX_FILES_ENV, str(DEPENDENCY_INDEX_MAX_FILES_DEFAULT)
            )
        )
    except (TypeError, ValueError):
        return DEPENDENCY_INDEX_MAX_FILES_DEFAULT


def index_dependency(
    ecosystem: str, package: str, project_dir: "str | Path", *, root: "Path | None" = None
) -> "dict[str, Any]":
    """Resolves, then indexes exactly one package's files into its own
    nested dependency store -- never the project source store. `truncated`
    reflects `iter_study_files`'s own over-cap count (`return_skipped=True`),
    never approximated from the returned list length, which is a false
    positive whenever the package has exactly `cap` eligible files. An
    ambiguous multi-version match (Rust) indexes nothing and reports
    `ambiguous: True` with the matched version list; a malformed
    ecosystem/package segment (e.g. a scoped npm name containing `/`)
    reports a clean refusal instead of propagating `DependencySegmentError`
    past this choke point. Narrowly scoped to that one malformed-input
    shape -- an unrelated `ValueError` such as `resolve_package_location`'s
    "unsupported ecosystem" signal (a value that IS one safe path segment
    but names no supported ecosystem) still propagates, since silently
    reporting that as a plain not-found would hide a real caller mistake
    behind a misleading message.
    """
    from iai_mcp.study import iter_study_files
    from iai_mcp.tools import DependencySegmentError, dependency_store_root_for, index_paths

    try:
        location = resolve_package_location(ecosystem, package, project_dir)
    except AmbiguousDependencyVersionError as exc:
        return {
            "resolved": False,
            "path": None,
            "indexed": 0,
            "truncated": False,
            "store_root": None,
            "ambiguous": True,
            "matched_versions": exc.matched_versions,
        }
    except DependencySegmentError:
        return {
            "resolved": False,
            "path": None,
            "indexed": 0,
            "truncated": False,
            "store_root": None,
        }
    if location is None:
        return {
            "resolved": False,
            "path": None,
            "indexed": 0,
            "truncated": False,
            "store_root": None,
        }

    store_root = dependency_store_root_for(ecosystem, package, root=root)
    cap = _dependency_index_max_files()
    files, skipped_over_cap = iter_study_files(location, max_files=cap, return_skipped=True)
    indexed = index_paths([str(f) for f in files], store_root=store_root)
    return {
        "resolved": True,
        "path": str(location),
        "indexed": indexed,
        "truncated": skipped_over_cap > 0,
        "store_root": str(store_root),
    }


def is_dependency_store_indexed(store_root: Path) -> bool:
    """True when a dependency's nested store dir exists -- the not-indexed
    signal a search surface checks before treating an empty hit list as a
    genuine zero-result search rather than "never indexed"."""
    return store_root.is_dir()


DEPENDENCY_WATERMARK_DIRNAME = ".watermarks"
DEPENDENCY_LOCK_STALE_SECONDS_ENV = "IAI_MCP_DEPENDENCY_LOCK_STALE_SECONDS"
DEPENDENCY_LOCK_STALE_SECONDS = 60


def _dependency_lock_stale_seconds() -> int:
    try:
        return int(
            os.environ.get(
                DEPENDENCY_LOCK_STALE_SECONDS_ENV, str(DEPENDENCY_LOCK_STALE_SECONDS)
            )
        )
    except (TypeError, ValueError):
        return DEPENDENCY_LOCK_STALE_SECONDS


_LOCKFILE_NAMES = ("requirements.txt", "poetry.lock", "package-lock.json", "Cargo.lock")


def lockfile_signature(project_dir: "str | Path") -> "str | None":
    """Content-hash signature of the first present lockfile/manifest among
    `requirements.txt`, `poetry.lock`, `package-lock.json`, `Cargo.lock`.
    None when none is present -- degrades to always-reindex, mirroring the
    SessionStart hook's unset-signature degrade.
    """
    base = Path(project_dir)
    for name in _LOCKFILE_NAMES:
        candidate = base / name
        if candidate.is_file():
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            return f"{name}:{digest}"
    return None


def _dependency_watermark_key(ecosystem: str, package: str) -> str:
    return hashlib.sha256(f"{ecosystem}:{package}".encode("utf-8")).hexdigest()


def dependency_watermark_path(
    ecosystem: str, package: str, *, root: "Path | None" = None
) -> Path:
    """`default_dependency_store_root()/.watermarks/<sha256 of
    ecosystem+package>` -- filesystem-safe regardless of the package name.
    """
    from iai_mcp.tools import default_dependency_store_root

    base = root or default_dependency_store_root()
    return base / DEPENDENCY_WATERMARK_DIRNAME / _dependency_watermark_key(ecosystem, package)


def dependency_lock_path(ecosystem: str, package: str, *, root: "Path | None" = None) -> Path:
    watermark = dependency_watermark_path(ecosystem, package, root=root)
    return watermark.with_name(watermark.name + ".lock")


def _acquire_dependency_lock(
    lock_dir: Path, *, stale_seconds: "int | None" = None
) -> "str | None":
    """`os.mkdir` is atomic on POSIX, the same guarantee the shell hook's
    `mkdir` relies on -- two racing callers cannot both succeed. A held,
    non-stale lock means the caller SKIPS (fail-safe, never blocks). A
    lock older than `stale_seconds` (env-overridable via
    `DEPENDENCY_LOCK_STALE_SECONDS_ENV`, default `DEPENDENCY_LOCK_STALE_SECONDS`)
    is presumed abandoned and stolen via a single `os.rename` -- never
    remove-then-mkdir, which two racing stealers could both win. Returns
    the winning nonce, or None when the caller must skip.
    """
    if stale_seconds is None:
        stale_seconds = _dependency_lock_stale_seconds()
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.time_ns()}"
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        try:
            mtime = lock_dir.stat().st_mtime
        except OSError:
            return None
        if time.time() - mtime <= stale_seconds:
            return None
        steal_tmp = lock_dir.with_name(lock_dir.name + f".stale.{os.getpid()}")
        shutil.rmtree(steal_tmp, ignore_errors=True)
        try:
            os.rename(lock_dir, steal_tmp)
        except OSError:
            return None
        shutil.rmtree(steal_tmp, ignore_errors=True)
        try:
            os.mkdir(lock_dir)
        except FileExistsError:
            return None
    owner_file = lock_dir / "owner"
    owner_file.write_text(nonce, encoding="utf-8")
    # A legitimately-stolen lock's original owner must never delete the new
    # holder's lock on its own release -- release only fires when the
    # nonce read back still matches the one this call just wrote.
    if owner_file.read_text(encoding="utf-8") != nonce:
        return None
    return nonce


def _release_dependency_lock(lock_dir: Path, nonce: str) -> None:
    try:
        owner = (lock_dir / "owner").read_text(encoding="utf-8")
    except OSError:
        return
    if owner == nonce:
        shutil.rmtree(lock_dir, ignore_errors=True)


def index_dependency_gated(
    ecosystem: str, package: str, project_dir: "str | Path", *, root: "Path | None" = None
) -> "dict[str, Any]":
    """Freshness-gated wrapper around `index_dependency`. Acquires the
    mkdir-atomic lock first (matching the SessionStart hook's own check-
    run-write ordering), then checks the lockfile-content watermark under
    the lock: an unchanged signature skips the re-index (`reused=True`); a
    changed or absent signature re-indexes and advances the watermark. A
    held, non-stale lock makes a concurrent caller skip its own re-index
    entirely (`resolved=False`) rather than block or double-index. The
    watermark is only ever written for a `resolved: True` report -- a
    failed resolution (not-found, ambiguous, malformed segment) must never
    seed a "fresh" watermark, or a later identical call would report
    `reused, resolved: True` for a package that was never actually
    indexed.
    """
    watermark_file = dependency_watermark_path(ecosystem, package, root=root)
    lock_dir = dependency_lock_path(ecosystem, package, root=root)

    nonce = _acquire_dependency_lock(lock_dir)
    if nonce is None:
        return {"reused": True, "resolved": False, "skipped_reason": "locked"}
    try:
        signature = lockfile_signature(project_dir)
        if signature is not None and watermark_file.is_file():
            prior = watermark_file.read_text(encoding="utf-8")
            if prior == signature:
                return {"reused": True, "resolved": True, "skipped_reason": "fresh"}
        report = index_dependency(ecosystem, package, project_dir, root=root)
        if signature is not None and report.get("resolved"):
            watermark_file.parent.mkdir(parents=True, exist_ok=True)
            watermark_file.write_text(signature, encoding="utf-8")
        report["reused"] = False
        return report
    finally:
        _release_dependency_lock(lock_dir, nonce)


def describe_index_outcome(
    ecosystem: str, package: str, report: "dict[str, Any]"
) -> "tuple[str, int]":
    """Renders one `index_dependency_gated` report as a CLI-ready
    `(message, exit_code)` pair. Both CLI surfaces (`tools_cli.py`'s
    `_cmd_index`, `cli/_tools.py`'s `cmd_tools_index`) call this so the
    two never drift on how they describe a reused/locked-skip/ambiguous/
    not-found/indexed outcome. An ambiguous multi-version match is
    branched BEFORE the generic not-found message, and each message
    carries a distinct, machine-greppable marker (`reused (skipped_reason=
    ...)`, `is ambiguous: ...`) so a caller -- human, script, or the MCP
    wrapper parsing this same text -- never renders it identically to a
    genuine not-found.
    """
    if report.get("reused") and report.get("skipped_reason") == "locked":
        return (
            f"{ecosystem} package {package} reused (skipped_reason=locked, "
            "held by a concurrent indexer)",
            0,
        )
    if report.get("reused") and report.get("skipped_reason") == "fresh":
        return (
            f"{ecosystem} package {package} reused (skipped_reason=fresh, "
            "lockfile unchanged)",
            0,
        )
    if report.get("ambiguous"):
        versions = ", ".join(report.get("matched_versions", []))
        return (
            f"{ecosystem} package {package!r} is ambiguous: "
            f"{len(report.get('matched_versions', []))} versions matched ({versions})",
            1,
        )
    if not report["resolved"]:
        return (f"{ecosystem} package not found: {package}", 1)
    suffix = " (truncated)" if report["truncated"] else ""
    return (
        f"indexed {report['indexed']} files under {report['store_root']} "
        f"for {ecosystem} package {package}{suffix}",
        0,
    )
