from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
COMMUNITY_PY = REPO_ROOT / "src" / "iai_mcp" / "community.py"
GRAPH_PY = REPO_ROOT / "src" / "iai_mcp" / "graph.py"
SUMMARY_MD = (
    REPO_ROOT
    / ".planning"
    / "phases"
    / "21-custom-mit-licensed-leiden-implementation"
    / "21-09-SUMMARY.md"
)


def test_pyproject_toml_does_not_require_leidenalg() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    offenders = [d for d in deps if d.lower().startswith("leidenalg")]
    assert not offenders, (
        f"leidenalg must NOT be in [project] dependencies "
        f"(license-clean shipping is pure-MIT). "
        f"Found offenders: {offenders}"
    )


def test_pyproject_toml_does_not_require_python_igraph() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    offenders = [d for d in deps if d.lower().startswith("python-igraph")]
    assert not offenders, (
        f"python-igraph must NOT be in [project] dependencies "
        f"(pure-MIT shipping path). "
        f"Found offenders: {offenders}"
    )


def test_pyproject_toml_no_gpl_deps_anywhere() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    extras = data["project"].get("optional-dependencies", {})

    bad_required = [
        d
        for d in deps
        if d.lower().startswith("leidenalg") or d.lower().startswith("python-igraph")
    ]
    assert not bad_required, (
        f"GPL deps MUST NOT be in [project] dependencies "
        f"(pure-MIT). Found: {bad_required}"
    )

    for group_name, group_deps in extras.items():
        bad = [
            d
            for d in group_deps
            if d.lower().startswith("leidenalg")
            or d.lower().startswith("python-igraph")
        ]
        assert not bad, (
            f"GPL deps MUST NOT be installable via any optional-dependencies "
            f"group (FULL removal, no [research] "
            f"escape hatch). Group [{group_name}] contained: {bad}"
        )

    assert "research" not in extras, (
        "[research] optional-dependencies group MUST NOT exist "
        "(no GPL-installable path retained)."
    )


def test_community_py_no_leidenalg_import() -> None:
    src = COMMUNITY_PY.read_text()
    assert "import leidenalg" not in src, (
        "community.py MUST NOT contain `import leidenalg` "
        "(production path is pure-MIT custom Leiden). "
        "Found a stray import; _run_leiden must stay deleted."
    )


def test_community_py_no_igraph_import() -> None:
    src = COMMUNITY_PY.read_text()
    assert "import igraph" not in src, (
        "community.py MUST NOT contain `import igraph`. "
        "igraph is fully retired from the project."
    )


def test_community_py_does_not_export_run_leiden() -> None:
    src = COMMUNITY_PY.read_text()
    assert "def _run_leiden" not in src, (
        "_run_leiden MUST be deleted from community.py "
        "(no leidenalg shim survives "
        "in the production module)."
    )


def test_community_py_does_not_export_map_to_stable_uuids() -> None:
    src = COMMUNITY_PY.read_text()
    assert "def _map_to_stable_uuids" not in src, (
        "_map_to_stable_uuids MUST be deleted from community.py "
        "(event-driven lineage via LineageTracker supersedes "
        "the cosine-match heuristic)."
    )


def test_community_py_imports_run_mosaic() -> None:
    src = COMMUNITY_PY.read_text()
    pattern = re.compile(
        r"from\s+iai_mcp\.mosaic\s+import[\s(]+[\w,\s]*run_mosaic",
        re.MULTILINE,
    )
    assert pattern.search(src) is not None, (
        "community.py MUST import `run_mosaic` from `iai_mcp.mosaic` "
        "(MOSAIC rename). "
        "Match attempted: `from iai_mcp.mosaic import ... run_mosaic`."
    )


def test_graph_py_has_no_igraph_references() -> None:
    src = GRAPH_PY.read_text()
    assert "import igraph" not in src, (
        "graph.py MUST NOT import igraph (single-backend adjacency-dict "
        "storage; igraph was retired)."
    )
    assert "_HAS_IGRAPH" not in src, (
        "graph.py MUST NOT define _HAS_IGRAPH (igraph fully retired)."
    )
    assert "IGRAPH_THRESHOLD" not in src, (
        "graph.py MUST NOT define IGRAPH_THRESHOLD (dual-backend "
        "switching retired)."
    )
    assert "_rebuild_igraph" not in src, (
        "graph.py MUST NOT define _rebuild_igraph (dual-backend "
        "switching retired)."
    )
    assert "_maybe_switch_backend" not in src, (
        "graph.py MUST NOT define _maybe_switch_backend (dual-backend "
        "switching retired)."
    )


def test_mosaic_modules_are_importable() -> None:
    from iai_mcp.mosaic import run_mosaic  # noqa: F401
    from iai_mcp.mosaic_lineage import (  # noqa: F401
        LineageEvent,
        LineageReport,
        LineageTracker,
        init_partitions,
    )
    from iai_mcp.mosaic_policy import (  # noqa: F401
        CPM_MODULARITY_FLOOR,
        all_communities_connected,
        compute_singleton_ratio,
        should_fall_back_to_flat,
    )


def test_legacy_custom_leiden_modules_are_absent() -> None:
    legacy_paths = (
        "iai_mcp.custom_leiden",
        "iai_mcp.custom_leiden_lineage",
        "iai_mcp.custom_leiden_policy",
    )
    for module_path in legacy_paths:
        spec = importlib.util.find_spec(module_path)
        assert spec is None, (
            f"Legacy module path `{module_path}` MUST NOT be importable "
            f"(MOSAIC rename). Found spec: {spec}. "
            f"Check that no `custom_leiden*.py` file got reintroduced in "
            f"src/iai_mcp/."
        )


def test_legacy_run_custom_leiden_symbol_is_absent() -> None:
    import iai_mcp.mosaic as _mod
    assert not hasattr(_mod, "run_custom_leiden"), (
        "iai_mcp.mosaic MUST NOT expose the legacy `run_custom_leiden` "
        "symbol (renamed to `run_mosaic`)."
    )
    assert hasattr(_mod, "run_mosaic"), (
        "iai_mcp.mosaic MUST expose the renamed `run_mosaic` symbol "
        "(MOSAIC rename)."
    )


def test_iai_mcp_community_imports_without_leidenalg() -> None:
    blocker = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_module(self, fullname, path=None):\n"
        "        if fullname == 'leidenalg':\n"
        "            return self\n"
        "        return None\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'leidenalg':\n"
        "            raise ImportError('leidenalg unavailable (post-removal test)')\n"
        "        return None\n"
        "    def load_module(self, fullname):\n"
        "        raise ImportError('leidenalg unavailable (post-removal test)')\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "from iai_mcp.community import detect_communities, CommunityAssignment\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        },
    )
    assert result.returncode == 0, (
        f"community.py top-level import path MUST NOT depend on leidenalg "
        f"(pure-MIT goal). "
        f"Subprocess stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    assert "OK" in result.stdout, (
        f"Subprocess did not reach the success line. stdout:\n{result.stdout}"
    )


def test_detect_communities_still_runs_after_gpl_removal() -> None:
    from iai_mcp.community import detect_communities
    from iai_mcp.graph import MemoryGraph

    g = MemoryGraph()
    clique_a = [uuid4() for _ in range(150)]
    clique_b = [uuid4() for _ in range(150)]
    import random
    for i, n in enumerate(clique_a):
        rng = random.Random(i)
        g.add_node(
            n, community_id=None, embedding=[rng.random() for _ in range(384)]
        )
    for i, n in enumerate(clique_b):
        rng = random.Random(10_000 + i)
        g.add_node(
            n, community_id=None, embedding=[rng.random() for _ in range(384)]
        )
    for i in range(150):
        for j in range(i + 1, 150):
            g.add_edge(clique_a[i], clique_a[j])
            g.add_edge(clique_b[i], clique_b[j])

    a = detect_communities(g, prior=None)
    assert a.backend == "leiden-custom", (
        f"detect_communities MUST route through MOSAIC post-21-07/21-09 "
        f"(no leidenalg-* backend tag should survive). Got: {a.backend}"
    )
    assert a.modularity >= 0.20, (
        f"2-clique N=300 graph should produce CPM-Q >= 0.20; got {a.modularity}"
    )


def _parse_post_removal_wall_time_from_summary() -> float | None:
    if not SUMMARY_MD.exists():
        return None
    text = SUMMARY_MD.read_text()
    pattern = re.compile(
        r"post[- ]removal[^:\n]*?warm[^:\n]*?wall[- ]?time[^:\n]*?:\s*`?(\d+(?:\.\d+)?)\s*s",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    return float(match.group(1))


@pytest.mark.perf_post_removal
def test_leiden09_perf_post_removal() -> None:
    if importlib.util.find_spec("igraph") is not None:
        pytest.skip(
            "Task 3 prerequisite not met: igraph still installed. "
            "Run `pip uninstall -y python-igraph leidenalg` first, then "
            "re-run the LFR gauntlet to record the post-removal wall-time."
        )
    if not SUMMARY_MD.exists():
        pytest.skip(
            "post-removal wall-time summary not present; this perf-gauntlet "
            "check only runs where the summary artifact has been recorded."
        )
    wall_time = _parse_post_removal_wall_time_from_summary()
    if wall_time is None:
        pytest.fail(
            "21-09-SUMMARY.md exists but no `post-removal warm wall-time: "
            "<X> s` line was found. Task 3 must record the LEIDEN-09 "
            "measurement in the SUMMARY for this gate to pass."
        )

    assert wall_time <= 30.0, (
        f"LEIDEN-09 HARD CAP BREACH: post-removal warm wall-time "
        f"{wall_time:.2f}s > 30s. Release-blocker."
    )

    if wall_time >= 5.0:
        import warnings
        warnings.warn(
            f"LEIDEN-09 soft-target miss: post-removal warm wall-time "
            f"{wall_time:.2f}s >= 5s soft target (hard cap 30s still met). "
            f"Acceptable per the 'warm-target soft, "
            f"hard-cap mandatory'.",
            stacklevel=2,
        )
