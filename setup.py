"""Build configuration hook for the iai-mcp wheel.

All project metadata lives in pyproject.toml.  This file exists solely to
register a custom build_py subclass that:
- compiles the TypeScript MCP wrapper and stages the resulting JS files into
  the wheel build directory (build_lib) before the wheel is assembled;
- stages the native-extension type stubs (*.pyi, py.typed) from the Rust
  workspace beside the compiled extension in the wheel.

Both operations write into build_lib, never into the source tree, so an
editable checkout stays clean.

Editable installs (pip install -e .) skip the npm build entirely.  The owner's
install script (scripts/install.sh) builds the wrapper separately; the resolver
falls back to mcp-wrapper/dist/ on an editable install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _OrigBuildPy
from setuptools_rust import Binding, RustExtension

# Platform-conditional native features (why this lives here and not in
# pyproject's static ext-modules table): on macOS the embedder's matmuls
# MUST link Apple Accelerate BLAS — without it a 512-token encode runs a
# naive matmul and takes seconds instead of milliseconds, starving every
# consumer of the embedder. accelerate-src does not build on Linux, so the
# feature is added only on darwin.
_NATIVE_FEATURES = ["extension-module"]
if sys.platform == "darwin":
    _NATIVE_FEATURES.append("accelerate")

_REPO_ROOT = Path(__file__).parent
_WRAPPER_SRC = _REPO_ROOT / "mcp-wrapper"

# Tracked type stubs for the native extension; staged flat beside the
# compiled .so in the wheel so that `Path(iai_mcp_native.__file__).parent`
# finds them after installation.
_NATIVE_STUB_SRC = _REPO_ROOT / "rust" / "iai_mcp_native" / "iai_mcp_native"
_NATIVE_STUB_FILES = [
    # Primary stub for the flat-layout wheel (importable as iai_mcp_native.pyi).
    ("__init__.pyi", "iai_mcp_native.pyi"),
    ("embed.pyi", "embed.pyi"),
    ("graph.pyi", "graph.pyi"),
    ("py.typed", "py.typed"),
]


class _BuildWithWrapper(_OrigBuildPy):
    """build_py subclass that compiles the TS wrapper and stages native stubs.

    At wheel-build time: collects the package into build_lib (via the parent
    ``build_py``), runs ``npm ci && npm run build`` inside mcp-wrapper/, then
    stages the resulting JS files into build_lib/iai_mcp/_wrapper/ and stages
    the native extension type stubs flat into build_lib/ so they ship beside
    the compiled extension.  The source tree is never touched.

    At editable-install time (``pip install -e .``): returns immediately without
    touching npm.  The editable resolver finds the wrapper via mcp-wrapper/dist/
    and the stubs via the maturin editable package directory.
    """

    def run(self) -> None:
        # FIRM REQUIREMENT: editable installs must never trigger npm.
        # The owner's install script builds the wrapper as a separate step.
        if self.editable_mode:
            super().run()
            return

        # Collect the package into build_lib FIRST so build_lib/iai_mcp/ exists,
        # THEN stage the freshly compiled JS into build_lib/iai_mcp/_wrapper/
        # and the native stubs flat into build_lib/.
        # Staging into build_lib (not the source tree) keeps the checkout clean.
        super().run()
        self._build_ts_wrapper()
        self._stage_native_stubs()

    def _build_ts_wrapper(self) -> None:
        """Compile the TypeScript wrapper and stage the JS output into build_lib.

        When npm is unavailable (containerized wheel builds: manylinux and
        friends have no Node), a prebuilt ``mcp-wrapper/dist/`` compiled on
        the host is staged as-is. ``IAI_MCP_WRAPPER_PREBUILT=1`` forces the
        prebuilt path even when npm exists. With npm present and no force
        flag, the wrapper is always recompiled so a dev checkout can never
        package stale JS."""
        if not (_WRAPPER_SRC / "package.json").exists():
            raise RuntimeError(
                "mcp-wrapper/package.json not found.  The TypeScript source must be "
                "present to build the MCP wrapper.  If you are building from an sdist, "
                "ensure MANIFEST.in includes the mcp-wrapper source."
            )

        dist_dir = _WRAPPER_SRC / "dist"
        npm_exe = shutil.which("npm")
        force_prebuilt = os.environ.get("IAI_MCP_WRAPPER_PREBUILT") == "1"

        bundle_file = _WRAPPER_SRC / "dist-bundle" / "index.js"

        if npm_exe is not None and not force_prebuilt:
            # Install exact locked dependencies.
            subprocess.run(
                [npm_exe, "ci", "--prefer-offline"],
                cwd=str(_WRAPPER_SRC),
                check=True,
            )

            # Compile TypeScript → dist/*.js
            subprocess.run(
                [npm_exe, "run", "build"],
                cwd=str(_WRAPPER_SRC),
                check=True,
            )

            if not dist_dir.exists():
                raise RuntimeError(
                    f"Expected {dist_dir} after 'npm run build' but the directory is absent.  "
                    "Check the TypeScript compiler output above for errors."
                )

            # Bundle → dist-bundle/index.js. The wheel must ship a
            # SELF-CONTAINED wrapper: the tsc tree keeps bare imports
            # (@modelcontextprotocol/sdk, zod) that no wheel install can
            # resolve — a staged tsc tree means every wheel user's MCP
            # registration dies with ERR_MODULE_NOT_FOUND.
            subprocess.run(
                [npm_exe, "run", "bundle"],
                cwd=str(_WRAPPER_SRC),
                check=True,
            )
            if not bundle_file.exists():
                raise RuntimeError(
                    f"Expected {bundle_file} after 'npm run bundle' but it is absent.  "
                    "Check the esbuild output above for errors."
                )
        elif dist_dir.exists():
            print(
                f"npm {'forced off' if force_prebuilt else 'not found'}; "
                f"staging prebuilt MCP wrapper from {dist_dir}"
            )
        else:
            raise RuntimeError(
                "Node.js/npm is required to build the MCP wrapper.  "
                "Install Node.js >=18 and ensure 'npm' is on your PATH, or "
                "compile the wrapper on the host first (npm ci && npm run "
                "build in mcp-wrapper/) so mcp-wrapper/dist/ ships prebuilt "
                "into this build environment."
            )

        # Stage into the wheel build directory (build_lib/iai_mcp/_wrapper/)
        # — never into the source tree, so an editable checkout stays clean
        # and its resolver falls back to mcp-wrapper/dist/ as intended.
        wrapper_dest = Path(self.build_lib) / "iai_mcp" / "_wrapper"
        if wrapper_dest.exists():
            shutil.rmtree(wrapper_dest)
        wrapper_dest.mkdir(parents=True)

        if bundle_file.exists():
            # The self-contained bundle is the only wrapper a wheel may ship.
            shutil.copy2(bundle_file, wrapper_dest / "index.js")
        else:
            # Prebuilt host provided only the tsc tree (no bundle). Ship it
            # with a loud warning: bare imports cannot resolve from a wheel.
            print(
                "WARN: mcp-wrapper/dist-bundle/index.js absent — staging the "
                "tsc tree, whose bare imports do NOT resolve from a wheel "
                "install. Run 'npm run bundle' on the build host.",
                file=sys.stderr,
            )
            js_files = sorted(dist_dir.glob("*.js"))
            if not js_files:
                raise RuntimeError(
                    f"No *.js files found in {dist_dir}.  "
                    "The 'npm run build' step produced no output."
                )
            for js_file in js_files:
                shutil.copy2(js_file, wrapper_dest / js_file.name)

    def _stage_native_stubs(self) -> None:
        """Stage the native extension type stubs flat into build_lib.

        setuptools-rust places the compiled extension flat at the build_lib
        root (i.e. ``build_lib/iai_mcp_native.cpython-*.so``), so stubs must
        land at the same level to be found via
        ``Path(iai_mcp_native.__file__).parent`` after installation.
        """
        build_lib_root = Path(self.build_lib)
        missing = []
        for src_name, dest_name in _NATIVE_STUB_FILES:
            src = _NATIVE_STUB_SRC / src_name
            if src.exists():
                shutil.copy2(src, build_lib_root / dest_name)
            else:
                missing.append(str(src))
        if missing:
            raise RuntimeError(
                "Native extension stub(s) missing from "
                f"{_NATIVE_STUB_SRC}: {missing}. A wheel must never ship "
                "without native stubs; regenerate via "
                "'cargo run --bin stub_gen -p iai_mcp_native --features stubgen' "
                "and flatten "
                "the staged output into the tracked stub files."
            )


setup(
    cmdclass={"build_py": _BuildWithWrapper},
    rust_extensions=[
        RustExtension(
            "iai_mcp_native",
            path="rust/iai_mcp_native/Cargo.toml",
            binding=Binding.PyO3,
            features=_NATIVE_FEATURES,
            args=["--no-default-features"],
            # ALWAYS release: setuptools-rust defaults in-place/develop builds
            # to the dev profile, and an unoptimized engine must never be the
            # binary a daemon actually loads.
            debug=False,
        )
    ],
)
