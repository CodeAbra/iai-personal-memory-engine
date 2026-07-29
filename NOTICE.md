# NOTICE — Third-party attributions

This project includes the following third-party software. Each is used under
its own license; the license name, upstream URL, and authorship for every
runtime dependency are listed below. The project itself is MIT-licensed (see
`LICENSE`); the license-compatibility summary at the foot of this file confirms
every runtime dep is MIT, BSD, Apache-2.0, or PSF and therefore compatible.

For Python deps, version-pin information lives in `pyproject.toml`; the
versions listed below are the resolved versions observed at the time of the
release-hardening pass. For npm deps, direct
declarations live in `mcp-wrapper/package.json` and resolved versions in
`mcp-wrapper/package-lock.json`.

## Python runtime dependencies

| Package               | Resolved version | License             | Author / maintainer                          | Upstream URL                                              |
| --------------------- | ---------------- | ------------------- | -------------------------------------------- | --------------------------------------------------------- |
| anthropic             | 0.102.0          | MIT                 | Anthropic                                    | https://github.com/anthropics/anthropic-sdk-python        |
| cachetools            | 7.1.1            | MIT                 | Thomas Kemmer                                | https://github.com/tkem/cachetools/                       |
| cryptography          | 48.0.0           | Apache-2.0 or BSD-3 | Python Cryptographic Authority + contributors | https://github.com/pyca/cryptography                      |
| keyring               | 25.7.0           | MIT                 | Kang Zhang (Jason R. Coombs maintains)       | https://github.com/jaraco/keyring                         |
| lancedb               | 0.30.2           | Apache-2.0          | LanceDB devs                                 | https://github.com/lancedb/lancedb                        |
| networkx              | 3.6.1            | BSD-3-Clause        | Aric Hagberg et al.                          | https://networkx.org/                                     |
| numba                 | 0.65.1           | BSD-2-Clause        | Numba project (Anaconda Inc.)                | https://numba.pydata.org                                  |
| numpy                 | 2.2.6            | BSD-3-Clause        | Travis Oliphant et al.                       | https://numpy.org                                         |
| psutil                | 7.2.2            | BSD-3-Clause        | Giampaolo Rodola                             | https://github.com/giampaolo/psutil                       |
| pyarrow               | 24.0.0           | Apache-2.0          | Apache Arrow project                         | https://arrow.apache.org/                                 |
| pydantic              | 2.13.4           | MIT                 | Samuel Colvin et al.                         | https://github.com/pydantic/pydantic                      |
| structlog             | 25.5.0           | MIT or Apache-2.0   | Hynek Schlawack                              | https://github.com/hynek/structlog                        |
| tiktoken              | 0.13.0           | MIT                 | OpenAI (Shantanu Jain)                       | https://github.com/openai/tiktoken                        |
| torch-hd              | 5.8.4            | MIT                 | Hyperdimensional Computing org               | https://github.com/hyperdimensional-computing/torchhd     |

## Python optional dependencies

These are installable via `pip install -e .[compress]` or `pip install -e
.[dev]` but are NOT pulled in by default. The `compress` extra is opt-in
because the underlying model weights are large; the `dev` extra is for test
runs only.

### `compress` extra (opt-in)

| Package    | Resolved version | License    | Upstream URL                                |
| ---------- | ---------------- | ---------- | ------------------------------------------- |
| llmlingua  | varies           | MIT        | https://github.com/microsoft/LLMLingua      |
| accelerate | varies           | Apache-2.0 | https://github.com/huggingface/accelerate   |

### `dev` extra (test-only, not shipped at runtime)

| Package     | Resolved version | License | Upstream URL                          |
| ----------- | ---------------- | ------- | ------------------------------------- |
| pytest      | >=8.0            | MIT     | https://github.com/pytest-dev/pytest  |
| pytest-cov  | >=5.0            | MIT     | https://github.com/pytest-dev/pytest-cov |
| ruff        | >=0.5.0          | MIT     | https://github.com/astral-sh/ruff     |

## TypeScript wrapper runtime dependencies

The `mcp-wrapper/` subdirectory contains the TypeScript MCP wrapper. Its
runtime dependencies (NOT devDependencies, which are build-only and not
shipped) are:

| Package                       | Version pin (package.json) | License | Upstream URL                                                |
| ----------------------------- | -------------------------- | ------- | ----------------------------------------------------------- |
| @modelcontextprotocol/sdk     | ^1.0.0                     | MIT     | https://github.com/modelcontextprotocol/typescript-sdk      |
| zod                           | ^3.23.0                    | MIT     | https://github.com/colinhacks/zod                           |

The wheel ships the wrapper as a single esbuild bundle
(`iai_mcp/_wrapper/index.js`) with `@modelcontextprotocol/sdk` and `zod`
inlined — that artifact REDISTRIBUTES both packages, and their MIT license
texts travel with this NOTICE.

The wrapper's `devDependencies` (`@types/node`, `typescript`, `tsx`,
`esbuild`) are build-time only — they are not bundled into the shipped wrapper
artifact and therefore do not require runtime attribution. They are nonetheless
listed in `mcp-wrapper/package.json` for transparency.

## License compatibility summary

Every runtime dependency above is licensed under one of: **MIT**, **BSD-2** or
**BSD-3**, **Apache-2.0**, or a permissive dual-license that includes one of
the above (e.g. `cryptography` is "Apache-2.0 OR BSD-3-Clause"; `structlog` is
"MIT OR Apache-2.0").

All of these are compatible with this project's MIT license declared in
`LICENSE`. There are no copyleft (GPL / LGPL / AGPL) runtime dependencies in
the shipped distribution.

A custom MIT-licensed Leiden implementation removed the
project's previous dependence on `leidenalg` and `python-igraph`, both of
which were copyleft-licensed. The replacement MOSAIC community-detection
backend (`src/iai_mcp/mosaic.py`) is original code under the project's MIT
license, with the Numba JIT runtime (BSD-2-Clause) as its only Numba-specific
dep. After that change the repository is fully MIT-compatible for both static
distribution and dynamic linking.

## Updating this file

When `pyproject.toml [project] dependencies` or `mcp-wrapper/package.json
dependencies` changes, regenerate this file. The fastest path is:

```
pip install pip-licenses
pip-licenses --packages lancedb pyarrow numpy pydantic \
  torch-hd structlog networkx numba anthropic tiktoken \
  cryptography keyring cachetools psutil \
  --format=markdown --with-urls --with-authors
```

then merge the resulting table into the `## Python runtime dependencies`
section above. For npm, `mcp-wrapper/package.json` is short enough to update
the wrapper table by hand.

The regression-gate test in `tests/test_release_hardening.py` enforces that
every name in `pyproject.toml [project] dependencies` and the two direct
wrapper deps appear by substring somewhere in this file.
