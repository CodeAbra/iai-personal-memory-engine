# Contributing to iai-mcp

Thanks for considering a contribution. This project is small and opinionated. Read this before opening a large PR.

## Scope

iai-mcp is a local memory layer for MCP-over-stdio hosts. Contributions that fit the scope:

- Bug fixes in capture, recall, consolidation, daemon lifecycle, or the MCP wrapper.
- New bench harnesses, or improvements to existing ones.
- Documentation fixes and clarifications.
- Platform support. macOS, Linux, and Windows all run the daemon; fixes and
  coverage for any of them are welcome.
- Compatibility with additional MCP-over-stdio hosts (with evidence it was tested end to end).

Contributions outside scope (will likely be declined):

- Cloud sync, remote storage, or any network-exposed surface beyond the existing local UNIX socket.
- Major architectural rewrites without prior discussion in an issue.
- Replacing the three-tier memory model (episodic / semantic / procedural).
- Telemetry of any kind.

If you're not sure, open an issue first to discuss before writing code.

## Development setup

```bash
git clone https://github.com/CodeAbra/iai-personal-memory-engine.git
cd iai-personal-memory-engine
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The storage, vector, and graph engines are Rust; `pip install -e` builds them on
a fresh checkout. After editing anything under `rust/`, rebuild explicitly —
reinstalling can silently keep the previously built extension:

```bash
python setup.py build_ext --inplace
```

Build the TypeScript MCP wrapper:

```bash
cd mcp-wrapper
npm install
npm run build
cd ..
```

## Tests

```bash
pytest
```

A handful of daemon and bridge tests are sensitive to test-pollution and may pass cleanly on a rerun. The harness retries up to twice automatically. If a test fails three times, it's a real failure.

For changes to a specific area, the relevant test files:

- Capture: `tests/test_capture_*`
- Recall: `tests/test_recall_*`
- Consolidation: `tests/test_consolidation_*`, `tests/test_sleep_*`
- Daemon lifecycle: `tests/test_daemon_*`, `tests/test_fsm_*`
- MCP wrapper: `mcp-wrapper/test/`

## Lint

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

PRs are expected to pass `ruff check` clean.

## Benchmarks

If your change touches retrieval, capture, or consolidation, include before/after numbers from the relevant bench in the PR description:

```bash
python -m bench.verbatim
python -m bench.neural_map
python -m bench.tokens
python -m bench.longmemeval_blind --split S --out /tmp/out.json
```

Don't tune to the bench. The LongMemEval-S run is blind on purpose.

## Commit style

- Imperative mood: `Fix X`, `Add Y`, not `Fixed X` or `Adding Y`.
- One logical change per commit.
- Keep messages short. One-line subject, body only when motivation is non-obvious.
- Credit only people. Use an assistant if it helps you, but strip what it adds to
  the message before you commit: no `Co-Authored-By:` line naming a tool or
  model, and no "generated / refined / written by" note in the body. You are the
  author of what you submit, and the history should read that way.

Check a message before pushing:

```bash
git log -1 --format='%B' | grep -iE 'co-authored-by|generated (with|by)|refined by'
```

Anything it prints does not belong in the commit. Amend it out.

## Pull requests

- Small PRs land faster than large ones.
- Reference the issue your PR addresses.
- Include test coverage for changed behaviour, or explain in the PR description why it isn't applicable.
- For retrieval / capture / consolidation changes, include bench re-runs.

The PR template will prompt for the relevant items.


## Reporting issues that involve security

Do not open a public issue for anything that looks like a security defect. See [SECURITY.md](SECURITY.md) for the private reporting flow.

## Code of conduct

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
