# iai-pme — Claude Code plugin

Persistent local memory for your assistant. Every turn is captured verbatim,
relevant memory is injected at session start and before each turn, and the
memory tools are available over MCP.

## Install

The plugin carries the wiring; the engine itself is a Python package:

```bash
pip install iai-pme
```

Then add the marketplace and install the plugin:

```
/plugin marketplace add CodeAbra/iai-personal-memory-engine
/plugin install iai-pme@iai-pme
```

Restart the session. Nothing else to configure — capture and recall are
automatic from that point on.

If the engine lives in a virtualenv that isn't first on your `PATH`, point the
plugin at that interpreter:

```bash
export IAI_MCP_PYTHON=/path/to/venv/bin/python
```

## What it wires

| Event | What happens |
|---|---|
| `SessionStart` | Assembles a small slice of relevant memory and injects it before your first prompt. |
| `UserPromptSubmit` | Appends the turn to the session buffer (file IO only, ~5 ms), and serves the foresight pack — memories the engine expects the next turn to need, each marked with its age and how often it has been revised. |
| `Stop` | Rolls the session buffer over for the engine to absorb while idle. |
| MCP | Fifteen memory tools, including cue recall that returns contradictions alongside matches, time-anchored recall, and explicit capture. |

Every hook is fail-safe: an empty store, an absent engine or an unreachable
daemon yields empty output, and the session starts normally.

## Requirements

- macOS or Linux (Windows support is in beta)
- Python 3.11 or 3.12 with `iai-pme` installed
- Node.js 18+ — the MCP server runs on it

## Privacy

Memory lives at `~/.iai-mcp/`, encrypted at rest. Embeddings are computed
locally, there is no account and no telemetry, and the only network traffic is
the model call your assistant already makes.

Full documentation: [the repository README](../README.md).
