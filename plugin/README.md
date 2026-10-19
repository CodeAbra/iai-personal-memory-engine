# iai-pme — Claude Code plugin

Persistent local memory for your assistant. Every turn is captured verbatim,
relevant memory is injected at session start and before each turn, and the
memory tools are available over MCP.

## Install

The plugin carries the wiring; the engine itself is a Python package. Install
the engine version this plugin expects:

```bash
pip install "iai-pme==3.3.0"
```

Then add the marketplace and install the plugin:

```
/plugin marketplace add CodeAbra/iai-personal-memory-engine
/plugin install iai-memory@iai-pme
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

The memory store at `~/.iai-mcp/` is encrypted at rest with AES-256-GCM, and
embeddings are computed locally. There is no account and no telemetry. The
hooks read this host's session transcripts under `~/.claude/projects/` to
capture turns.

Two kinds of files under `~/.iai-mcp/` are not encrypted:

- **The capture spool.** To stay within its few-millisecond budget, the
  per-turn hook appends each prompt and reply as plain text (owner-only, mode
  0600) to `~/.iai-mcp/.deferred-captures/<session>.live.jsonl`. The engine
  encrypts those lines on its next pass (at startup, while you work, and when
  the session goes idle) and then absorbs them into the store. While the
  engine is not running they stay plain text. A store keyed by
  `IAI_MCP_CRYPTO_PASSPHRASE` instead of a key file cannot encrypt the spool,
  so there they stay plain text.
- **The injection caches.** The hooks inject memory from plain-text copies the
  engine keeps next to the store: the session-start brief, session continuity,
  the working-task summary and the next-turn pack (`~/.iai-mcp/*.cached.md`
  and their `.state.json` files). The engine rewrites them as it refreshes
  them, and they stay on disk until you delete the store.

### Network

Beyond the model calls your assistant already makes, the engine goes online in
three places. Each has an off switch; set it where the engine runs, then
restart it with `iai-mcp daemon restart`.

| What | When | Off switch |
|---|---|---|
| Nightly consolidation via `claude -p` | On by default. During the nightly sleep cycle it sends selected memories to Claude through your own Claude Code login to consolidate them, within at most 1% of your daily quota. | `IAI_MCP_REM_DISABLED=1` |
| Version check | Once a day, a request to pypi.org for the latest `iai-pme` release. | `IAI_MCP_VERSION_CHECK=0` |
| Embedding model download | On first use, when the `bge-small-en-v1.5` weights are not yet in the Hugging Face cache (`~/.cache/huggingface/`). | `IAI_MCP_EMBED_OFFLINE=1` (the engine then fails loudly instead of downloading) |

`iai-mcp self-update` downloads a new release only when you run it.

### Your Claude Code login

Each time the MCP server starts it runs `iai-mcp doctor`, and before each
nightly consolidation the engine checks that your Claude Code login is a
subscription. To do that it reads Claude Code's login record:
`~/.claude/.credentials.json`, or on macOS the Claude Code item in your login
Keychain. It looks only at the subscription type, the granted scopes and the
expiry; `iai-mcp doctor` prints the subscription type, and nothing from the
record is sent anywhere. The plugin asks for no keys or tokens of its own.

### Background engine

The MCP server starts the memory engine (`iai-mcp daemon`) in the background
when none is running. The engine keeps running after the session ends and
after the plugin is disabled.

- **Stop it:** disable the plugin first, or the next session starts it again,
  then run `iai-mcp daemon stop`.
- **Login service:** if you installed one with `iai-mcp daemon install`,
  remove it with `iai-mcp daemon uninstall`.
- **Delete all memory:** stop the engine, then delete `~/.iai-mcp/` (the store
  and its key). The embedding model in `~/.cache/huggingface/` can be deleted
  too.

Full documentation: [the repository README](../README.md).
