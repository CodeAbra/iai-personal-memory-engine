# Replaceable embedding providers

iai-mcp uses the native Rust `bge-small-en-v1.5` embedder by default. A local
HTTP provider can replace it completely for multilingual, domain-specific, or
shared-model deployments. iai-mcp does not import or construct the native
embedder when the HTTP provider is selected.

## Configuration

```bash
export IAI_MCP_EMBED_PROVIDER=http
export IAI_MCP_EMBED_URL=http://127.0.0.1:4488
export IAI_MCP_EMBED_DIM=1024
export IAI_MCP_EMBED_MODEL_ID=your-model-id
export IAI_MCP_EMBED_TIMEOUT_SEC=30
```

Only unauthenticated loopback HTTP URLs are accepted. The model service stays
local, and iai-mcp adds no ML framework dependency. Multiple iai-mcp processes
can share one warm model service instead of loading one model copy per store.

## Protocol

iai-mcp sends `POST /embed` unless `IAI_MCP_EMBED_URL` already ends in
`/embed`:

```json
{
  "texts": ["What should be recalled?"],
  "input_type": "query"
}
```

`input_type` is either `query` for retrieval cues or `document` for memories.
The provider owns tokenization, prefixes, pooling, normalization, batching, and
model loading. This distinction supports asymmetric models without teaching
iai-mcp about any specific model family.

The response is:

```json
{
  "model": "your-model-id",
  "dimensions": 1024,
  "vectors": [[0.01, -0.02, 0.03]]
}
```

iai-mcp rejects a wrong model identifier, vector count, dimension, non-numeric
value, or non-finite value.

## Migrating an existing store

Changing a model invalidates every stored vector, even when the old and new
models use the same dimension. Stop the daemon, make a store backup, configure
the new provider, and run:

```bash
iai-mcp migrate --reembed-to-configured-provider --dry-run
iai-mcp migrate --reembed-to-configured-provider
```

The migration flushes pending in-process writes, stages a complete replacement
table, preserves storage-only fields and encrypted payloads, keeps the previous
records table, and updates the persisted dimension. Restart iai-mcp before
recall so its vector indexes reopen with the new dimension, then run:

```bash
iai-mcp doctor
```

Starting a populated store with a provider whose dimension differs from the
store fails fast and points to the migration command. This prevents mixed-model
or mixed-dimension indexes from being used silently.
