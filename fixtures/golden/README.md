# Golden fixtures

Frozen reference artifacts that pin the behaviour of the Rust components against
the proven reference implementation. Every golden is regenerated deterministically
by a dumper under `scripts/golden/` and asserted byte-for-byte by the Rust loader.

## Format contract

A golden is a pair of sibling files sharing one stem:

- `<stem>.bin` — the raw payload.
- `<stem>.json` — a manifest describing the payload and carrying its `sha256`.

Two payload shapes are supported, selected by the manifest `format` field:

- `raw_bytes` — a numeric array serialised as raw **little-endian**, C-contiguous
  bytes in `<stem>.bin`. This is the implemented path. `payload` is `null`.
- `json` — reserved for structured data carried inline in the manifest `payload`
  field. `<stem>.bin` is still written (and hashed) so the loader contract stays
  uniform; the structured value also lives in `payload`.

### Manifest schema

```json
{
  "name": "projection",
  "subsystem": "hdc",
  "format": "raw_bytes",
  "payload": null,
  "dtype": "f32",
  "shape": [384, 10000],
  "byte_order": "le",
  "sha256": "<hex of the .bin contents>",
  "source": "<path the data was derived from>",
  "generated_by": "<path of the dumper script>"
}
```

| field          | meaning                                                              |
| -------------- | ------------------------------------------------------------------- |
| `name`         | logical golden name                                                  |
| `subsystem`    | owning subsystem (e.g. `hdc`)                                        |
| `format`       | `raw_bytes` or `json`                                                |
| `payload`      | inline structured value when `format == "json"`, else `null`        |
| `dtype`        | element type of the array (`f32`, `u8`, ...)                         |
| `shape`        | array dimensions; `product(shape) * sizeof(dtype)` == `.bin` length  |
| `byte_order`   | always `le` for the raw_bytes path                                   |
| `sha256`       | hex digest over the exact bytes of `<stem>.bin`                      |
| `source`       | where the data was derived from                                      |
| `generated_by` | dumper that produced the pair                                       |

## Loader guarantees

On every load the Rust loader:

1. reads the manifest,
2. reads the sibling `.bin`,
3. asserts `bin.len() == product(shape) * sizeof(dtype)`,
4. re-hashes the `.bin` and asserts it equals `manifest.sha256`,
5. asserts `byte_order == "le"`.

A tampered or corrupted `.bin` fails the hash or length check and the load errors.

## Layout

- `hdc/projection.{bin,json}` — the frozen projection matrix `P[384, 10000]`
  (`f32`, little-endian, 15,360,000 bytes).
- `_rails/round_trip.{bin,json}` — a synthetic 16-byte fixture exercising the
  loader end to end.

## Regenerating

Run a dumper under the project virtualenv, for example:

```bash
source .venv/bin/activate
python scripts/golden/dump_projection.py
python scripts/golden/dump_rails.py
```

Each dumper asserts the payload's identity (shape, length, and — for the
projection — its locked sha256) before writing, so a regeneration that drifts
fails loudly instead of overwriting a good golden.
