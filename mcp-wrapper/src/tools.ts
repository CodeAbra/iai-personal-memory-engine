
import type { PythonCoreBridge } from "./bridge.js";

import { spawn, type SpawnOptions } from "node:child_process";

export const BANK_FALLBACK_LIMIT = 20;

export const TOOL_NAMES = [
  "memory_recall",
  "memory_recall_structural",
  "memory_reinforce",
  "memory_contradict",
  "memory_capture",
  "memory_consolidate",
  "profile_get_set",
  "curiosity_pending",
  "schema_list",
  "events_query",
  "topology",
  "camouflaging_status",
  "episodes_recent",
] as const;

export type ToolName = (typeof TOOL_NAMES)[number];

interface ToolAnnotations {
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
}

interface ToolSchema {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  outputSchema?: Record<string, unknown>;
  annotations?: ToolAnnotations;
}

export const toolSchemas: Record<ToolName, ToolSchema> = {
  memory_recall: {
    name: "memory_recall",
    description:
      "Recall verbatim memories by cue. Returns hits + anti_hits with derived valid_from/valid_to. Read-only.",
    inputSchema: {
      type: "object",
      properties: {
        cue: {
          type: "string",
          description:
            "Natural-language query to match against stored memories. " +
            "Embedded server-side via bge-small-en-v1.5 (384d) unless " +
            "`cue_embedding` is supplied.",
        },
        budget_tokens: {
          type: "integer",
          description:
            "Soft token budget for the response (default 1500). Hits are " +
            "appended until the next would exceed this budget; at least " +
            "one hit is always returned.",
          default: 1500,
        },
        session_id: {
          type: "string",
          description:
            "Current session id; gets written into every recalled record's " +
            "provenance. Omit to use '-'.",
        },
        cue_embedding: {
          type: "array",
          items: { type: "number" },
          description:
            "Optional pre-computed embedding vector for the cue " +
            "(EMBED_DIM=384 floats; bge-small-en-v1.5). " +
            "When omitted, the daemon embeds the cue server-side. " +
            "Used by memory_contradict and tests that need byte-stable embeddings.",
        },
        language: {
          type: "string",
          description:
            "Optional ISO-639-1 language hint for the sleep-suggestion path " +
            "(8 supported: en/ru/ja/ar/de/fr/es/zh). Defaults to 'en' " +
            "when omitted. Hot-path retrieval is language-agnostic; this " +
            "key only affects the sleep-suggestion regex pre-screen.",
        },
      },
      required: ["cue"],
    },
    outputSchema: {
      type: "object",
      properties: {
        hits: { type: "array", items: { type: "object" } },
        anti_hits: { type: "array", items: { type: "object" } },
        activation_trace: { type: "array", items: { type: "string" } },
        budget_used: { type: "integer" },
        hints: { type: "array", items: { type: "object" } },
        cue_mode: { type: "string", enum: ["verbatim", "concept"] },
        patterns_observed: { type: "array", items: { type: "object" } },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  memory_reinforce: {
    name: "memory_reinforce",
    description:
      "Boost Hebbian edges among co-retrieved record ids. Mutates edge weights. Use when two records co-answered.",
    inputSchema: {
      type: "object",
      properties: {
        ids: {
          type: "array",
          items: { type: "string", format: "uuid" },
          description:
            "Record UUIDs that were co-retrieved in the current context. " +
            "Edges between every pair are incremented; identical pair sets " +
            "are idempotent within one session.",
        },
      },
      required: ["ids"],
    },
    outputSchema: {
      type: "object",
      properties: {
        edges_boosted: { type: "integer" },
        new_weights: {
          type: "object",
          additionalProperties: { type: "number" },
        },
      },
    },
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  memory_contradict: {
    name: "memory_contradict",
    description:
      "Mark a record contradicted; new fact stored as a NEW record (old NEVER deleted). Mutates store.",
    inputSchema: {
      type: "object",
      properties: {
        id: {
          type: "string",
          format: "uuid",
          description: "UUID of the record being contradicted.",
        },
        new_fact: {
          type: "string",
          description:
            "The updated verbatim fact. Stored as a new record; the old " +
            "record is preserved (episodic write-once) and linked via a " +
            "`contradicts` edge.",
        },
        cue_embedding: {
          type: "array",
          items: { type: "number" },
          description:
            "Optional pre-computed embedding vector for the contradicting " +
            "fact (EMBED_DIM=384 floats; bge-small-en-v1.5). When omitted, " +
            "the daemon embeds new_fact server-side.",
        },
      },
      required: ["id", "new_fact"],
    },
    outputSchema: {
      type: "object",
      properties: {
        original_id: { type: "string", format: "uuid" },
        new_record_id: { type: "string", format: "uuid" },
        edge_type: { type: "string" },
        ts: { type: "string", format: "date-time" },
      },
    },
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: false,
    },
  },
  memory_capture: {
    name: "memory_capture",
    description:
      "Capture a verbatim turn. Auto-dedups at cos>=0.95 (reinforces). " +
      "Use for corrections + load-bearing decisions.",
    inputSchema: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description:
            "Verbatim text to capture (user utterance, Claude decision, or observation). " +
            "Min 12 chars, max 8000 (longer is truncated).",
        },
        cue: {
          type: "string",
          description:
            "Short natural-language cue used for embedding + dedup lookup. " +
            "If empty, `text` itself is embedded.",
        },
        tier: {
          type: "string",
          enum: ["working", "episodic", "semantic", "procedural", "parametric"],
          default: "episodic",
          description:
            "Memory tier. Default 'episodic' (verbatim user utterances). " +
            "Use 'semantic' for induced summaries, 'procedural' for learned behaviour notes.",
        },
        session_id: {
          type: "string",
          description: "Current session id for provenance.",
        },
        role: {
          type: "string",
          enum: ["user", "assistant", "system"],
          default: "user",
          description: "Who produced this turn — tags the record for filtering.",
        },
      },
      required: ["text"],
    },
    outputSchema: {
      type: "object",
      properties: {
        status: {
          type: "string",
          enum: ["inserted", "reinforced", "skipped"],
        },
        record_id: { type: "string", format: "uuid" },
        reason: { type: "string" },
      },
    },
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: false,
    },
  },
  memory_consolidate: {
    name: "memory_consolidate",
    description:
      "Trigger sleep-cycle consolidation: schema induction, FSRS decay, Hebbian pruning. Mutates store; idempotent in one sleep window.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description:
            "Optional session id used for provenance tagging on the " +
            "consolidate event. Defaults to '-' when omitted.",
        },
      },
    },
    outputSchema: {
      type: "object",
      properties: {
        mode: { type: "string" },
        tier: { type: "string" },
        summaries_created: { type: "integer" },
        decay_result: { type: "object" },
        schema_candidates: { type: "array" },
      },
    },
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  profile_get_set: {
    name: "profile_get_set",
    description:
      "Read or write a profile knob (11 sealed: 10 AUTIST + wake_depth). operation get|set; returns knob value.",
    inputSchema: {
      type: "object",
      properties: {
        operation: {
          type: "string",
          enum: ["get", "set"],
          description:
            "Whether to read or write a knob. 'get' with no `knob` returns " +
            "all live + deferred knob values; 'set' requires both `knob` " +
            "and `value`.",
        },
        knob: {
          type: "string",
          description:
            "Knob name. Omit on 'get' to retrieve all live + deferred knobs. " +
            "Required on 'set'.",
        },
        value: {
          description:
            "New value when operation='set'. Any JSON-serialisable type " +
            "matching the knob's declared type in the sealed registry.",
        },
      },
      required: ["operation"],
    },
    outputSchema: {
      type: "object",
    },
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  curiosity_pending: {
    name: "curiosity_pending",
    description:
      "List pending curiosity questions queued by the sleep daemon. Read-only. Filter by session_id.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description:
            "Only return questions from this session. Omit to return " +
            "questions from every session in the queue.",
        },
      },
    },
    outputSchema: {
      type: "object",
      properties: {
        questions: { type: "array", items: { type: "object" } },
        count: { type: "integer" },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  schema_list: {
    name: "schema_list",
    description:
      "List induced schemas (Tier-0 + Tier-1) from sleep consolidation. Read-only. Filter by domain and confidence_min.",
    inputSchema: {
      type: "object",
      properties: {
        domain: {
          type: "string",
          description:
            "Only return schemas tagged with this domain (e.g. 'coding'). " +
            "Omit to return schemas across all domains.",
        },
        confidence_min: {
          type: "number",
          description:
            "Minimum parsed confidence (0.0-1.0). Default 0.0 returns all " +
            "schemas; raise to 0.5+ to filter out low-evidence candidates.",
          default: 0.0,
        },
      },
    },
    outputSchema: {
      type: "object",
      properties: {
        schemas: { type: "array", items: { type: "object" } },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  events_query: {
    name: "events_query",
    description:
      "Query user-visible events (kind whitelist). Read-only. Optional since (ISO-8601), severity, limit.",
    inputSchema: {
      type: "object",
      properties: {
        kind: {
          type: "string",
          description:
            "Event kind. Must be in the whitelist " +
            "(s4_contradiction, trajectory_metric, ...).",
        },
        since: {
          type: "string",
          description:
            "ISO-8601 timestamp; only events at or after this are returned. " +
            "Omit to return events from the start of the log.",
        },
        severity: {
          type: "string",
          enum: ["info", "warning", "critical"],
          description:
            "Optional severity filter. Omit to return all severities.",
        },
        limit: {
          type: "integer",
          description:
            "Maximum events returned (default 100, capped at 1000 by " +
            "the daemon regardless of the value supplied).",
          default: 100,
        },
      },
      required: ["kind"],
    },
    outputSchema: {
      type: "object",
      properties: {
        events: { type: "array", items: { type: "object" } },
        count: { type: "integer" },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  memory_recall_structural: {
    name: "memory_recall_structural",
    description:
      "Structural recall via TEM role->filler bindings (BSC hypervectors). Read-only. Prefer over memory_recall for role-filler queries.",
    inputSchema: {
      type: "object",
      properties: {
        structure_query: {
          type: "object",
          description:
            "Optional role->filler map, e.g. {\"agent\": \"agent_name\"}. Each value is hashed to a filler hypervector. When omitted or empty, query HV is zero-filled and every row with structure_hv is scored (expensive at large N).",
          additionalProperties: { type: "string" },
        },
        budget_tokens: {
          type: "integer",
          description:
            "Soft token budget for the response (default 2000). Hits are " +
            "appended until the next would exceed this budget.",
          default: 2000,
        },
        max_records: {
          type: "integer",
          description:
            "Hard cap on records scanned after fetch (default 5000, max 50000). Prevents accidental full-corpus scans from `{}`.",
          default: 5000,
        },
      },
      required: [],
    },
    outputSchema: {
      type: "object",
      properties: {
        hits: { type: "array", items: { type: "object" } },
        anti_hits: { type: "array", items: { type: "object" } },
        activation_trace: { type: "array", items: { type: "string" } },
        budget_used: { type: "integer" },
        structural_query_size: { type: "integer" },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  topology: {
    name: "topology",
    description:
      "Snapshot of memory-graph topology: N, C, L, sigma, community_count, regime, source/as_of/age_s. Read-only diagnostic; sigma never toggles retrieval.",
    inputSchema: { type: "object", properties: {} },
    outputSchema: {
      type: "object",
      properties: {
        N: { type: "integer" },
        C: { type: "number" },
        L: { type: "number" },
        sigma: { type: ["number", "null"] },
        community_count: { type: "integer" },
        rich_club_ratio: { type: "number" },
        regime: { type: "string" },
        source: { type: "string", enum: ["live", "cached", "insufficient"] },
        as_of: { type: ["string", "null"] },
        age_s: { type: ["number", "null"] },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  camouflaging_status: {
    name: "camouflaging_status",
    description:
      "Detect formality/register camouflaging via weekly trajectory points (window_size). Read-only detector; does not relax register.",
    inputSchema: {
      type: "object",
      properties: {
        window_size: {
          type: "integer",
          description:
            "Weekly points in the sliding window (default 5). Larger " +
            "windows smooth the formality trend at the cost of " +
            "responsiveness to recent register shifts.",
          default: 5,
        },
      },
    },
    outputSchema: {
      type: "object",
      properties: {
        detected: { type: "boolean" },
        trajectory_slope: { type: "number" },
        current_mean: { type: "number" },
        sample_count: { type: "integer" },
        camouflaging_relaxation: { type: "number" },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  episodes_recent: {
    name: "episodes_recent",
    description:
      "Returns the N most-recent user-turn records, time-desc. " +
      "Optional session_id filter. GLOBAL across all projects.",
    inputSchema: {
      type: "object",
      properties: {
        n: {
          type: "integer",
          description: "How many turns to return (default 10, max 1000).",
        },
        session_id: {
          type: "string",
          description: "Filter to a specific session UUID.",
        },
      },
    },
    outputSchema: {
      type: "object",
      properties: {
        turns: { type: "array", items: { type: "object" } },
        count: { type: "integer" },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
};

function isDaemonDownError(err: unknown): boolean {
  if (err instanceof Error) {
    if (err.name === "DaemonUnreachableError") return true;
    const msg = err.message;
    if (
      msg.includes("daemon_unreachable") ||
      msg.includes("socket dead") ||
      msg.includes("DaemonUnreachable") ||
      msg.includes("ECONNREFUSED") ||
      msg.includes("ENOENT") ||
      msg.includes("connect ETIMEDOUT")
    ) {
      return true;
    }
  }
  return false;
}

export async function runDirectRecency(
  args: Record<string, unknown>,
  spawnFn: typeof spawn = spawn,
): Promise<Record<string, unknown> | null> {
  const cli = process.env.IAI_MCP_CLI ?? "iai";
  const n = String(args.n ?? 10);
  const spawnArgs: string[] = ["last", "--json", "--n", n];
  const sessionId = args.session_id;
  if (sessionId && typeof sessionId === "string") {
    spawnArgs.push("--session", sessionId);
  }
  return new Promise((resolve) => {
    const opts: SpawnOptions = { stdio: ["ignore", "pipe", "pipe"] };
    const proc = spawnFn(cli, spawnArgs, opts);
    let stdout = "";
    const stdoutStream = proc.stdout;
    if (stdoutStream) {
      stdoutStream.setEncoding("utf-8");
      stdoutStream.on("data", (chunk: string) => { stdout += chunk; });
    }
    const t = setTimeout(() => {
      try { proc.kill(); } catch {  }
      resolve(null);
    }, 5_000);
    proc.on("error", () => {
      clearTimeout(t);
      resolve(null);
    });
    proc.on("close", (code: number | null) => {
      clearTimeout(t);
      if (code !== 0) {
        resolve(null);
        return;
      }
      try {
        const parsed = JSON.parse(stdout) as Record<string, unknown>;
        parsed["_source"] = "direct-store";
        resolve(parsed);
      } catch {
        resolve(null);
      }
    });
  });
}

export async function runDirectWrite(
  args: Record<string, unknown>,
  spawnFn: typeof spawn = spawn,
): Promise<Record<string, unknown> | null> {
  const cli = process.env.IAI_MCP_CLI ?? "iai";
  const literal = String(args.literal ?? args.text ?? "");
  const spawnArgs: string[] = ["capture", "--json", literal];
  const sessionId = args.session_id;
  if (sessionId && typeof sessionId === "string") {
    spawnArgs.push("--session-id", sessionId);
  }
  return new Promise((resolve) => {
    const opts: SpawnOptions = { stdio: ["ignore", "pipe", "pipe"] };
    const proc = spawnFn(cli, spawnArgs, opts);
    let stdout = "";
    const stdoutStream = proc.stdout;
    if (stdoutStream) {
      stdoutStream.setEncoding("utf-8");
      stdoutStream.on("data", (chunk: string) => { stdout += chunk; });
    }
    const t = setTimeout(() => {
      try { proc.kill(); } catch {  }
      resolve(null);
    }, 5_000);
    proc.on("error", () => {
      clearTimeout(t);
      resolve(null);
    });
    proc.on("close", (code: number | null) => {
      clearTimeout(t);
      if (code !== 0) {
        resolve(null);
        return;
      }
      try {
        const parsed = JSON.parse(stdout) as Record<string, unknown>;
        parsed["_source"] = "direct-store";
        resolve(parsed);
      } catch {
        resolve({ _source: "direct-store", status: "inserted" });
      }
    });
  });
}

export async function runDirectRecall(
  args: Record<string, unknown>,
  spawnFn: typeof spawn = spawn,
): Promise<Record<string, unknown> | null> {
  const cli = process.env.IAI_MCP_CLI ?? "iai";
  const cue = String(args.cue ?? "");
  if (!cue) return null;
  const limit = String(
    typeof args.limit === "number"
      ? args.limit
      : typeof args.budget_tokens === "number"
        ? Math.max(1, Math.ceil(args.budget_tokens / 300))
        : 10,
  );
  const spawnArgs: string[] = ["recall", "--json", "--limit", limit, cue];
  return new Promise((resolve) => {
    const opts: SpawnOptions = { stdio: ["ignore", "pipe", "pipe"] };
    const proc = spawnFn(cli, spawnArgs, opts);
    let stdout = "";
    const stdoutStream = proc.stdout;
    if (stdoutStream) {
      stdoutStream.setEncoding("utf-8");
      stdoutStream.on("data", (chunk: string) => { stdout += chunk; });
    }
    const t = setTimeout(() => {
      try { proc.kill(); } catch {  }
      resolve(null);
    }, 5_000);
    proc.on("error", () => {
      clearTimeout(t);
      resolve(null);
    });
    proc.on("close", (code: number | null) => {
      clearTimeout(t);
      if (code !== 0) {
        resolve(null);
        return;
      }
      try {
        const parsed = JSON.parse(stdout) as Record<string, unknown>;
        parsed["_source"] = "direct-store";
        resolve(parsed);
      } catch {
        resolve(null);
      }
    });
  });
}

export async function handleToolCall(
  bridge: PythonCoreBridge,
  name: ToolName,
  args: Record<string, unknown>,
  spawnFn: typeof spawn = spawn,
): Promise<unknown> {
  try {
    await bridge.start();
  } catch (startErr) {
    if (name === "episodes_recent") {
      const direct = await runDirectRecency(args, spawnFn);
      if (direct !== null) {
        return direct;
      }
      throw startErr;
    }
    if (name === "memory_capture") {
      const direct = await runDirectWrite(args, spawnFn);
      if (direct !== null) {
        return direct;
      }
      throw startErr;
    }
    if (name === "memory_recall") {
      const direct = await runDirectRecall(args, spawnFn);
      if (direct !== null) {
        return direct;
      }
      if (process.env.IAI_MCP_BANK_FALLBACK !== "0") {
        const fallback = await runBankFallback(
          String(args.cue ?? ""),
          BANK_FALLBACK_LIMIT,
          spawnFn,
        );
        if (fallback !== null) {
          return fallback;
        }
      }
    }
    throw startErr;
  }
  return invokeTool(bridge, name, args, spawnFn);
}

export async function runBankFallback(
  query: string,
  limit: number,
  spawnFn: typeof spawn = spawn,
): Promise<Record<string, unknown> | null> {
  const cli = process.env.IAI_MCP_CLI ?? "iai-mcp";
  const args = [
    "bank-recall",
    "--query", query,
    "--limit", String(limit),
    "--json",
  ];
  return new Promise((resolve) => {
    const opts: SpawnOptions = { stdio: ["ignore", "pipe", "pipe"] };
    const proc = spawnFn(cli, args, opts);
    let stdout = "";
    const stdoutStream = proc.stdout;
    if (stdoutStream) {
      stdoutStream.setEncoding("utf-8");
      stdoutStream.on("data", (chunk: string) => { stdout += chunk; });
    }
    const t = setTimeout(() => {
      try { proc.kill(); } catch {  }
      resolve(null);
    }, 5_000);
    proc.on("error", () => {
      clearTimeout(t);
      resolve(null);
    });
    proc.on("close", (code: number | null) => {
      clearTimeout(t);
      if (code !== 0) {
        resolve(null);
        return;
      }
      try {
        const parsed = JSON.parse(stdout) as Record<string, unknown>;
        parsed["_source"] = "bank-fallback";
        resolve(parsed);
      } catch {
        resolve(null);
      }
    });
  });
}

export async function invokeTool(
  bridge: PythonCoreBridge,
  name: ToolName,
  args: Record<string, unknown>,
  spawnFn: typeof spawn = spawn,
): Promise<unknown> {
  switch (name) {
    case "memory_recall": {
      try {
        return await bridge.call("memory_recall", args);
      } catch (err) {
        const direct = await runDirectRecall(args, spawnFn);
        if (direct !== null) {
          return direct;
        }
        if (process.env.IAI_MCP_BANK_FALLBACK !== "0") {
          const fallback = await runBankFallback(
            String(args.cue ?? ""),
            BANK_FALLBACK_LIMIT,
            spawnFn,
          );
          if (fallback !== null) {
            return fallback;
          }
        }
        throw err;
      }
    }
    case "memory_reinforce":
      return bridge.call("memory_reinforce", args);
    case "memory_contradict":
      return bridge.call("memory_contradict", args);
    case "memory_capture": {
      try {
        return await bridge.call("memory_capture", args);
      } catch (err) {
        if (!isDaemonDownError(err)) {
          throw err;
        }
        const direct = await runDirectWrite(args, spawnFn);
        if (direct !== null) {
          return direct;
        }
        throw err;
      }
    }
    case "memory_consolidate":
      return bridge.call("memory_consolidate", args);
    case "profile_get_set": {
      const op = args.operation as string;
      if (op === "get") {
        return bridge.call("profile_get", { knob: args.knob ?? null });
      }
      if (op === "set") {
        return bridge.call("profile_set", {
          knob: args.knob,
          value: args.value,
        });
      }
      throw new Error(`unknown operation ${op}`);
    }
    case "curiosity_pending":
      return bridge.call("curiosity_pending", args);
    case "schema_list":
      return bridge.call("schema_list", args);
    case "events_query":
      return bridge.call("events_query", args);
    case "memory_recall_structural":
      return bridge.call("memory_recall_structural", args);
    case "topology":
      return bridge.call("topology", args);
    case "camouflaging_status":
      return bridge.call("camouflaging_status", args);
    case "episodes_recent": {
      try {
        return await bridge.call("episodes_recent", args);
      } catch (err) {
        if (!isDaemonDownError(err)) {
          throw err;
        }
        const direct = await runDirectRecency(args, spawnFn);
        if (direct !== null) {
          return direct;
        }
        throw err;
      }
    }
    default: {
      const _exhaustive: never = name;
      throw new Error(
        `Tool not implemented: ${_exhaustive as string}. ` +
        `Available tools: ${TOOL_NAMES.join(", ")}`,
      );
    }
  }
}
