
import type { PythonCoreBridge } from "./bridge.js";

// Wire contract with the daemon socket (iai_mcp.errors.ERR_EMBEDDER_REFUSAL).
export const ERR_EMBEDDER_REFUSAL = -32011;

import { spawn, type SpawnOptions } from "node:child_process";

export const BANK_FALLBACK_LIMIT = 20;

export const TOOL_NAMES = [
  "memory_recall",
  "memory_search",
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
  "episodes_recent",
  "memory_temporal_recall",
  "claim_check",
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
      "Recall verbatim memories by cue — decisions, preferences, prior " +
      "discussion, rationale. Call before a repository search. Returns " +
      "hits + anti_hits.",
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
        ann_path_used: { type: "boolean" },
        exact_authority_used: { type: "boolean" },
        overnight_digest: { type: ["object", "null"] },
        pask_teachback: { type: ["object", "null"] },
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
        session_id: {
          type: "string",
          description:
            "Session identifier for correlating this reinforcement with the " +
            "session's retrieval history. Optional; omit for old clients.",
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
        epistemic_status: {
          type: "string",
          enum: ["fact", "estimate", "hypothesis", "opinion", "unknown"],
          default: "unknown",
          description:
            "Caller-declared epistemic status of the corrected fact. Omit " +
            "for 'unknown' (default, no behavior change). A value outside " +
            "the enum is coerced to 'unknown' server-side, never rejected.",
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
      "Capture a verbatim turn (auto-dedups near-duplicates). " +
      "Use for corrections, not for minting standing-order directives.",
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
        epistemic_status: {
          type: "string",
          enum: ["fact", "estimate", "hypothesis", "opinion", "unknown"],
          default: "unknown",
          description:
            "Caller-declared epistemic status. Omit for 'unknown' (default, no behavior change). " +
            "A value outside the enum is coerced to 'unknown' server-side, never rejected.",
        },
        salience_level: {
          type: "string",
          enum: ["unflagged", "notable", "critical"],
          default: "unflagged",
          description:
            "Caller-declared salience level for a decision, correction, or load-bearing " +
            "preference marked in-turn. Additive rank-fusion boost only -- never a merge/drop " +
            "lock. Omit for 'unflagged' (default, no behavior change). A value outside the " +
            "enum is coerced to 'unflagged' server-side, never rejected.",
        },
        next_action: {
          type: "string",
          description:
            "Optional immediate next step for the current live session task. " +
            "Folded verbatim onto this session's own working-tier entry after " +
            "the capture completes; surfaces at the next session start and on " +
            "every subsequent turn until updated again.",
        },
        focus: {
          type: "string",
          description:
            "Optional current point of attention for the live session task. " +
            "Folded verbatim onto this session's own working-tier entry after " +
            "the capture completes, alongside next_action.",
        },
        agent_id: {
          type: "string",
          description:
            "Optional id of a background agent this capture is spawning or " +
            "completing. Combine with agent_role and agent_expected_artifact " +
            "to register a pending agent; combine with agent_complete_id on a " +
            "later call to mark it done.",
        },
        agent_role: {
          type: "string",
          description:
            "Optional role of the spawned background agent (for example " +
            "'research' or 'implement'). Required alongside agent_id and " +
            "agent_expected_artifact to register a spawn; omitted otherwise.",
        },
        agent_expected_artifact: {
          type: "string",
          description:
            "Optional artifact the spawned background agent is expected to " +
            "produce. Required alongside agent_id and agent_role to register " +
            "a spawn; omitted otherwise.",
        },
        agent_complete_id: {
          type: "string",
          description:
            "Optional id of a previously spawned background agent to mark " +
            "complete on this call.",
        },
        agent_model: {
          type: "string",
          description:
            "Optional model label for the spawned background agent, recorded " +
            "on the registry entry when agent_id/agent_role/" +
            "agent_expected_artifact register a spawn.",
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
      "Read or write a profile knob (10 sealed: 9 AUTIST + wake_depth). operation get|set; returns knob value.",
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
        total: { type: "integer" },
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
  memory_search: {
    name: "memory_search",
    description:
      "Use for code/doc search; returns hints to verify — never replaces " +
      "a repository search.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search text: identifiers, phrases, or a question.",
        },
        k: {
          type: "integer",
          description: "Max hits (default 8, max 24).",
          default: 8,
        },
      },
      required: ["query"],
    },
    outputSchema: {
      type: "object",
      properties: {
        hits: { type: "array", items: { type: "object" } },
        frame: { type: "string" },
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
      "Snapshot of memory-graph topology: N, C, L, sigma, community_count, regime. Read-only diagnostic; sigma never toggles retrieval.",
    inputSchema: { type: "object", properties: {} },
    outputSchema: {
      type: "object",
      properties: {
        N: { type: "integer" },
        C: { type: ["number", "null"] },
        L: { type: ["number", "null"] },
        sigma: { type: ["number", "null"] },
        community_count: { type: "integer" },
        rich_club_ratio: { type: ["number", "null"] },
        regime: { type: "string" },
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
  memory_temporal_recall: {
    name: "memory_temporal_recall",
    description:
      "Time-travel recall: as_of bounds records, changed_since filters events. Read-only.",
    inputSchema: {
      type: "object",
      properties: {
        cue: {
          type: "string",
          description:
            "Optional natural-language cue. If omitted, the records side " +
            "returns recency-ordered rows bounded by as_of.",
        },
        as_of: {
          type: "string",
          description:
            "ISO-8601 timestamp. Bounds the records side: " +
            "records.created_at <= as_of.",
        },
        changed_since: {
          type: "string",
          description:
            "ISO-8601 timestamp. Bounds the events side: " +
            "events.ts > changed_since (strict).",
        },
        limit: {
          type: "integer",
          description: "Maximum items per side (default 10).",
          default: 10,
        },
      },
    },
    outputSchema: {
      type: "object",
      properties: {
        hits: { type: "array", items: { type: "object" } },
        changed_since_events: { type: "array", items: { type: "object" } },
        _scope: { type: "string" },
      },
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
  claim_check: {
    name: "claim_check",
    description:
      "Check a claim (e.g. 'X is not done') against memory. Returns " +
      "hits + anti_hits + a freshness verdict in one call.",
    inputSchema: {
      type: "object",
      properties: {
        cue: {
          type: "string",
          description: "The claim to check, e.g. 'the dashboard is not built yet'.",
        },
        session_id: {
          type: "string",
          description: "Current session id; gets written into provenance. Omit to use '-'.",
        },
        budget_tokens: {
          type: "integer",
          description: "Soft token budget for the underlying recall (default 1500).",
        },
      },
      required: ["cue"],
    },
    outputSchema: {
      type: "object",
      properties: {
        hits: { type: "array", items: { type: "object" } },
        anti_hits: { type: "array", items: { type: "object" } },
        verdict: { type: "object" },
        verdict_reason: { type: "string" },
        _source: { type: "string" },
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

// Forwards only literal/text + session_id; tier/role/epistemic_status/salience_level
// land as server defaults on this degraded path. The record is still persisted.
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
  return new Promise((resolve, reject) => {
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
        // The CLI child reports a refusal as a JSON error document on a
        // non-zero exit — silently discarding it would relabel a store
        // misconfiguration as an empty degraded rail. Keyed on the typed
        // error_code, never the exit status (argparse also exits 2).
        try {
          const doc = JSON.parse(stdout) as Record<string, unknown>;
          if (doc["error_code"] === ERR_EMBEDDER_REFUSAL) {
            const refusal = new Error(
              String(doc["error"] ?? "embedder refused"),
            ) as Error & { code?: number };
            refusal.code = ERR_EMBEDDER_REFUSAL;
            reject(refusal);
            return;
          }
        } catch {  }
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

export async function runDirectTemporal(
  args: Record<string, unknown>,
  spawnFn: typeof spawn = spawn,
): Promise<Record<string, unknown> | null> {
  const cli = process.env.IAI_MCP_CLI ?? "iai";
  const cue = String(args.cue ?? "");
  const limit = String(
    typeof args.limit === "number" ? args.limit : 10,
  );
  const spawnArgs: string[] = ["temporal-recall", "--json", "--limit", limit];
  if (typeof args.as_of === "string" && args.as_of) {
    spawnArgs.push("--as-of", args.as_of);
  }
  if (typeof args.changed_since === "string" && args.changed_since) {
    spawnArgs.push("--changed-since", args.changed_since);
  }
  if (cue) {
    spawnArgs.push(cue);
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
    if (name === "memory_temporal_recall") {
      const direct = await runDirectTemporal(args, spawnFn);
      if (direct !== null) {
        return direct;
      }
      throw startErr;
    }
    if (name === "memory_search") {
      const direct = await runDirectSearch(args, spawnFn);
      if (direct !== null) {
        return direct;
      }
      throw startErr;
    }
    throw startErr;
  }
  return invokeTool(bridge, name, args, spawnFn);
}

export async function runDirectSearch(
  args: Record<string, unknown>,
  spawnFn: typeof spawn = spawn,
): Promise<Record<string, unknown> | null> {
  const cli = process.env.IAI_MCP_CLI ?? "iai";
  const query = String(args.query ?? "");
  if (!query) return null;
  const limit = String(typeof args.k === "number" ? args.k : 8);
  const spawnArgs: string[] = ["search", "--json", "--limit", limit, query];
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
    }, 15_000);
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
        // An embedder refusal (typed wire code from the daemon socket) is
        // store misconfiguration, not availability — the degraded rails
        // cannot answer it either, so relabeling it as an unreachable
        // daemon hides the repair the operator must run.
        if ((err as { code?: number })?.code === ERR_EMBEDDER_REFUSAL) {
          throw err;
        }
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
    case "memory_temporal_recall": {
      try {
        return await bridge.call("memory_temporal_recall", args);
      } catch (err) {
        if ((err as { code?: number })?.code === ERR_EMBEDDER_REFUSAL) {
          throw err;
        }
        const direct = await runDirectTemporal(args, spawnFn);
        if (direct !== null) {
          return direct;
        }
        throw err;
      }
    }
    case "memory_search": {
      try {
        return await bridge.call("memory_search", args);
      } catch (err) {
        if (!isDaemonDownError(err)) {
          throw err;
        }
        const direct = await runDirectSearch(args, spawnFn);
        if (direct !== null) {
          return direct;
        }
        throw err;
      }
    }
    case "claim_check":
      return bridge.call("claim_check", args);
    default: {
      const _exhaustive: never = name;
      throw new Error(
        `Tool not implemented: ${_exhaustive as string}. ` +
        `Available tools: ${TOOL_NAMES.join(", ")}`,
      );
    }
  }
}
