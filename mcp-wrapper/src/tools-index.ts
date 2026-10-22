#!/usr/bin/env node

import { pathToFileURL } from "node:url";
import { spawn, type SpawnOptions } from "node:child_process";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const CODE_SEARCH_TIMEOUT_MS = 15_000;
const CODE_INDEX_TIMEOUT_MS = 30_000;
const SESSION_SEARCH_TIMEOUT_MS = 15_000;
const STATS_TIMEOUT_MS = 15_000;
const COMMAND_INSIGHT_TIMEOUT_MS = 15_000;
const EXECUTE_TIMEOUT_MS = 60_000;
const GIT_QUERY_TIMEOUT_MS = 15_000;
// Node's setTimeout takes a 32-bit signed ms count; above this it wraps and
// fires (almost) immediately instead of waiting the requested duration.
const MAX_EXECUTE_TIMEOUT_SEC = 2_147_473;
// The snippet byte budget bounds per-response CONTENT, not row count -- an
// unclamped k still lets a caller ask for an arbitrarily large hit list
// (one row of path/score/line-range per hit, no snippet, once the budget is
// spent). Cap row count independently of that budget.
const MAX_CODE_SEARCH_K = 200;
// Mirrors MAX_BATCH_COMMANDS in src/iai_mcp/tools/exec_cli.py -- keep both in sync.
const MAX_BATCH_COMMANDS = 64;

type ToolCallResult = {
  content: unknown[];
  structuredContent?: Record<string, unknown>;
  isError?: boolean;
};

const CODE_SEARCH_LANES = ["whole", "symbol", "xref", "vector", "fused", "literal", "importers", "goto-def", "references"] as const;
type CodeSearchLane = (typeof CODE_SEARCH_LANES)[number];

function isCodeSearchLane(value: string): value is CodeSearchLane {
  return (CODE_SEARCH_LANES as readonly string[]).includes(value);
}

function laneToSubcommand(lane: CodeSearchLane): string {
  return lane === "whole" ? "search" : `search-${lane}`;
}

const CODE_SEARCH_STORES = ["code", "output", "dependency"] as const;
type CodeSearchStore = (typeof CODE_SEARCH_STORES)[number];

function isCodeSearchStore(value: string): value is CodeSearchStore {
  return (CODE_SEARCH_STORES as readonly string[]).includes(value);
}

const GIT_QUERY_MODES = ["log", "blame", "diff"] as const;
type GitQueryMode = (typeof GIT_QUERY_MODES)[number];

function isGitQueryMode(value: string): value is GitQueryMode {
  return (GIT_QUERY_MODES as readonly string[]).includes(value);
}

const codeSearchSchema = {
  name: "code_search",
  description:
    "Search the code index (BM25 over indexed files). Returns ranked " +
    "hits with a bounded snippet per hit. Read-only.",
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Search text: identifiers, phrases, or a question.",
      },
      k: {
        type: "integer",
        description: "Max ranked hits (default 10, clamped server-side to 200).",
        default: 10,
        minimum: 1,
        maximum: MAX_CODE_SEARCH_K,
      },
      lane: {
        type: "string",
        enum: [...CODE_SEARCH_LANES],
        default: "whole",
        description:
          "Which search lane to query: whole-file BM25 (default), " +
          "symbol (exact identifier), xref (cross-file reference " +
          "injection), vector (semantic, embed-worker-backed), fused " +
          "(RRF over the base lanes, gated semantic addition), literal " +
          "(exact byte-substring scan, no case-folding/regex/tokenizer -- " +
          "the only lane that can prove a literal's absence, not just " +
          "report unknown), importers (reverse query: every file " +
          "importing/calling `query`, use with def_path to scope by " +
          "target), goto-def (Python AST-backed go-to-definition for " +
          "`query`, referenced from from_path; requires from_path), or " +
          "references (Python AST-backed find-references: every call " +
          "site of `query` defined in def_path, each tagged resolved or " +
          "unresolved; requires def_path).",
      },
      store: {
        type: "string",
        enum: [...CODE_SEARCH_STORES],
        default: "code",
        description:
          "Which store to query: 'code' (default, the source-code index), " +
          "'output' (previously captured command output, when a run was " +
          "indexed with auto_index), or 'dependency' (one indexed " +
          "third-party package; requires ecosystem and package, lane " +
          "must be 'whole').",
      },
      ecosystem: {
        type: "string",
        description: "store='dependency' only, required: dependency ecosystem, e.g. python.",
      },
      package: {
        type: "string",
        description: "store='dependency' only, required: dependency package name.",
      },
      def_path: {
        type: "string",
        description:
          "importers lane: scope resolved hits to importers/callers " +
          "whose target is this file; unresolved rows stay reported at " +
          "the symbol level. references lane, required: the file " +
          "`query` is defined in.",
      },
      from_path: {
        type: "string",
        description:
          "goto-def lane only, required: the file `query` is referenced " +
          "from.",
      },
    },
    required: ["query"],
  },
  outputSchema: {
    type: "object",
    properties: {
      query: { type: "string" },
      lane: { type: "string" },
      hits: {
        type: "array",
        items: {
          type: "object",
          properties: {
            path: { type: "string" },
            score: { type: "number" },
            line: {
              type: "integer",
              description:
                "literal lane: 1-based first-match line; goto-def lane: " +
                "the resolved definition's 1-based line.",
            },
            kind: {
              type: "string",
              description:
                "importers lane only: \"import\" or \"call\", the edge " +
                "kind this hit resolved through.",
            },
            status: {
              type: "string",
              description:
                "references lane only, per-row: \"resolved\" (this call " +
                "site's edge structurally confirms the queried " +
                "definition) or \"unresolved\" (the name matches but no " +
                "confirmed link exists, e.g. reachable only through a " +
                "wildcard import) -- distinct from the response-level " +
                "`status` field.",
            },
            line_start: { type: "integer" },
            line_end: { type: "integer" },
            snippet: { type: "string" },
            elided_lines: {
              type: "integer",
              description:
                "Query-matching lines in the file that fall outside the " +
                "returned window; always present, independent of budget state.",
            },
            budget_elided_lines: {
              type: "integer",
              description:
                "0 unless this hit's snippet was withheld once the total " +
                "response byte budget was exhausted, in which case it holds " +
                "the line count the withheld window would have shown. " +
                "`snippet === \"\" && budget_elided_lines > 0` means " +
                "withheld-for-budget, not that no other matches exist -- " +
                "check `elided_lines` for that.",
            },
          },
        },
      },
      vector_lane: {
        type: "string",
        description: "fused lane only: not_attempted | ok | degraded",
      },
      reason: {
        type: "string",
        description: "fused lane only, set when vector_lane is degraded",
      },
      scope: {
        type: "object",
        description:
          "whole/symbol/xref/fused lanes only (absent on vector): live " +
          "corpus scope this response was computed against, read fresh " +
          "from the on-disk manifest -- server-computed and " +
          "query-independent, never derived from the query string. " +
          "For whole/symbol lanes this is the exact manifest read the " +
          "hits themselves came from. xref/fused inherit the whole-file " +
          "lane's scope, read immediately after that lane's own search, " +
          "not guaranteed atomic with it under a concurrent index write.",
        properties: {
          n_files: {
            type: "integer",
            description:
              "count of files ever indexed and not since tombstoned by a " +
              "re-index -- not guaranteed to equal files currently on disk " +
              "(a file removed without a re-index still counts).",
          },
          generation: { type: "integer" },
        },
      },
      status: {
        type: "string",
        description:
          "present on every lane: for whole/symbol/xref/fused/vector, " +
          "\"unknown\" when hits is empty (absence not established) or " +
          "\"hits\" when non-empty -- for store='dependency' a query " +
          "against a package that was never indexed reports " +
          "\"not_indexed\" instead of \"unknown\"; the literal lane instead reports " +
          "\"absent\" (a PROVEN absence over files_scanned) or \"hits\"; " +
          "the importers lane reports \"no_importers\" (no resolved edge " +
          "over the indexed xref graph -- see `unresolved` for a " +
          "distinct edge-exists-but-unresolved case) or \"hits\"; the " +
          "goto-def lane reports \"resolved\" (hits has one entry), " +
          "\"unresolved\" (parsed but no definite target for query), " +
          "\"unsupported_language\" (from_path is not Python), or " +
          "\"parse_error\" (from_path is Python but recorded parse-failed); " +
          "the references lane reports \"hits\" (hits has one or more " +
          "entries, each row-tagged resolved/unresolved), \"no_references\" " +
          "(def_path is indexed and parsed but query has zero rows), " +
          "\"unsupported_language\" (def_path is not Python), or " +
          "\"parse_error\" (def_path is Python but recorded parse-failed).",
      },
      def_path: {
        type: "string",
        description: "importers/references lanes only, echoed back when supplied in the request.",
      },
      from_path: {
        type: "string",
        description: "goto-def lane only, echoed back from the request.",
      },
      unresolved: {
        type: "array",
        items: {
          type: "object",
          properties: { path: { type: "string" } },
        },
        description:
          "importers lane only, omitted when empty: files with an " +
          "unresolved reference to this symbol -- reported at the " +
          "whole-symbol level, never filtered by def_path.",
      },
      truncated: {
        type: "boolean",
        description:
          "importers/references lanes only, omitted when false: `hits` " +
          "(and, importers lane only, `unresolved`) exceeded k and were " +
          "independently bounded.",
      },
      files_scanned: {
        type: "integer",
        description:
          "literal lane only: files actually read this scan -- the " +
          "denominator the absence proof is keyed to, excluding any " +
          "unreadable or truncated file.",
      },
      files_unreadable: {
        type: "integer",
        description:
          "literal lane only, omitted when zero: manifest entries that " +
          "failed to open/read at scan time, excluded from files_scanned.",
      },
      files_truncated: {
        type: "integer",
        description:
          "literal lane only, omitted when zero: files whose bounded read " +
          "hit the cap with no in-prefix match, excluded from " +
          "files_scanned since a truncated miss is not a proven absence.",
      },
    },
  },
  annotations: {
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

const codeIndexSchema = {
  name: "code_index",
  description:
    "Walk a directory and index its files, or resolve+index one third-" +
    "party dependency package into its own store. Supply path OR " +
    "ecosystem+package+project_dir, never both. Mutates the on-disk index.",
  inputSchema: {
    type: "object",
    properties: {
      path: {
        type: "string",
        description: "Directory to walk and index.",
      },
      ecosystem: {
        type: "string",
        description:
          "Dependency mode only, required with package and project_dir: " +
          "dependency ecosystem, e.g. python.",
      },
      package: {
        type: "string",
        description: "Dependency mode only, required: dependency package name.",
      },
      project_dir: {
        type: "string",
        description:
          "Dependency mode only, required: project directory whose own " +
          "environment resolves the package.",
      },
    },
    required: [],
  },
  outputSchema: {
    type: "object",
    properties: {
      path: { type: "string" },
      ecosystem: { type: "string" },
      package: { type: "string" },
      project_dir: { type: "string" },
      indexed_files: { type: "integer" },
    },
  },
  annotations: {
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    openWorldHint: false,
  },
};

const executeSchema = {
  name: "execute",
  description:
    "Run a command via the executor daemon. Returns captured output " +
    "and a run handle. Use for shell commands. Only printed output " +
    "reaches this response; the rest stays retrievable in the run's " +
    "own artifact files on disk. Set auto_index to make it " +
    "searchable via code_search.",
  inputSchema: {
    type: "object",
    properties: {
      command: {
        type: "string",
        description: "Program to run.",
      },
      args: {
        type: "array",
        items: { type: "string" },
        description: "Arguments to the program, each a discrete argv element.",
        default: [],
      },
      timeout: {
        type: "number",
        description: "Seconds before the executor times out the run.",
        maximum: MAX_EXECUTE_TIMEOUT_SEC,
      },
      auto_index: {
        type: "boolean",
        default: false,
        description:
          "Opt-in: index this run's captured output for later retrieval " +
          "via code_search(store='output'). Never default-on. Output " +
          "shaped like a credential is refused, not indexed. Not applied " +
          "when the run also backgrounds -- index_error is set instead.",
      },
      mode: {
        type: "string",
        enum: [...GIT_QUERY_MODES],
        description:
          "Optional git-query mode: dispatches to a bounded git-log/" +
          "git-blame/git-diff subcommand sharing that subcommand's own " +
          "bound-by-construction validation, injection guard, and " +
          "secret-gate scrub, instead of running command/args/auto_index. " +
          "A scope parameter is required per mode: since/until/path or a " +
          "revision for log, file for blame, rev_a/rev_b/path for diff.",
      },
      since: { type: "string", description: "log mode: lower date bound (git's --since syntax)." },
      until: { type: "string", description: "log mode: upper date bound (git's --until syntax)." },
      author: { type: "string", description: "log mode: author filter (git's --author syntax)." },
      path: { type: "string", description: "log/diff modes: limit to this path." },
      // repo is intentionally unrestricted (git -C <repo>, any path) -- this
      // grants no more access than the pre-existing execute(command="git",
      // args=["-C", ...]) path; it is a convenience parameter, not a sandbox.
      repo: {
        type: "string",
        description:
          "any git-query mode: path to the git repository (default: the " +
          "executor daemon's working directory, which may not match your " +
          "shell -- pass repo for a reliable target).",
      },
      rev_range: { type: "string", description: "log mode: a revision or revision range." },
      rev_a: {
        type: "string",
        description: "diff mode: a revision range, or the first of two revisions.",
      },
      rev_b: { type: "string", description: "diff mode: the second of two revisions." },
      file: { type: "string", description: "blame mode: file to blame." },
      start_line: {
        type: "integer",
        description: "blame mode: 1-based inclusive start line of an optional line range.",
      },
      end_line: {
        type: "integer",
        description: "blame mode: 1-based inclusive end line of an optional line range.",
      },
    },
  },
  outputSchema: {
    type: "object",
    properties: {
      run_id: { type: "string" },
      state: { type: "string" },
      stdout: { type: "string" },
      stderr: { type: "string" },
      truncated: {
        type: "boolean",
        description: "git-query modes only: the tail-window bound was crossed.",
      },
      elided_bytes: {
        type: "integer",
        description: "git-query modes only: bytes elided by the tail-window bound.",
      },
      indexed: {
        type: "integer",
        description: "Present only when auto_index was set: files indexed.",
      },
      refused: {
        type: "array",
        items: { type: "string" },
        description:
          "Present only when auto_index was set: paths refused by the " +
          "secret-shaped-content gate and left out of the index.",
      },
    },
  },
  annotations: {
    readOnlyHint: false,
    destructiveHint: true,
    idempotentHint: false,
    openWorldHint: true,
  },
};

const sessionSearchSchema = {
  name: "session_search",
  description:
    "Query one session's captured events by session id. Read-only; " +
    "use for reviewing session history.",
  inputSchema: {
    type: "object",
    properties: {
      session_id: {
        type: "string",
        description: "Session id to query.",
      },
      session_ids: {
        type: "array",
        items: { type: "string" },
        description:
          "Optional explicit list of additional session ids to include, " +
          "returning the union of their events. session_id remains required " +
          "and is always the first id queried.",
      },
      substring: {
        type: "string",
        description: "Case-insensitive substring filter on event data.",
      },
      category: {
        type: "string",
        description: "Exact category filter.",
      },
      since: {
        type: "string",
        description: "RFC3339 lower bound (inclusive) on event timestamp.",
      },
      until: {
        type: "string",
        description: "RFC3339 upper bound (inclusive) on event timestamp.",
      },
    },
    required: ["session_id"],
  },
  outputSchema: {
    type: "object",
    properties: {
      session_id: {
        type: "string",
        description:
          "The single queried session id. Present only when no additional " +
          "session_ids were supplied; a multi-id query returns session_ids " +
          "instead, since events then span more than one session.",
      },
      session_ids: {
        type: "array",
        items: { type: "string" },
        description:
          "The full list of queried session ids, in query order. Present " +
          "only when session_ids was supplied in the request; replaces " +
          "session_id because events then span multiple sessions.",
      },
      events: {
        type: "array",
        items: { type: "object" },
      },
    },
  },
  annotations: {
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

const statsSchema = {
  name: "stats",
  description:
    "Aggregate code-index, session, and executor counts from the " +
    "on-disk stores. Read-only. Use for a subsystem health snapshot.",
  inputSchema: {
    type: "object",
    properties: {},
  },
  outputSchema: {
    type: "object",
    properties: {
      code_index: { type: "object" },
      sessions: { type: "object" },
      executor: {
        type: "object",
        properties: {
          run_count: { type: "integer" },
          redirect_count: {
            type: "integer",
            description: "Exact count of noisy commands the enforce hook rewrote to run through the sandbox executor.",
          },
          bytes_captured: {
            type: "integer",
            description: "Real stdout_offset+stderr_offset sum across every run's meta.json -- bytes kept in the sandbox rather than returned to the model.",
          },
        },
      },
    },
  },
  annotations: {
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

const commandInsightSchema = {
  name: "command_insight",
  description:
    "Actionable report over accumulated run history: names failing " +
    "commands and the most recent failure, not just a count. Read-only.",
  inputSchema: {
    type: "object",
    properties: {},
  },
  outputSchema: {
    type: "object",
    properties: {
      failing_runs: {
        type: "array",
        items: {
          type: "object",
          properties: {
            run_id: { type: "string" },
            cmd: { type: "string" },
            args: { type: "array", items: { type: "string" } },
            exit_code: { type: "integer" },
            started_at: { type: "integer" },
          },
        },
      },
      most_recent_failure: {
        type: ["object", "null"],
        description: "The newest entry of failing_runs by started_at, or null when there are no failures.",
      },
    },
  },
  annotations: {
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};

const batchExecuteSchema = {
  name: "batch_execute",
  description:
    "Run commands under the executor daemon's batch concurrency " +
    "semantics. Returns one result per command, in input order, " +
    "each bounded like execute's output. A failing command returns " +
    "as a result with a non-zero state, not a tool error.",
  inputSchema: {
    type: "object",
    properties: {
      commands: {
        type: "array",
        items: {
          type: "object",
          properties: {
            command: { type: "string", description: "Program to run." },
            args: {
              type: "array",
              items: { type: "string" },
              description: "Arguments to the program, each a discrete argv element.",
            },
          },
          required: ["command"],
        },
        maxItems: MAX_BATCH_COMMANDS,
        description: "Commands to run as a batch, in the order results are returned.",
      },
      concurrency: {
        type: "integer",
        default: 1,
        minimum: 1,
        maximum: 8,
        description:
          "1 = one shared timeout budget run in sequence; 2-8 = each command gets " +
          "its own timeout, run in parallel.",
      },
      timeout_ms: {
        type: "integer",
        description: "Total shared budget at concurrency=1, per-command timeout at concurrency 2-8.",
      },
    },
    required: ["commands"],
  },
  outputSchema: {
    type: "object",
    properties: {
      results: {
        type: "array",
        items: {
          type: "object",
          properties: {
            run_id: { type: "string" },
            state: { type: "string" },
            stdout: { type: "string" },
            stderr: { type: "string" },
          },
        },
        description: "One result per input command, in input order.",
      },
    },
  },
  annotations: {
    readOnlyHint: false,
    destructiveHint: true,
    idempotentHint: false,
    openWorldHint: true,
  },
};

// At concurrency 1 the batch shares one timeout budget so the worst case is
// timeout_ms; at 2-8 each command gets its own budget so the worst case
// scales with the batch length divided across the parallel slots.
function batchWorstCaseMs(count: number, concurrency: number, timeoutMs: number | undefined): number {
  if (timeoutMs === undefined) return EXECUTE_TIMEOUT_MS;
  return concurrency === 1 ? timeoutMs : Math.ceil(count / concurrency) * timeoutMs;
}

function parseIndexedCount(stdout: string): number | null {
  const m = stdout.trim().match(/^indexed (\d+) files under /);
  return m ? Number(m[1]) : null;
}

// A gated dependency index that was skipped (freshness-gate reuse, or a
// held concurrent lock) never prints "indexed N files under ..." -- its
// success line instead carries this machine-greppable marker (see
// dependency.py's describe_index_outcome). Checked before falling back to
// the missing-count error path so a real reuse/skip is never mistaken for
// an unparseable CLI failure.
function parseDependencySkippedReason(stdout: string): string | null {
  const m = stdout.trim().match(/reused \(skipped_reason=(\w+)/);
  return m ? m[1] : null;
}

function parseSessionEvents(stdout: string): Record<string, unknown>[] {
  const trimmed = stdout.trim();
  if (!trimmed || trimmed === "no events") return [];
  const events: Record<string, unknown>[] = [];
  for (const line of trimmed.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    events.push(JSON.parse(t));
  }
  return events;
}

function errorResult(message: string): ToolCallResult {
  return {
    content: [{ type: "text" as const, text: message }],
    isError: true,
  };
}

interface CliOutput {
  stdout: string;
  stderr: string;
}

type CliRunResult =
  | { ok: true; result: CliOutput }
  | { ok: false; error: ToolCallResult };

function runCli(
  argv: string[],
  timeoutMs: number,
  toolLabel: string,
  spawnFn: typeof spawn,
): Promise<CliRunResult> {
  const interpreter = process.env.IAI_MCP_PYTHON ?? "python3";
  const opts: SpawnOptions = { stdio: ["ignore", "pipe", "pipe"], windowsHide: true };
  return new Promise((resolve) => {
    let proc;
    try {
      proc = spawnFn(interpreter, argv, opts);
    } catch (e) {
      resolve({
        ok: false,
        error: errorResult(`${toolLabel}: failed to spawn ${interpreter}: ${(e as Error).message}`),
      });
      return;
    }
    let stdout = "";
    let stderr = "";
    const stdoutStream = proc.stdout;
    if (stdoutStream) {
      stdoutStream.setEncoding("utf-8");
      stdoutStream.on("data", (chunk: string) => { stdout += chunk; });
    }
    const stderrStream = proc.stderr;
    if (stderrStream) {
      stderrStream.setEncoding("utf-8");
      stderrStream.on("data", (chunk: string) => { stderr += chunk; });
    }
    const t = setTimeout(() => {
      try { proc.kill(); } catch { }
      resolve({ ok: false, error: errorResult(`${toolLabel}: timed out after ${timeoutMs}ms`) });
    }, timeoutMs);
    proc.on("error", (e: Error) => {
      clearTimeout(t);
      resolve({ ok: false, error: errorResult(`${toolLabel}: ${interpreter} not runnable: ${e.message}`) });
    });
    proc.on("close", (code: number | null) => {
      clearTimeout(t);
      if (code !== 0) {
        resolve({ ok: false, error: errorResult(`${toolLabel}: exit ${code}: ${stderr.trim()}`) });
        return;
      }
      resolve({ ok: true, result: { stdout, stderr } });
    });
  });
}

async function runCodeSearch(
  query: string,
  k: number,
  lane: CodeSearchLane,
  store: CodeSearchStore,
  defPath: string | undefined,
  fromPath: string | undefined,
  ecosystem: string | undefined,
  pkg: string | undefined,
  spawnFn: typeof spawn,
): Promise<ToolCallResult> {
  const argv =
    lane === "goto-def"
      ? ["-m", "iai_mcp.tools_cli", laneToSubcommand(lane), query, "--from-path", fromPath as string, "--json"]
      : ["-m", "iai_mcp.tools_cli", laneToSubcommand(lane), query, "--k", String(k), "--json"];
  if (store === "output") {
    argv.push("--store", "output");
  }
  if (store === "dependency") {
    argv.push("--store", "dependency", "--ecosystem", ecosystem as string, "--package", pkg as string);
  }
  if ((lane === "importers" || lane === "references") && defPath !== undefined) {
    argv.push("--def-path", defPath);
  }
  const r = await runCli(argv, CODE_SEARCH_TIMEOUT_MS, "code_search", spawnFn);
  if (!r.ok) return r.error;
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(r.result.stdout);
  } catch (e) {
    return errorResult(`code_search: failed to parse CLI JSON output: ${(e as Error).message}`);
  }
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
    structuredContent: payload,
  };
}

type CodeIndexTarget =
  | { path: string }
  | { ecosystem: string; pkg: string; projectDir: string };

async function runCodeIndex(target: CodeIndexTarget, spawnFn: typeof spawn): Promise<ToolCallResult> {
  const argv =
    "path" in target
      ? ["-m", "iai_mcp.tools_cli", "index", target.path]
      : [
          "-m", "iai_mcp.tools_cli", "index",
          "--ecosystem", target.ecosystem,
          "--package", target.pkg,
          "--project-dir", target.projectDir,
        ];
  const r = await runCli(argv, CODE_INDEX_TIMEOUT_MS, "code_index", spawnFn);
  if (!r.ok) return r.error;

  if ("path" in target) {
    const count = parseIndexedCount(r.result.stdout);
    if (count === null) {
      return errorResult(`code_index: could not parse indexed-file count from CLI output: ${r.result.stdout.slice(-200)}`);
    }
    const payload = { path: target.path, indexed_files: count };
    return {
      content: [{ type: "text" as const, text: JSON.stringify(payload) }],
      structuredContent: payload,
    };
  }

  const skippedReason = parseDependencySkippedReason(r.result.stdout);
  if (skippedReason !== null) {
    const payload = {
      ecosystem: target.ecosystem,
      package: target.pkg,
      project_dir: target.projectDir,
      reused: true,
      skipped_reason: skippedReason,
    };
    return {
      content: [{ type: "text" as const, text: JSON.stringify(payload) }],
      structuredContent: payload,
    };
  }

  const count = parseIndexedCount(r.result.stdout);
  if (count === null) {
    return errorResult(`code_index: could not parse indexed-file count from CLI output: ${r.result.stdout.slice(-200)}`);
  }
  const payload = {
    ecosystem: target.ecosystem,
    package: target.pkg,
    project_dir: target.projectDir,
    indexed_files: count,
  };
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
    structuredContent: payload,
  };
}

interface SessionSearchOpts {
  substring?: string;
  category?: string;
  since?: string;
  until?: string;
  alsoSessionIds?: string[];
}

async function runSessionSearch(
  sessionId: string,
  opts: SessionSearchOpts,
  spawnFn: typeof spawn,
): Promise<ToolCallResult> {
  const argv = ["-m", "iai_mcp.tools_cli", "session-search", sessionId];
  for (const extra of opts.alsoSessionIds ?? []) {
    argv.push("--also-session", extra);
  }
  if (opts.substring) argv.push("--substring", opts.substring);
  if (opts.category) argv.push("--category", opts.category);
  if (opts.since) argv.push("--since", opts.since);
  if (opts.until) argv.push("--until", opts.until);
  const r = await runCli(argv, SESSION_SEARCH_TIMEOUT_MS, "session_search", spawnFn);
  if (!r.ok) return r.error;
  let events: Record<string, unknown>[];
  try {
    events = parseSessionEvents(r.result.stdout);
  } catch (e) {
    return errorResult(`session_search: failed to parse event JSON: ${(e as Error).message}`);
  }
  const alsoSessionIds = opts.alsoSessionIds ?? [];
  const payload =
    alsoSessionIds.length > 0
      ? { session_ids: [sessionId, ...alsoSessionIds], events }
      : { session_id: sessionId, events };
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
    structuredContent: payload,
  };
}

async function runStats(spawnFn: typeof spawn): Promise<ToolCallResult> {
  const r = await runCli(["-m", "iai_mcp.tools_cli", "stats"], STATS_TIMEOUT_MS, "stats", spawnFn);
  if (!r.ok) return r.error;
  let stats: Record<string, unknown>;
  try {
    stats = JSON.parse(r.result.stdout.trim());
  } catch (e) {
    return errorResult(`stats: failed to parse CLI output: ${(e as Error).message}`);
  }
  return {
    content: [{ type: "text" as const, text: JSON.stringify(stats) }],
    structuredContent: stats,
  };
}

async function runCommandInsight(spawnFn: typeof spawn): Promise<ToolCallResult> {
  const r = await runCli(
    ["-m", "iai_mcp.tools_cli", "command-insight"],
    COMMAND_INSIGHT_TIMEOUT_MS,
    "command_insight",
    spawnFn,
  );
  if (!r.ok) return r.error;
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(r.result.stdout.trim());
  } catch (e) {
    return errorResult(`command_insight: failed to parse CLI output: ${(e as Error).message}`);
  }
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
    structuredContent: payload,
  };
}

async function runExecute(
  command: string,
  cmdArgs: string[],
  timeoutSec: number | undefined,
  autoIndex: boolean,
  spawnFn: typeof spawn,
): Promise<ToolCallResult> {
  const argv = ["-m", "iai_mcp.tools_cli", "run"];
  if (timeoutSec !== undefined) {
    argv.push("--timeout", String(timeoutSec));
  }
  if (autoIndex) {
    argv.push("--auto-index");
  }
  argv.push("--json", command, ...cmdArgs);
  const nodeTimeoutMs =
    timeoutSec !== undefined
      ? Math.max(EXECUTE_TIMEOUT_MS, timeoutSec * 1000 + 10_000)
      : EXECUTE_TIMEOUT_MS;
  const r = await runCli(argv, nodeTimeoutMs, "execute", spawnFn);
  if (!r.ok) return r.error;
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(r.result.stdout);
  } catch (e) {
    return errorResult(`execute: failed to parse CLI JSON output: ${(e as Error).message}`);
  }
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
    structuredContent: payload,
  };
}

interface GitQueryParams {
  since?: string;
  until?: string;
  author?: string;
  path?: string;
  // repo is forwarded to the CLI verbatim (git -C <repo>, any path) --
  // this grants no more access than the pre-existing execute(command="git",
  // args=["-C", ...]) path; it is a convenience parameter, not a sandbox.
  repo?: string;
  revRange?: string;
  revA?: string;
  revB?: string;
  file?: string;
  startLine?: number;
  endLine?: number;
}

async function runGitQuery(
  mode: GitQueryMode,
  params: GitQueryParams,
  spawnFn: typeof spawn,
): Promise<ToolCallResult> {
  const argv = ["-m", "iai_mcp.tools_cli"];
  if (mode === "log") {
    argv.push("git-log");
    if (params.since !== undefined) argv.push("--since", params.since);
    if (params.until !== undefined) argv.push("--until", params.until);
    if (params.author !== undefined) argv.push("--author", params.author);
    if (params.path !== undefined) argv.push("--path", params.path);
    if (params.repo !== undefined) argv.push("--repo", params.repo);
    argv.push("--json");
    if (params.revRange !== undefined) argv.push(params.revRange);
  } else if (mode === "blame") {
    argv.push("git-blame");
    if (params.startLine !== undefined) argv.push("--start-line", String(params.startLine));
    if (params.endLine !== undefined) argv.push("--end-line", String(params.endLine));
    if (params.repo !== undefined) argv.push("--repo", params.repo);
    argv.push("--json");
    if (params.file !== undefined) argv.push(params.file);
  } else {
    argv.push("git-diff");
    if (params.path !== undefined) argv.push("--path", params.path);
    if (params.repo !== undefined) argv.push("--repo", params.repo);
    argv.push("--json");
    if (params.revA !== undefined) argv.push(params.revA);
    if (params.revB !== undefined) argv.push(params.revB);
  }
  const r = await runCli(argv, GIT_QUERY_TIMEOUT_MS, "execute", spawnFn);
  if (!r.ok) return r.error;
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(r.result.stdout);
  } catch (e) {
    return errorResult(`execute: failed to parse CLI JSON output: ${(e as Error).message}`);
  }
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
    structuredContent: payload,
  };
}

async function runBatchExecute(
  commands: Array<{ command: string; args?: string[] }>,
  concurrency: number,
  timeoutMs: number | undefined,
  spawnFn: typeof spawn,
): Promise<ToolCallResult> {
  const argv = [
    "-m",
    "iai_mcp.tools_cli",
    "batch-run",
    "--concurrency",
    String(concurrency),
    "--commands-json",
    JSON.stringify(commands),
    "--json",
  ];
  if (timeoutMs !== undefined) {
    argv.push("--timeout-ms", String(timeoutMs));
  }
  const nodeTimeoutMs = Math.max(
    EXECUTE_TIMEOUT_MS,
    batchWorstCaseMs(commands.length, concurrency, timeoutMs) + 10_000,
  );
  const r = await runCli(argv, nodeTimeoutMs, "batch_execute", spawnFn);
  if (!r.ok) return r.error;
  let parsed: unknown;
  try {
    parsed = JSON.parse(r.result.stdout);
  } catch (e) {
    return errorResult(`batch_execute: failed to parse CLI JSON output: ${(e as Error).message}`);
  }
  if (!Array.isArray(parsed)) {
    return errorResult(`batch_execute: expected a JSON array from the CLI, got ${typeof parsed}`);
  }
  const payload = { results: parsed };
  return {
    content: [{ type: "text" as const, text: JSON.stringify(payload) }],
    structuredContent: payload,
  };
}

export function buildServer(spawnFn: typeof spawn = spawn): { server: Server } {
  const server = new Server(
    {
      name: "iai-tools",
      version: "1.0.0",
    },
    {
      capabilities: { tools: {} },
    },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: [
        codeSearchSchema,
        codeIndexSchema,
        executeSchema,
        sessionSearchSchema,
        statsSchema,
        commandInsightSchema,
        batchExecuteSchema,
      ],
    };
  });

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const name = req.params.name;
    const args = req.params.arguments ?? {};

    if (name === "code_search") {
      const query = typeof args.query === "string" ? args.query : "";
      if (!query) return errorResult("code_search: query is required");
      const kRaw = typeof args.k === "number" ? args.k : 10;
      if (kRaw < 1) {
        return errorResult(`code_search: k must be >= 1, got ${JSON.stringify(args.k)}`);
      }
      const k = Math.min(kRaw, MAX_CODE_SEARCH_K);
      const laneArg = typeof args.lane === "string" ? args.lane : "whole";
      if (!isCodeSearchLane(laneArg)) {
        return errorResult(
          `code_search: lane must be one of ${CODE_SEARCH_LANES.join(", ")}, got ${JSON.stringify(laneArg)}`,
        );
      }
      const storeArg = typeof args.store === "string" ? args.store : "code";
      if (!isCodeSearchStore(storeArg)) {
        return errorResult(
          `code_search: store must be one of ${CODE_SEARCH_STORES.join(", ")}, got ${JSON.stringify(storeArg)}`,
        );
      }
      if (storeArg === "output" && laneArg !== "whole") {
        return errorResult("code_search: store='output' is only supported with lane='whole'");
      }
      const ecosystemArg = typeof args.ecosystem === "string" ? args.ecosystem : undefined;
      const packageArg = typeof args.package === "string" ? args.package : undefined;
      if (storeArg === "dependency") {
        if (laneArg !== "whole") {
          return errorResult("code_search: store='dependency' is only supported with lane='whole'");
        }
        if (!ecosystemArg || !packageArg) {
          return errorResult("code_search: store='dependency' requires ecosystem and package");
        }
      }
      const defPathArg = typeof args.def_path === "string" ? args.def_path : undefined;
      if (defPathArg !== undefined && laneArg !== "importers" && laneArg !== "references") {
        return errorResult("code_search: def_path is only supported with lane='importers' or lane='references'");
      }
      if (laneArg === "references" && defPathArg === undefined) {
        return errorResult("code_search: def_path is required with lane='references'");
      }
      const fromPathArg = typeof args.from_path === "string" ? args.from_path : undefined;
      if (fromPathArg !== undefined && laneArg !== "goto-def") {
        return errorResult("code_search: from_path is only supported with lane='goto-def'");
      }
      if (laneArg === "goto-def" && fromPathArg === undefined) {
        return errorResult("code_search: from_path is required with lane='goto-def'");
      }
      return runCodeSearch(query, k, laneArg, storeArg, defPathArg, fromPathArg, ecosystemArg, packageArg, spawnFn);
    }

    if (name === "code_index") {
      const pathArg = typeof args.path === "string" && args.path !== "" ? args.path : undefined;
      const idxEcosystemArg = typeof args.ecosystem === "string" ? args.ecosystem : undefined;
      const idxPackageArg = typeof args.package === "string" ? args.package : undefined;
      const idxProjectDirArg = typeof args.project_dir === "string" ? args.project_dir : undefined;

      const depFields = [idxEcosystemArg, idxPackageArg, idxProjectDirArg];
      const hasDepAny = depFields.some((f) => f !== undefined);
      const hasDepAll = depFields.every((f) => f !== undefined);

      if (hasDepAny && !hasDepAll) {
        return errorResult("code_index: ecosystem/package/project_dir must all be supplied together");
      }
      if (pathArg !== undefined && hasDepAll) {
        return errorResult("code_index: supply either path or ecosystem/package/project_dir, not both");
      }
      if (pathArg === undefined && !hasDepAll) {
        return errorResult("code_index: supply either path or ecosystem/package/project_dir");
      }

      return pathArg !== undefined
        ? runCodeIndex({ path: pathArg }, spawnFn)
        : runCodeIndex(
            {
              ecosystem: idxEcosystemArg as string,
              pkg: idxPackageArg as string,
              projectDir: idxProjectDirArg as string,
            },
            spawnFn,
          );
    }

    if (name === "execute") {
      const modeArg = typeof args.mode === "string" ? args.mode : undefined;
      if (modeArg !== undefined) {
        if (!isGitQueryMode(modeArg)) {
          return errorResult(
            `execute: mode must be one of ${GIT_QUERY_MODES.join(", ")}, got ${JSON.stringify(modeArg)}`,
          );
        }
        const params: GitQueryParams = {
          since: typeof args.since === "string" ? args.since : undefined,
          until: typeof args.until === "string" ? args.until : undefined,
          author: typeof args.author === "string" ? args.author : undefined,
          path: typeof args.path === "string" ? args.path : undefined,
          repo: typeof args.repo === "string" ? args.repo : undefined,
          revRange: typeof args.rev_range === "string" ? args.rev_range : undefined,
          revA: typeof args.rev_a === "string" ? args.rev_a : undefined,
          revB: typeof args.rev_b === "string" ? args.rev_b : undefined,
          file: typeof args.file === "string" ? args.file : undefined,
          startLine: typeof args.start_line === "number" ? args.start_line : undefined,
          endLine: typeof args.end_line === "number" ? args.end_line : undefined,
        };
        return runGitQuery(modeArg, params, spawnFn);
      }

      const command = typeof args.command === "string" ? args.command : "";
      if (!command) return errorResult("execute: command is required unless mode is set");
      const cmdArgs = Array.isArray(args.args)
        ? args.args.filter((a): a is string => typeof a === "string")
        : [];
      const timeout = typeof args.timeout === "number" ? args.timeout : undefined;
      if (timeout !== undefined && (!Number.isFinite(timeout) || timeout > MAX_EXECUTE_TIMEOUT_SEC)) {
        return errorResult(
          `execute: timeout must be a finite number of seconds <= ${MAX_EXECUTE_TIMEOUT_SEC}`,
        );
      }
      const autoIndex = args.auto_index === true;
      return runExecute(command, cmdArgs, timeout, autoIndex, spawnFn);
    }

    if (name === "session_search") {
      const sessionId = typeof args.session_id === "string" ? args.session_id : "";
      if (!sessionId) return errorResult("session_search: session_id is required");
      const alsoSessionIds = Array.isArray(args.session_ids)
        ? args.session_ids.filter((id): id is string => typeof id === "string" && id.length > 0)
        : [];
      const opts: SessionSearchOpts = {
        substring: typeof args.substring === "string" ? args.substring : undefined,
        category: typeof args.category === "string" ? args.category : undefined,
        since: typeof args.since === "string" ? args.since : undefined,
        until: typeof args.until === "string" ? args.until : undefined,
        alsoSessionIds,
      };
      return runSessionSearch(sessionId, opts, spawnFn);
    }

    if (name === "stats") {
      return runStats(spawnFn);
    }

    if (name === "command_insight") {
      return runCommandInsight(spawnFn);
    }

    if (name === "batch_execute") {
      const rawCommands = args.commands;
      if (!Array.isArray(rawCommands) || rawCommands.length === 0) {
        return errorResult("batch_execute: commands must be a non-empty array");
      }
      if (rawCommands.length > MAX_BATCH_COMMANDS) {
        return errorResult(
          `batch_execute: commands must have at most ${MAX_BATCH_COMMANDS} elements, got ${rawCommands.length}`,
        );
      }
      const commands: Array<{ command: string; args?: string[] }> = [];
      for (const entry of rawCommands) {
        if (
          typeof entry !== "object" ||
          entry === null ||
          typeof (entry as Record<string, unknown>).command !== "string"
        ) {
          return errorResult("batch_execute: each command must be an object with a string 'command'");
        }
        const entryArgs = (entry as Record<string, unknown>).args;
        commands.push({
          command: (entry as Record<string, unknown>).command as string,
          args: Array.isArray(entryArgs)
            ? entryArgs.filter((a): a is string => typeof a === "string")
            : undefined,
        });
      }
      const concurrency = typeof args.concurrency === "number" ? args.concurrency : 1;
      if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 8) {
        return errorResult("batch_execute: concurrency must be an integer between 1 and 8");
      }
      if (args.timeout_ms !== undefined && typeof args.timeout_ms !== "number") {
        return errorResult("batch_execute: timeout_ms must be a positive integer number of milliseconds");
      }
      const timeoutMs = typeof args.timeout_ms === "number" ? args.timeout_ms : undefined;
      if (timeoutMs !== undefined && (!Number.isInteger(timeoutMs) || timeoutMs <= 0)) {
        return errorResult("batch_execute: timeout_ms must be a positive integer number of milliseconds");
      }
      if (
        timeoutMs !== undefined &&
        batchWorstCaseMs(commands.length, concurrency, timeoutMs) > MAX_EXECUTE_TIMEOUT_SEC * 1000
      ) {
        return errorResult(
          `batch_execute: worst-case wall time exceeds ${MAX_EXECUTE_TIMEOUT_SEC * 1000}ms`,
        );
      }
      return runBatchExecute(commands, concurrency, timeoutMs, spawnFn);
    }

    return errorResult(`unknown tool ${name}`);
  });

  return { server };
}

async function main(): Promise<void> {
  const { server } = buildServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);

  const shutdown = (): void => {
    process.exit(0);
  };
  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}

if (
  process.argv[1] != null &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  void main();
}
