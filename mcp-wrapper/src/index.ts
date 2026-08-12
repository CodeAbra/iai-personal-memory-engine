#!/usr/bin/env node

import { pathToFileURL } from "node:url";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import {
  emitSessionOpen,
  newSessionId,
  PythonCoreBridge,
} from "./bridge.js";
import {
  applyCacheBreakpoint,
  buildCachedSystemPrompt,
  type ContentBlock,
  type SessionPayloadRaw,
} from "./caching.js";
import { WrapperLifecycle } from "./lifecycle.js";
import {
  CONTEXT_EDITING_CONFIG,
  HOT_TOOLS,
  listHotTools,
} from "./registry.js";
import {
  emitSickWarningIfNeeded,
  probeDaemonDoctor,
} from "./sickWarning.js";
import { spawn } from "node:child_process";
import { handleToolCall, type ToolName } from "./tools.js";

export {
  applyCacheBreakpoint,
  buildCachedSystemPrompt,
  CONTEXT_EDITING_CONFIG,
  HOT_TOOLS,
};

export const SERVER_INSTRUCTIONS =
  "Call memory_recall BEFORE searching the repository when the question " +
  "is about a decision, a preference, a past discussion, or why " +
  "something is the way it is — that answer lives in the conversation " +
  "record, not in the tree, and a single budget-bounded recall displaces " +
  "the multi-step search it would otherwise take. Keep using file search " +
  "for the current state of the code: recall orders before search, it " +
  "never replaces it. Records are timestamped and carry valid_from/valid_to; " +
  "treat anything they say about code state as historical unless " +
  "confirmed against the tree.\n\nconfig: " +
  JSON.stringify({
    context_editing: CONTEXT_EDITING_CONFIG,
    hot_tools: HOT_TOOLS,
  });
export type { ContentBlock, SessionPayloadRaw };


function toolResult(payload: unknown) {
  const content = [
    { type: "text" as const, text: JSON.stringify(payload) },
  ];
  if (typeof payload === "object" && payload !== null) {
    return {
      content,
      structuredContent: payload as Record<string, unknown>,
    };
  }
  return { content };
}

export function buildServer(
  bridge?: PythonCoreBridge,
  spawnFn: typeof spawn = spawn,
): { server: Server; bridge: PythonCoreBridge } {
  const b = bridge ?? new PythonCoreBridge();

  const server = new Server(
    {
      name: "iai-mcp",
      version: "3.0.1",
    },
    {
      capabilities: { tools: {} },
      instructions: SERVER_INSTRUCTIONS,
    },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const t0 = Date.now();
    const tools = listHotTools();
    const listHandlerElapsedMs = Date.now() - t0;
    return {
      tools,
      _meta: { listHandlerElapsedMs },
    };
  });

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const name = req.params.name as ToolName;
    if (!HOT_TOOLS.includes(name)) {
      return {
        content: [{ type: "text" as const, text: `unknown tool ${name}` }],
        isError: true,
      };
    }
    try {
      const result = await handleToolCall(b, name, req.params.arguments ?? {}, spawnFn);
      return toolResult(result);
    } catch (e) {
      return {
        content: [
          { type: "text" as const, text: `error: ${(e as Error).message}` },
        ],
        isError: true,
      };
    }
  });

  const bootSessionId = newSessionId();
  server.oninitialized = () => {
    b.start()
      .then(() =>
        b.call<SessionPayloadRaw>("session_start_payload", {
          session_id: bootSessionId,
        }),
      )
      .catch(() => null);

    void probeDaemonDoctor()
      .then(emitSickWarningIfNeeded)
      .catch(() => null);
  };

  return { server, bridge: b };
}

async function main(): Promise<void> {
  const { server, bridge: b } = buildServer();

  const lifecycle = new WrapperLifecycle();

  const transport = new StdioServerTransport();
  await server.connect(transport);

  void lifecycle.ensureDaemonAlive().catch(() => null);

  lifecycle.scheduleAutoHeal();

  void lifecycle.registerHeartbeat().catch(() => null);

  void b
    .start()
    .then(() => emitSessionOpen(newSessionId()))
    .catch(() => {
    });

  const shutdown = async (): Promise<void> => {
    try {
      await lifecycle.cleanupHeartbeat();
    } catch {
    }
    b.disconnect();
    process.exit(0);
  };
  process.on("SIGTERM", () => { void shutdown(); });
  process.on("SIGINT", () => { void shutdown(); });
}

if (
  process.argv[1] != null &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  void main();
}
