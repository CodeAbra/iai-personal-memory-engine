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
import {
  shouldWakeDefaultDaemonForToolCall,
  WrapperLifecycle,
} from "./lifecycle.js";
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
export type { ContentBlock, SessionPayloadRaw };

interface ToolCallLifecycle {
  ensureDaemonAlive(opts?: { waitForReachableMs?: number }): Promise<void>;
}

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
  lifecycle?: ToolCallLifecycle,
  shouldWakeForToolCall: () => boolean = shouldWakeDefaultDaemonForToolCall,
): { server: Server; bridge: PythonCoreBridge } {
  const b = bridge ?? new PythonCoreBridge();

  const server = new Server(
    {
      name: "iai-mcp",
      version: "0.1.0",
    },
    {
      capabilities: { tools: {} },
      instructions: JSON.stringify({
        context_editing: CONTEXT_EDITING_CONFIG,
        hot_tools: HOT_TOOLS,
      }),
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
      if (lifecycle && shouldWakeForToolCall()) {
        await lifecycle.ensureDaemonAlive({ waitForReachableMs: 7_000 });
      }
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
  const lifecycle = new WrapperLifecycle();
  const { server, bridge: b } = buildServer(undefined, spawn, lifecycle);

  const transport = new StdioServerTransport();
  await server.connect(transport);

  if (shouldWakeDefaultDaemonForToolCall()) {
    void lifecycle.ensureDaemonAlive().catch(() => null);
  }

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
