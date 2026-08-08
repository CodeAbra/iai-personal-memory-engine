
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { EventEmitter } from "node:events";

import { invokeTool, handleToolCall, runDirectRecency, runDirectRecall, ERR_EMBEDDER_REFUSAL } from "../src/tools.js";
import type { PythonCoreBridge } from "../src/bridge.js";


type SpawnReturn = ReturnType<typeof import("node:child_process").spawn>;

function makeMockSpawnFn(payload: Record<string, unknown>) {
  const calls: Array<{ cmd: string; args: string[] }> = [];
  const spawnFn = (cmd: string, args: ReadonlyArray<string>): SpawnReturn => {
    calls.push({ cmd, args: [...args] });
    const proc = new EventEmitter() as EventEmitter & {
      stdout: EventEmitter & { setEncoding: (enc: string) => void };
      stderr: EventEmitter;
      kill: () => void;
    };
    const stdout = new EventEmitter() as EventEmitter & {
      setEncoding: (enc: string) => void;
    };
    stdout.setEncoding = () => {};
    proc.stdout = stdout;
    proc.stderr = new EventEmitter();
    proc.kill = () => {};
    setImmediate(() => {
      stdout.emit("data", JSON.stringify(payload));
      proc.emit("close", 0);
    });
    return proc as unknown as SpawnReturn;
  };
  return { spawnFn, calls };
}

function makeNeverCalledSpawnFn() {
  let wasCalled = false;
  const spawnFn = (cmd: string, args: ReadonlyArray<string>): SpawnReturn => {
    wasCalled = true;
    const proc = new EventEmitter() as EventEmitter & {
      stdout: EventEmitter & { setEncoding: (enc: string) => void };
      stderr: EventEmitter;
      kill: () => void;
    };
    const stdout = new EventEmitter() as EventEmitter & {
      setEncoding: (enc: string) => void;
    };
    stdout.setEncoding = () => {};
    proc.stdout = stdout;
    proc.stderr = new EventEmitter();
    proc.kill = () => {};
    return proc as unknown as SpawnReturn;
  };
  return { spawnFn, get wasCalled() { return wasCalled; } };
}

const DIRECT_RECENCY_PAYLOAD = {
  turns: [
    {
      record_id: "rec-1",
      literal_surface: "direct store turn",
      session_id: "test-session",
      captured_at: "2026-06-01T10:00:00+00:00",
    },
  ],
  count: 1,
};


describe("invokeTool episodes_recent daemon-down (call rejects)", () => {
  it("falls through to the direct-store CLI subcommand, NOT bank-recall", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECENCY_PAYLOAD);

    const mockBridge = {
      call: async () => { throw new Error("socket dead"); },
    } as unknown as PythonCoreBridge;

    const result = await invokeTool(mockBridge, "episodes_recent", { n: 5 }, spawnFn as any);

    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai (not iai-mcp bank-recall)");
    assert.ok(
      args.includes("last"),
      `args must include 'last'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      args.includes("--json"),
      `args must include '--json'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      !args.includes("bank-recall"),
      "must NOT spawn bank-recall for recency",
    );

    assert.ok(result !== null, "result must be non-null");
    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
    assert.ok(Array.isArray(r["turns"]), "turns must be an array");
  });
});


describe("handleToolCall episodes_recent — bridge.start rejects (C2-H1)", () => {
  it("routes to direct-store CLI when bridge.start() rejects with daemon-unreachable error", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECENCY_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ENOENT (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
    } as unknown as PythonCoreBridge;

    const result = await handleToolCall(mockBridge, "episodes_recent", { n: 5 }, spawnFn as any);

    assert.equal(calls.length, 1, "exactly one subprocess spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai");
    assert.ok(args.includes("last"), "args must include 'last'");
    assert.ok(args.includes("--json"), "args must include '--json'");
    assert.ok(!args.includes("bank-recall"), "must NOT spawn bank-recall");

    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});


describe("handleToolCall — warm invokeTool error propagates (C3-H3)", () => {
  it("rejects with the warm error and never spawns direct-store or bank-recall", async () => {
    const { spawnFn: neverSpawn, wasCalled } = makeNeverCalledSpawnFn();

    const warmError = new Error("RPC timeout: method episodes_recent timed out");

    const mockBridge = {
      start: async () => {  },
      call: async () => { throw warmError; },
    } as unknown as PythonCoreBridge;

    await assert.rejects(
      () => handleToolCall(mockBridge, "episodes_recent", { n: 5 }, neverSpawn as any),
      (err: Error) => {
        assert.equal(
          err.message,
          warmError.message,
          "must propagate the original warm error",
        );
        return true;
      },
    );
    assert.equal(wasCalled, false, "direct-store spawn must NOT be triggered by a warm error");
  });
});


describe("index.ts buildServer CallToolRequest — start rejects → direct-store (C3-H1/CL4-H2)", () => {
  it("routes episodes_recent to direct-store via the factory-registered handler (mandatory real test)", async () => {
    const { buildServer } = await import("../src/index.js");

    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECENCY_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ECONNREFUSED (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
      disconnect: () => {},
    } as unknown as PythonCoreBridge;

    const { server } = buildServer(mockBridge, spawnFn as any);

    const requestHandlers = (server as any)._requestHandlers as Map<
      string,
      (req: unknown, extra: unknown) => Promise<unknown>
    >;
    const handler = requestHandlers.get("tools/call");
    assert.ok(handler, "CallToolRequest handler must be registered");

    const req = {
      method: "tools/call",
      params: {
        name: "episodes_recent",
        arguments: { n: 5 },
      },
    };
    const result = await handler(req, {}) as Record<string, unknown>;

    assert.ok(!result["isError"], `must not be an error; got: ${JSON.stringify(result)}`);

    const sc = (result["structuredContent"] ?? {}) as Record<string, unknown>;
    assert.equal(
      sc["_source"] ?? (JSON.parse((result["content"] as any)?.[0]?.text ?? "{}")["_source"]),
      "direct-store",
      "_source must be 'direct-store' in the response payload",
    );

    assert.equal(calls.length, 1, "exactly one subprocess spawned");
    assert.equal(calls[0].cmd, "iai");
    assert.ok(calls[0].args.includes("last"));
    assert.ok(calls[0].args.includes("--json"));
    assert.ok(!calls[0].args.includes("bank-recall"), "must not spawn bank-recall");
  });
});


const DIRECT_WRITE_PAYLOAD = {
  id: "rec-write-1",
  status: "inserted",
};

describe("invokeTool memory_capture daemon-down (call rejects) → direct-store write", () => {
  it("falls through to the direct-write CLI subcommand, NOT bank-recall", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_WRITE_PAYLOAD);

    const mockBridge = {
      call: async () => { throw new Error("socket dead"); },
    } as unknown as PythonCoreBridge;

    const result = await invokeTool(mockBridge, "memory_capture", { literal: "test capture text" }, spawnFn as any);

    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai (not iai-mcp bank-recall)");
    assert.ok(
      args.includes("capture"),
      `args must include 'capture'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      args.includes("--json"),
      `args must include '--json'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      !args.includes("bank-recall"),
      "must NOT spawn bank-recall for a write",
    );

    assert.ok(result !== null, "result must be non-null");
    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});


describe("handleToolCall memory_capture — bridge.start rejects (C2-H1) → direct write", () => {
  it("routes to direct-write CLI when bridge.start() rejects with daemon-unreachable error", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_WRITE_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ENOENT (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
    } as unknown as PythonCoreBridge;

    const result = await handleToolCall(mockBridge, "memory_capture", { literal: "test text" }, spawnFn as any);

    assert.equal(calls.length, 1, "exactly one subprocess spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai");
    assert.ok(args.includes("capture"), "args must include 'capture'");
    assert.ok(args.includes("--json"), "args must include '--json'");
    assert.ok(!args.includes("bank-recall"), "must NOT spawn bank-recall");

    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});


describe("index.ts buildServer CallToolRequest memory_capture — start rejects → direct-store write (C3-H1)", () => {
  it("routes memory_capture to direct-write via the factory-registered handler (mandatory real test)", async () => {
    const { buildServer } = await import("../src/index.js");

    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_WRITE_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ECONNREFUSED (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
      disconnect: () => {},
    } as unknown as PythonCoreBridge;

    const { server } = buildServer(mockBridge, spawnFn as any);

    const requestHandlers = (server as any)._requestHandlers as Map<
      string,
      (req: unknown, extra: unknown) => Promise<unknown>
    >;
    const handler = requestHandlers.get("tools/call");
    assert.ok(handler, "CallToolRequest handler must be registered");

    const req = {
      method: "tools/call",
      params: {
        name: "memory_capture",
        arguments: { literal: "test capture text" },
      },
    };
    const result = await handler(req, {}) as Record<string, unknown>;

    assert.ok(!result["isError"], `must not be an error; got: ${JSON.stringify(result)}`);

    assert.equal(calls.length, 1, "exactly one subprocess spawned");
    assert.equal(calls[0].cmd, "iai");
    assert.ok(calls[0].args.includes("capture"), "must spawn capture subcommand");
    assert.ok(calls[0].args.includes("--json"));
    assert.ok(!calls[0].args.includes("bank-recall"), "must not spawn bank-recall");

    const sc = (result["structuredContent"] ?? {}) as Record<string, unknown>;
    const source = sc["_source"] ?? (() => {
      try {
        return JSON.parse((result["content"] as any)?.[0]?.text ?? "{}")["_source"];
      } catch { return undefined; }
    })();
    assert.equal(source, "direct-store", "_source must be 'direct-store' in the response payload");
  });
});


const DIRECT_RECALL_PAYLOAD = {
  hits: [
    {
      literal_surface: "degraded store recall hit text",
      score: 0.0,
      _degraded: true,
    },
  ],
  count: 1,
  _source: "direct-store",
};

describe("invokeTool memory_recall daemon-down (call rejects) → direct-store recall", () => {
  it("falls through to the direct-recall CLI subcommand FIRST (not bank)", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECALL_PAYLOAD);

    const mockBridge = {
      call: async () => { throw new Error("socket dead"); },
    } as unknown as PythonCoreBridge;

    const result = await invokeTool(mockBridge, "memory_recall", { cue: "test recall cue", budget_tokens: 3000 }, spawnFn as any);

    assert.ok(calls.length >= 1, "at least one subprocess must be spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai (not iai-mcp bank-recall)");
    assert.ok(
      args.includes("recall"),
      `first spawn args must include 'recall'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      args.includes("--json"),
      `first spawn args must include '--json'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      !args.includes("bank-recall"),
      "must NOT spawn bank-recall as the primary daemon-down path",
    );

    assert.ok(result !== null, "result must be non-null");
    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});


describe("handleToolCall memory_recall — bridge.start rejects (C2-H1) → direct-store recall", () => {
  it("routes to direct-recall CLI when bridge.start() rejects", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECALL_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ENOENT (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
    } as unknown as PythonCoreBridge;

    const result = await handleToolCall(mockBridge, "memory_recall", { cue: "test cue" }, spawnFn as any);

    assert.ok(calls.length >= 1, "at least one subprocess spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai");
    assert.ok(args.includes("recall"), "first spawn must include 'recall'");
    assert.ok(args.includes("--json"), "first spawn must include '--json'");
    assert.ok(!args.includes("bank-recall"), "must NOT spawn bank-recall as primary");

    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});


describe("index.ts buildServer CallToolRequest memory_recall — start rejects → direct-store (C3-H1/CL4-H2)", () => {
  it("routes memory_recall to direct-store via the factory-registered handler (mandatory real test)", async () => {
    const { buildServer } = await import("../src/index.js");

    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECALL_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ECONNREFUSED (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
      disconnect: () => {},
    } as unknown as PythonCoreBridge;

    const { server } = buildServer(mockBridge, spawnFn as any);

    const requestHandlers = (server as any)._requestHandlers as Map<
      string,
      (req: unknown, extra: unknown) => Promise<unknown>
    >;
    const handler = requestHandlers.get("tools/call");
    assert.ok(handler, "CallToolRequest handler must be registered");

    const req = {
      method: "tools/call",
      params: {
        name: "memory_recall",
        arguments: { cue: "test recall cue", budget_tokens: 3000 },
      },
    };
    const result = await handler(req, {}) as Record<string, unknown>;

    assert.ok(!result["isError"], `must not be an error; got: ${JSON.stringify(result)}`);

    assert.ok(calls.length >= 1, "at least one subprocess spawned");
    assert.equal(calls[0].cmd, "iai");
    assert.ok(calls[0].args.includes("recall"), "first spawn must include 'recall' subcommand");
    assert.ok(calls[0].args.includes("--json"));
    assert.ok(!calls[0].args.includes("bank-recall"), "must not spawn bank-recall as primary");

    const sc = (result["structuredContent"] ?? {}) as Record<string, unknown>;
    const source = sc["_source"] ?? (() => {
      try {
        return JSON.parse((result["content"] as any)?.[0]?.text ?? "{}")["_source"];
      } catch { return undefined; }
    })();
    assert.equal(source, "direct-store", "_source must be 'direct-store' in the response payload");
  });
});

describe("invokeTool memory_recall embedder refusal → surfaced, never relabeled", () => {
  it("re-throws a re-embedding-migration refusal without spawning any fallback", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECALL_PAYLOAD);

    const refusal = new Error(
      "store vector space refused",
    ) as Error & { code?: number };
    refusal.code = ERR_EMBEDDER_REFUSAL;
    const mockBridge = {
      call: async () => { throw refusal; },
    } as unknown as PythonCoreBridge;

    await assert.rejects(
      invokeTool(mockBridge, "memory_recall", { cue: "probe" }, spawnFn as any),
      /vector space refused/,
    );
    assert.equal(calls.length, 0, "no direct/bank fallback may run on a refusal");
  });
});

describe("invokeTool memory_temporal_recall embedder refusal → surfaced", () => {
  it("re-throws the wire code without spawning the temporal fallback", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECALL_PAYLOAD);
    const refusal = new Error("refused") as Error & { code?: number };
    refusal.code = ERR_EMBEDDER_REFUSAL;
    const mockBridge = {
      call: async () => { throw refusal; },
    } as unknown as PythonCoreBridge;

    await assert.rejects(
      invokeTool(mockBridge, "memory_temporal_recall", { cue: "probe" }, spawnFn as any),
      /refused/,
    );
    assert.equal(calls.length, 0, "no temporal fallback may run on a refusal");
  });
});

describe("runDirectRecall child refusal channel", () => {
  it("rejects with the wire code when the CLI child exits non-zero with an error_code doc", async () => {
    const calls: Array<{ cmd: string; args: string[] }> = [];
    const spawnFn = ((cmd: string, args: ReadonlyArray<string>) => {
      calls.push({ cmd, args: [...args] });
      const proc = new EventEmitter() as EventEmitter & {
        stdout: EventEmitter & { setEncoding: (enc: string) => void };
        stderr: EventEmitter;
        kill: () => void;
      };
      const stdout = new EventEmitter() as EventEmitter & {
        setEncoding: (enc: string) => void;
      };
      stdout.setEncoding = () => {};
      proc.stdout = stdout;
      proc.stderr = new EventEmitter();
      proc.kill = () => {};
      setImmediate(() => {
        stdout.emit("data", JSON.stringify({
          error: "embedder refused: vector space mismatch",
          error_code: ERR_EMBEDDER_REFUSAL,
          hits: [],
          count: 0,
        }));
        proc.emit("close", 2);
      });
      return proc;
    }) as any;

    await assert.rejects(
      runDirectRecall({ cue: "probe" }, spawnFn as any),
      (err: Error & { code?: number }) =>
        err.code === ERR_EMBEDDER_REFUSAL && /embedder refused/.test(err.message),
    );
  });

  it("resolves null on a non-zero exit WITHOUT an error_code doc (argparse exits)", async () => {
    const spawnFn = ((cmd: string, args: ReadonlyArray<string>) => {
      const proc = new EventEmitter() as EventEmitter & {
        stdout: EventEmitter & { setEncoding: (enc: string) => void };
        stderr: EventEmitter;
        kill: () => void;
      };
      const stdout = new EventEmitter() as EventEmitter & {
        setEncoding: (enc: string) => void;
      };
      stdout.setEncoding = () => {};
      proc.stdout = stdout;
      proc.stderr = new EventEmitter();
      proc.kill = () => {};
      setImmediate(() => { proc.emit("close", 2); });
      return proc;
    }) as any;

    const result = await runDirectRecall({ cue: "probe" }, spawnFn as any);
    assert.equal(result, null);
  });
});
