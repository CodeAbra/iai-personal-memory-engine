import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { EventEmitter } from "node:events";
import type { spawn as SpawnFn, ChildProcess } from "node:child_process";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { buildServer } from "../src/tools-index.js";

const MAX_EXECUTE_TIMEOUT_SEC = 2_147_473;
const MAX_BATCH_COMMANDS = 64;

function fakeSpawn(stdoutPayload: string, exitCode = 0): typeof SpawnFn {
  const fn = (_cmd: string, _argv: readonly string[]) => {
    const stdout = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const stderr = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const proc = Object.assign(new EventEmitter(), { stdout, stderr, kill: () => {} });
    queueMicrotask(() => {
      if (stdoutPayload) stdout.emit("data", stdoutPayload);
      proc.emit("close", exitCode);
    });
    return proc as unknown as ChildProcess;
  };
  return fn as unknown as typeof SpawnFn;
}

function fakeSpawnCapturing(capturedArgv: string[][]): typeof SpawnFn {
  const fn = (_cmd: string, argv: readonly string[]) => {
    capturedArgv.push([...argv]);
    const stdout = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const stderr = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const proc = Object.assign(new EventEmitter(), { stdout, stderr, kill: () => {} });
    queueMicrotask(() => proc.emit("close", 0));
    return proc as unknown as ChildProcess;
  };
  return fn as unknown as typeof SpawnFn;
}

function fakeSpawnCapturingWithPayload(capturedArgv: string[][], stdoutPayload: string): typeof SpawnFn {
  const fn = (_cmd: string, argv: readonly string[]) => {
    capturedArgv.push([...argv]);
    const stdout = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const stderr = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const proc = Object.assign(new EventEmitter(), { stdout, stderr, kill: () => {} });
    queueMicrotask(() => {
      stdout.emit("data", stdoutPayload);
      proc.emit("close", 0);
    });
    return proc as unknown as ChildProcess;
  };
  return fn as unknown as typeof SpawnFn;
}

async function callTool(
  spawnFn: typeof SpawnFn,
  name: string,
  args: Record<string, unknown>,
): Promise<any> {
  const { server } = buildServer(spawnFn);
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test-client", version: "0.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  return client.callTool({ name, arguments: args });
}

const ONE_COMMAND = [{ command: "echo", args: ["hi"] }];

describe("batch_execute JSON-parse path", () => {
  it("wraps a well-formed CLI JSON array as structuredContent.results", async () => {
    const results = [{ run_id: "a", state: "exited:0", stdout: "hi\n", stderr: "" }];
    const result = await callTool(fakeSpawn(JSON.stringify(results)), "batch_execute", {
      commands: ONE_COMMAND,
    });
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, { results });
  });

  it("rejects a non-array CLI payload naming that an array was expected", async () => {
    const result = await callTool(fakeSpawn(JSON.stringify({ not: "an array" })), "batch_execute", {
      commands: ONE_COMMAND,
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /expected a JSON array/);
  });

  it("returns a parse-error result instead of throwing on non-JSON stdout", async () => {
    const result = await callTool(fakeSpawn("not json"), "batch_execute", { commands: ONE_COMMAND });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /batch_execute: failed to parse CLI JSON output/);
  });
});

describe("batch_execute input validation", () => {
  it("rejects empty commands before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "batch_execute", { commands: [] });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0, "an empty commands array must never reach a spawned CLI call");
  });

  it("rejects a batch over the size cap before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const oversized = Array.from({ length: MAX_BATCH_COMMANDS + 1 }, (_, i) => ({
      command: "echo",
      args: [String(i)],
    }));
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "batch_execute", {
      commands: oversized,
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /at most 64 elements/);
    assert.equal(capturedArgv.length, 0, "an oversized batch must never reach a spawned CLI call");
  });

  it("declares a maxItems bound on the commands schema matching the runtime cap", async () => {
    const { server } = buildServer(fakeSpawn(""));
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "0.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    const { tools } = await client.listTools();
    const batchExecute = tools.find((t) => t.name === "batch_execute");
    const properties = (batchExecute!.inputSchema as { properties?: Record<string, any> }).properties ?? {};
    assert.equal(properties.commands?.maxItems, MAX_BATCH_COMMANDS);
  });

  it("rejects concurrency outside 1-8 before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "batch_execute", {
      commands: ONE_COMMAND,
      concurrency: 9,
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /concurrency must be an integer between 1 and 8/);
    assert.equal(capturedArgv.length, 0);
  });

  it("rejects a negative timeout_ms before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "batch_execute", {
      commands: ONE_COMMAND,
      timeout_ms: -5000,
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /positive integer/);
    assert.equal(capturedArgv.length, 0, "a negative timeout_ms must never reach a spawned CLI call");
  });

  it("rejects a zero timeout_ms before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "batch_execute", {
      commands: ONE_COMMAND,
      timeout_ms: 0,
    });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });

  it("rejects a non-integer timeout_ms before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "batch_execute", {
      commands: ONE_COMMAND,
      timeout_ms: 1500.5,
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /positive integer/);
    assert.equal(capturedArgv.length, 0, "a non-integer timeout_ms must never reach a spawned CLI call");
  });

  it("rejects a non-numeric timeout_ms before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "batch_execute", {
      commands: ONE_COMMAND,
      timeout_ms: "5000",
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /positive integer/);
    assert.equal(capturedArgv.length, 0, "a non-numeric timeout_ms must never reach a spawned CLI call");
  });

  it("rejects a worst-case timeout above the 32-bit setTimeout cap before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const timeoutMs = MAX_EXECUTE_TIMEOUT_SEC * 1000;
    const fourCommands = [
      { command: "echo", args: ["a"] },
      { command: "echo", args: ["b"] },
      { command: "echo", args: ["c"] },
      { command: "echo", args: ["d"] },
    ];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "batch_execute", {
      commands: fourCommands,
      concurrency: 2,
      timeout_ms: timeoutMs,
    });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0, "an overflowing worst-case timeout must never reach a spawned CLI call");
  });
});

describe("batch_execute CLI argv assembly", () => {
  it("passes --commands-json with the exact JSON.stringify(commands) string", async () => {
    const capturedArgv: string[][] = [];
    await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify([])),
      "batch_execute",
      { commands: ONE_COMMAND },
    );
    assert.ok(capturedArgv[0].includes("--commands-json"));
    assert.ok(capturedArgv[0].includes(JSON.stringify(ONE_COMMAND)));
  });

  it("passes --timeout-ms only when timeout_ms was supplied", async () => {
    const withTimeout: string[][] = [];
    await callTool(
      fakeSpawnCapturingWithPayload(withTimeout, JSON.stringify([])),
      "batch_execute",
      { commands: ONE_COMMAND, timeout_ms: 5000 },
    );
    assert.ok(withTimeout[0].includes("--timeout-ms"));
    assert.ok(withTimeout[0].includes("5000"));

    const withoutTimeout: string[][] = [];
    await callTool(
      fakeSpawnCapturingWithPayload(withoutTimeout, JSON.stringify([])),
      "batch_execute",
      { commands: ONE_COMMAND },
    );
    assert.equal(withoutTimeout[0].includes("--timeout-ms"), false);
  });
});

describe("batch_execute per-command failure passthrough", () => {
  it("surfaces a failing command as a results element, not an MCP error", async () => {
    const results = [{ run_id: "a", state: "exited:1", stdout: "", stderr: "" }];
    const result = await callTool(fakeSpawn(JSON.stringify(results)), "batch_execute", {
      commands: ONE_COMMAND,
    });
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, { results });
  });
});

describe("batch_execute schema shape", () => {
  it("declares no auto-index property on its inputSchema", async () => {
    const { server } = buildServer(fakeSpawn(""));
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "0.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    const { tools } = await client.listTools();
    const batchExecute = tools.find((t) => t.name === "batch_execute");
    assert.ok(batchExecute, "expected batch_execute to be listed");
    const properties = (batchExecute!.inputSchema as { properties?: Record<string, unknown> }).properties ?? {};
    assert.equal("auto_index" in properties, false);
  });
});
