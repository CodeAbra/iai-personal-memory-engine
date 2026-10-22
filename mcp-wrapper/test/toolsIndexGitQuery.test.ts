import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { EventEmitter } from "node:events";
import type { spawn as SpawnFn, ChildProcess } from "node:child_process";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { buildServer } from "../src/tools-index.js";

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

function fakeSpawnCapturingNoCall(capturedArgv: string[][]): typeof SpawnFn {
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

// Simulates the shared body's non-zero-exit rejection (the same shape a
// real unqualified git-log/git-blame/git-diff CLI call produces on stderr).
function fakeSpawnRejecting(capturedArgv: string[][], stderrMessage: string, exitCode = 1): typeof SpawnFn {
  const fn = (_cmd: string, argv: readonly string[]) => {
    capturedArgv.push([...argv]);
    const stdout = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const stderr = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const proc = Object.assign(new EventEmitter(), { stdout, stderr, kill: () => {} });
    queueMicrotask(() => {
      stderr.emit("data", stderrMessage);
      proc.emit("close", exitCode);
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

const BOUNDED_PAYLOAD = {
  run_id: "a",
  state: "exited:0",
  stdout: "output",
  stderr: "",
  truncated: false,
  elided_bytes: 0,
};

describe("execute git-query mode dispatch", () => {
  it("spawns git-log with --json for mode=log and returns structuredContent", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify(BOUNDED_PAYLOAD)),
      "execute",
      { mode: "log", since: "1970-01-01" },
    );
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, BOUNDED_PAYLOAD);
    assert.ok(capturedArgv[0].includes("git-log"));
    assert.ok(capturedArgv[0].includes("--json"));
    assert.ok(capturedArgv[0].includes("--since"));
    assert.ok(capturedArgv[0].includes("1970-01-01"));
  });

  it("spawns git-blame with --json for mode=blame and returns structuredContent", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify(BOUNDED_PAYLOAD)),
      "execute",
      { mode: "blame", file: "a.txt" },
    );
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, BOUNDED_PAYLOAD);
    assert.ok(capturedArgv[0].includes("git-blame"));
    assert.ok(capturedArgv[0].includes("--json"));
    assert.ok(capturedArgv[0].includes("a.txt"));
  });

  it("spawns git-blame with a wrapper-built line range", async () => {
    const capturedArgv: string[][] = [];
    await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify(BOUNDED_PAYLOAD)),
      "execute",
      { mode: "blame", file: "a.txt", start_line: 2, end_line: 5 },
    );
    assert.ok(capturedArgv[0].includes("--start-line"));
    assert.ok(capturedArgv[0].includes("2"));
    assert.ok(capturedArgv[0].includes("--end-line"));
    assert.ok(capturedArgv[0].includes("5"));
  });

  it("spawns git-diff with --json for mode=diff and returns structuredContent", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify(BOUNDED_PAYLOAD)),
      "execute",
      { mode: "diff", rev_a: "HEAD~1", rev_b: "HEAD" },
    );
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, BOUNDED_PAYLOAD);
    assert.ok(capturedArgv[0].includes("git-diff"));
    assert.ok(capturedArgv[0].includes("--json"));
    assert.ok(capturedArgv[0].includes("HEAD~1"));
    assert.ok(capturedArgv[0].includes("HEAD"));
  });

  it("spawns git-diff scoped by --path", async () => {
    const capturedArgv: string[][] = [];
    await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify(BOUNDED_PAYLOAD)),
      "execute",
      { mode: "diff", path: "src" },
    );
    assert.ok(capturedArgv[0].includes("--path"));
    assert.ok(capturedArgv[0].includes("src"));
  });

  it("accepts a mode-only call with no command", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify(BOUNDED_PAYLOAD)),
      "execute",
      { mode: "log", path: "src" },
    );
    assert.equal(result.isError, undefined);
  });

  it("dispatches a plain execute with a command and no mode unchanged", async () => {
    const capturedArgv: string[][] = [];
    const payload = { run_id: "a", state: "exited:0", stdout: "hi\n", stderr: "" };
    const result = await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify(payload)),
      "execute",
      { command: "echo", args: ["hi"] },
    );
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, payload);
    assert.ok(capturedArgv[0].includes("run"));
    assert.ok(capturedArgv[0].includes("echo"));
    assert.equal(capturedArgv[0].includes("git-log"), false);
  });

  it("rejects execute with neither command nor mode, before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturingNoCall(capturedArgv), "execute", {});
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });

  it("rejects an unrecognized mode value, before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturingNoCall(capturedArgv), "execute", {
      mode: "show",
    });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });

  it("surfaces an unqualified git-query mode call as an error result", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(
      fakeSpawnRejecting(
        capturedArgv,
        "git-log: supply --since/--until, a revision range, or --path, to bound this query",
      ),
      "execute",
      { mode: "log" },
    );
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /to bound this query/);
  });
});

describe("execute git-query mode schema shape", () => {
  it("declares a mode enum with all three git-query shapes", async () => {
    const { server } = buildServer(fakeSpawnCapturingWithPayload([], ""));
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "0.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    const { tools } = await client.listTools();
    const executeTool = tools.find((t) => t.name === "execute");
    assert.ok(executeTool, "expected execute to be listed");
    const properties = (executeTool!.inputSchema as { properties?: Record<string, any> }).properties ?? {};
    assert.deepEqual(properties.mode?.enum, ["log", "blame", "diff"]);
  });

  it("keeps the MCP tool count at exactly 7", async () => {
    const { server } = buildServer(fakeSpawnCapturingWithPayload([], ""));
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "0.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    const { tools } = await client.listTools();
    assert.equal(tools.length, 7);
  });
});
