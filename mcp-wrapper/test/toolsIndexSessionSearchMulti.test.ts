import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { EventEmitter } from "node:events";
import type { spawn as SpawnFn, ChildProcess } from "node:child_process";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { buildServer } from "../src/tools-index.js";

function fakeSpawn(capturedArgv: string[][], stdout = ""): typeof SpawnFn {
  const fn = (_cmd: string, argv: readonly string[]) => {
    capturedArgv.push([...argv]);
    const stdoutStream = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const stderrStream = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const proc = Object.assign(new EventEmitter(), {
      stdout: stdoutStream,
      stderr: stderrStream,
      kill: () => {},
    });
    queueMicrotask(() => {
      stdoutStream.emit("data", stdout);
      proc.emit("close", 0);
    });
    return proc as unknown as ChildProcess;
  };
  return fn as unknown as typeof SpawnFn;
}

async function callSessionSearch(
  capturedArgv: string[][],
  args: Record<string, unknown>,
): Promise<void> {
  const { server } = buildServer(fakeSpawn(capturedArgv));
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test-client", version: "0.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  await client.callTool({ name: "session_search", arguments: args });
}

async function callSessionSearchForResult(
  args: Record<string, unknown>,
  stdout = "",
): Promise<Record<string, unknown>> {
  const capturedArgv: string[][] = [];
  const { server } = buildServer(fakeSpawn(capturedArgv, stdout));
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test-client", version: "0.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  const result = await client.callTool({ name: "session_search", arguments: args });
  return result.structuredContent as Record<string, unknown>;
}

describe("session_search explicit multi-id union", () => {
  it("builds one CLI call: first id positional, the rest as --also-session", async () => {
    const capturedArgv: string[][] = [];
    await callSessionSearch(capturedArgv, { session_id: "A", session_ids: ["B"] });

    assert.equal(capturedArgv.length, 1);
    const argv = capturedArgv[0];
    const sub = argv.indexOf("session-search");
    assert.notEqual(sub, -1, `expected session-search in argv, got: ${JSON.stringify(argv)}`);
    assert.equal(argv[sub + 1], "A");
    assert.deepEqual(argv.slice(sub + 2), ["--also-session", "B"]);
  });

  it("passes multiple additional ids as repeated --also-session flags, in order", async () => {
    const capturedArgv: string[][] = [];
    await callSessionSearch(capturedArgv, { session_id: "A", session_ids: ["B", "C"] });

    const argv = capturedArgv[0];
    const sub = argv.indexOf("session-search");
    assert.equal(argv[sub + 1], "A");
    assert.deepEqual(argv.slice(sub + 2), ["--also-session", "B", "--also-session", "C"]);
  });

  it("leaves the single-id request path unchanged (no --also-session emitted)", async () => {
    const capturedArgv: string[][] = [];
    await callSessionSearch(capturedArgv, { session_id: "A" });

    const argv = capturedArgv[0];
    const sub = argv.indexOf("session-search");
    assert.equal(argv[sub + 1], "A");
    assert.equal(argv.includes("--also-session"), false, JSON.stringify(argv));
  });

  it("ignores an empty session_ids array the same as omitting it", async () => {
    const capturedArgv: string[][] = [];
    await callSessionSearch(capturedArgv, { session_id: "A", session_ids: [] });

    const argv = capturedArgv[0];
    assert.equal(argv.includes("--also-session"), false, JSON.stringify(argv));
  });
});

describe("session_search response envelope reflects actual query scope", () => {
  it("single-id query: payload carries scalar session_id, no session_ids field", async () => {
    const payload = await callSessionSearchForResult({ session_id: "A" });

    assert.equal(payload.session_id, "A");
    assert.equal("session_ids" in payload, false, JSON.stringify(payload));
    assert.deepEqual(payload.events, []);
  });

  it("multi-id query: payload carries session_ids list, no scalar session_id claiming sole ownership", async () => {
    const payload = await callSessionSearchForResult({ session_id: "A", session_ids: ["B", "C"] });

    assert.equal("session_id" in payload, false, JSON.stringify(payload));
    assert.deepEqual(payload.session_ids, ["A", "B", "C"]);
    assert.deepEqual(payload.events, []);
  });

  it("empty session_ids array: treated as single-id, scalar session_id shape unchanged", async () => {
    const payload = await callSessionSearchForResult({ session_id: "A", session_ids: [] });

    assert.equal(payload.session_id, "A");
    assert.equal("session_ids" in payload, false, JSON.stringify(payload));
  });
});
