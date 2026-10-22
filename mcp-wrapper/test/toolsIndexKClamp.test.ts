import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { EventEmitter } from "node:events";
import type { spawn as SpawnFn, ChildProcess } from "node:child_process";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { buildServer } from "../src/tools-index.js";

function fakeSpawn(capturedArgv: string[][]): typeof SpawnFn {
  const fn = (_cmd: string, argv: readonly string[]) => {
    capturedArgv.push([...argv]);
    const stdout = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const stderr = Object.assign(new EventEmitter(), { setEncoding: () => {} });
    const proc = Object.assign(new EventEmitter(), { stdout, stderr, kill: () => {} });
    queueMicrotask(() => {
      stdout.emit("data", JSON.stringify({ query: "q", lane: "whole", hits: [] }));
      proc.emit("close", 0);
    });
    return proc as unknown as ChildProcess;
  };
  return fn as unknown as typeof SpawnFn;
}

describe("code_search k clamp", () => {
  it("clamps an oversized k in the argv handed to the CLI, never the raw caller value", async () => {
    const capturedArgv: string[][] = [];
    const { server } = buildServer(fakeSpawn(capturedArgv));

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "0.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);

    await client.callTool({ name: "code_search", arguments: { query: "anything", k: 999999 } });

    assert.equal(capturedArgv.length, 1);
    const argv = capturedArgv[0];
    const kIdx = argv.indexOf("--k");
    assert.notEqual(kIdx, -1, `expected --k in argv, got: ${JSON.stringify(argv)}`);
    assert.equal(argv[kIdx + 1], "200", "an oversized k must be clamped before spawning the CLI");
  });

  it("leaves a within-bound k untouched", async () => {
    const capturedArgv: string[][] = [];
    const { server } = buildServer(fakeSpawn(capturedArgv));

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "0.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);

    await client.callTool({ name: "code_search", arguments: { query: "anything", k: 15 } });

    const argv = capturedArgv[0];
    const kIdx = argv.indexOf("--k");
    assert.equal(argv[kIdx + 1], "15");
  });

  it("rejects a non-positive k instead of spawning the CLI with it", async () => {
    const capturedArgv: string[][] = [];
    const { server } = buildServer(fakeSpawn(capturedArgv));

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "0.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);

    for (const badK of [0, -1]) {
      const result = await client.callTool({ name: "code_search", arguments: { query: "anything", k: badK } });
      assert.equal(result.isError, true, `k=${badK} must be rejected, never silently clamped or spawned`);
    }
    assert.equal(capturedArgv.length, 0, "a rejected k must never reach the CLI spawn");
  });
});
