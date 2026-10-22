import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { EventEmitter } from "node:events";
import type { spawn as SpawnFn, ChildProcess } from "node:child_process";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { buildServer } from "../src/tools-index.js";

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

describe("execute JSON-parse path", () => {
  it("returns structuredContent for well-formed CLI JSON output", async () => {
    const payload = { run_id: "abc123", state: "exited:0", stdout: "hi", stderr: "" };
    const result = await callTool(fakeSpawn(JSON.stringify(payload)), "execute", {
      command: "echo",
      args: ["hi"],
    });
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, payload);
  });

  it("surfaces indexed/refused/index_error fields when the CLI emits them", async () => {
    const payload = {
      run_id: "abc123",
      state: "exited:0",
      stdout: "hi",
      stderr: "",
      indexed: 1,
      refused: [],
    };
    const result = await callTool(fakeSpawn(JSON.stringify(payload)), "execute", {
      command: "echo",
      args: ["hi"],
      auto_index: true,
    });
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, payload);
  });

  it("returns a parse-error result instead of throwing when the CLI emits non-JSON stdout", async () => {
    const result = await callTool(fakeSpawn("not json"), "execute", {
      command: "echo",
      args: ["hi"],
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /execute: failed to parse CLI JSON output/);
  });
});

describe("code_search store selector validation", () => {
  it("rejects a store value outside the enum before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "code_search", {
      query: "q",
      store: "not-a-real-store",
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /store must be one of/);
    assert.equal(capturedArgv.length, 0, "an invalid store value must never reach a spawned CLI call");
  });

  it("rejects store='output' combined with a non-whole lane", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturing(capturedArgv), "code_search", {
      query: "q",
      store: "output",
      lane: "symbol",
    });
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /only supported with lane='whole'/);
    assert.equal(capturedArgv.length, 0);
  });

  it("passes a valid store='output' through to the spawned CLI argv", async () => {
    const capturedArgv: string[][] = [];
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
    const result = await callTool(fn as unknown as typeof SpawnFn, "code_search", {
      query: "q",
      store: "output",
    });
    assert.equal(result.isError, undefined);
    assert.ok(capturedArgv[0].includes("--store"));
    assert.ok(capturedArgv[0].includes("output"));
  });
});

describe("command_insight", () => {
  it("returns the parsed failing_runs/most_recent_failure shape", async () => {
    const payload = { failing_runs: [], most_recent_failure: null };
    const result = await callTool(fakeSpawn(JSON.stringify(payload)), "command_insight", {});
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, payload);
  });

  it("returns a parse-error result instead of throwing on malformed CLI output", async () => {
    const result = await callTool(fakeSpawn("{not valid json"), "command_insight", {});
    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /command_insight: failed to parse CLI output/);
  });
});
