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

async function listTools(spawnFn: typeof SpawnFn) {
  const { server } = buildServer(spawnFn);
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test-client", version: "0.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  return client.listTools();
}

describe("code_search dependency store dispatch", () => {
  it("spawns search --store dependency with --ecosystem/--package", async () => {
    const capturedArgv: string[][] = [];
    const payload = { query: "q", lane: "whole", hits: [], status: "not_indexed" };
    const result = await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, JSON.stringify(payload)),
      "code_search",
      { query: "q", store: "dependency", ecosystem: "python", package: "alicelib" },
    );
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, payload);
    assert.ok(capturedArgv[0].includes("--store"));
    assert.ok(capturedArgv[0].includes("dependency"));
    assert.ok(capturedArgv[0].includes("--ecosystem"));
    assert.ok(capturedArgv[0].includes("python"));
    assert.ok(capturedArgv[0].includes("--package"));
    assert.ok(capturedArgv[0].includes("alicelib"));
  });

  it("rejects store='dependency' missing ecosystem/package, before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturingNoCall(capturedArgv), "code_search", {
      query: "q",
      store: "dependency",
    });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });

  it("rejects store='dependency' with a non-whole lane, before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturingNoCall(capturedArgv), "code_search", {
      query: "q",
      store: "dependency",
      ecosystem: "python",
      package: "alicelib",
      lane: "symbol",
    });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });

  it("rejects an unrecognized store value, before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturingNoCall(capturedArgv), "code_search", {
      query: "q",
      store: "not-a-real-store",
    });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });
});

describe("code_index dependency spec dispatch", () => {
  it("spawns index --ecosystem/--package/--project-dir for a dependency spec", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(
      fakeSpawnCapturingWithPayload(
        capturedArgv,
        "indexed 3 files under /home/.iai-mcp/tools-deps/python/alicelib for python package alicelib\n",
      ),
      "code_index",
      { ecosystem: "python", package: "alicelib", project_dir: "/proj" },
    );
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, {
      ecosystem: "python",
      package: "alicelib",
      project_dir: "/proj",
      indexed_files: 3,
    });
    assert.ok(capturedArgv[0].includes("index"));
    assert.ok(capturedArgv[0].includes("--ecosystem"));
    assert.ok(capturedArgv[0].includes("python"));
    assert.ok(capturedArgv[0].includes("--package"));
    assert.ok(capturedArgv[0].includes("alicelib"));
    assert.ok(capturedArgv[0].includes("--project-dir"));
    assert.ok(capturedArgv[0].includes("/proj"));
    assert.equal(capturedArgv[0].includes("/proj".concat("-not-a-path")), false);
  });

  it("still dispatches a plain path index unchanged", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(
      fakeSpawnCapturingWithPayload(capturedArgv, "indexed 5 files under /repo\n"),
      "code_index",
      { path: "/repo" },
    );
    assert.equal(result.isError, undefined);
    assert.deepEqual(result.structuredContent, { path: "/repo", indexed_files: 5 });
    assert.ok(capturedArgv[0].includes("index"));
    assert.ok(capturedArgv[0].includes("/repo"));
    assert.equal(capturedArgv[0].includes("--ecosystem"), false);
  });

  it("rejects both path and a dependency spec, before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturingNoCall(capturedArgv), "code_index", {
      path: "/repo",
      ecosystem: "python",
      package: "alicelib",
      project_dir: "/proj",
    });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });

  it("rejects neither path nor a dependency spec, before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturingNoCall(capturedArgv), "code_index", {});
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });

  it("rejects a partial dependency spec, before spawning the CLI", async () => {
    const capturedArgv: string[][] = [];
    const result = await callTool(fakeSpawnCapturingNoCall(capturedArgv), "code_index", {
      ecosystem: "python",
      package: "alicelib",
    });
    assert.equal(result.isError, true);
    assert.equal(capturedArgv.length, 0);
  });
});

describe("dependency store schema shape and tool count", () => {
  it("declares the store enum as exactly [code, output, dependency]", async () => {
    const { tools } = await listTools(fakeSpawnCapturingWithPayload([], ""));
    const codeSearchTool = tools.find((t) => t.name === "code_search");
    assert.ok(codeSearchTool, "expected code_search to be listed");
    const properties = (codeSearchTool!.inputSchema as { properties?: Record<string, any> }).properties ?? {};
    assert.deepEqual(properties.store?.enum, ["code", "output", "dependency"]);
  });

  it("keeps the MCP tool count at exactly 7", async () => {
    const { tools } = await listTools(fakeSpawnCapturingWithPayload([], ""));
    assert.equal(tools.length, 7);
  });
});
