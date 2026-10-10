
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { EventEmitter } from "node:events";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { defaultSpawnDoctorAuto } from "../src/lifecycle.js";
import { probeDaemonDoctor } from "../src/sickWarning.js";
import { buildServer } from "../src/tools-index.js";
import {
  runDirectRecency,
  runDirectWrite,
  runDirectRecall,
  runDirectTemporal,
  runDirectSearch,
  runBankFallback,
} from "../src/tools.js";

type CapturedSpawn = { cmd: string; args: string[]; opts: Record<string, unknown> };

type MockProc = EventEmitter & {
  stdout: EventEmitter & { setEncoding: (enc: string) => void };
  stderr: EventEmitter & { setEncoding: (enc: string) => void };
  kill: () => void;
  unref: () => void;
};

function makeMockProc(): MockProc {
  const proc = new EventEmitter() as MockProc;
  const stdout = new EventEmitter() as MockProc["stdout"];
  stdout.setEncoding = () => {};
  const stderr = new EventEmitter() as MockProc["stderr"];
  stderr.setEncoding = () => {};
  proc.stdout = stdout;
  proc.stderr = stderr;
  proc.kill = () => {};
  proc.unref = () => {};
  return proc;
}

// One capturing spawnFn double per assertion, shared shape: record cmd/args/opts,
// then emit stdout data (parseable JSON) followed by a clean exit so the caller's
// promise resolves and the test can assert on the captured options object.
function makeCapturingSpawn(payload: Record<string, unknown> = {}) {
  const calls: CapturedSpawn[] = [];
  const spawnFn = ((cmd: string, args: ReadonlyArray<string>, opts: Record<string, unknown>) => {
    calls.push({ cmd, args: [...args], opts });
    const proc = makeMockProc();
    setImmediate(() => {
      proc.stdout.emit("data", JSON.stringify(payload));
      proc.emit("close", 0);
    });
    return proc as unknown as ReturnType<typeof import("node:child_process").spawn>;
  }) as unknown as typeof import("node:child_process").spawn;
  return { spawnFn, calls };
}

describe("windowsHide: every spawn() call site suppresses the Windows console window", () => {
  it("lifecycle.ts defaultSpawnDoctorAuto passes windowsHide with detached/stdio intact", async () => {
    let captured: CapturedSpawn | null = null;
    const mockSpawn = ((cmd: string, args: ReadonlyArray<string>, opts: Record<string, unknown>) => {
      captured = { cmd, args: [...args], opts };
      const proc = new EventEmitter() as EventEmitter & { unref: () => void };
      proc.unref = () => {};
      return proc as unknown as ReturnType<typeof import("node:child_process").spawn>;
    }) as unknown as typeof import("node:child_process").spawn;

    const spawnDoctorAuto = defaultSpawnDoctorAuto(mockSpawn);
    await spawnDoctorAuto();

    assert.ok(captured, "spawn must be invoked");
    const c = captured as unknown as CapturedSpawn;
    assert.equal(c.cmd, "iai-mcp");
    assert.deepEqual(c.args, ["doctor", "--auto"]);
    assert.equal(c.opts.windowsHide, true, "windowsHide must be true");
    assert.equal(c.opts.detached, true, "detached must be preserved");
    assert.equal(c.opts.stdio, "ignore", "stdio must be preserved");
  });

  it("sickWarning.ts probeDaemonDoctor passes windowsHide with stdio intact", async () => {
    let captured: CapturedSpawn | null = null;
    const mockSpawn = ((cmd: string, args: ReadonlyArray<string>, opts: Record<string, unknown>) => {
      captured = { cmd, args: [...args], opts };
      const proc = makeMockProc();
      setImmediate(() => proc.emit("close", 0));
      return proc as unknown as ReturnType<typeof import("node:child_process").spawn>;
    }) as unknown as typeof import("node:child_process").spawn;

    const code = await probeDaemonDoctor(mockSpawn, 5_000);

    assert.equal(code, 0);
    assert.ok(captured, "spawn must be invoked");
    const c = captured as unknown as CapturedSpawn;
    assert.equal(c.opts.windowsHide, true, "windowsHide must be true");
    assert.deepEqual(c.opts.stdio, ["ignore", "ignore", "ignore"], "stdio must be preserved");
  });

  it("tools-index.ts runCli (via code_search) passes windowsHide with stdio intact", async () => {
    const captured: CapturedSpawn[] = [];
    const fakeSpawn = ((cmd: string, args: ReadonlyArray<string>, opts: Record<string, unknown>) => {
      captured.push({ cmd, args: [...args], opts });
      const proc = makeMockProc();
      setImmediate(() => {
        proc.stdout.emit("data", JSON.stringify({ query: "q", lane: "whole", hits: [] }));
        proc.emit("close", 0);
      });
      return proc as unknown as ReturnType<typeof import("node:child_process").spawn>;
    }) as unknown as typeof import("node:child_process").spawn;

    const { server } = buildServer(fakeSpawn);
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "0.0.0" });
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);

    await client.callTool({ name: "code_search", arguments: { query: "anything" } });

    assert.equal(captured.length, 1, "exactly one subprocess must be spawned");
    const c = captured[0];
    assert.equal(c.opts.windowsHide, true, "windowsHide must be true");
    assert.deepEqual(c.opts.stdio, ["ignore", "pipe", "pipe"], "stdio must be preserved");
  });

  it("tools.ts runDirectRecency passes windowsHide with stdio intact", async () => {
    const { spawnFn, calls } = makeCapturingSpawn({ turns: [], count: 0 });
    await runDirectRecency({ n: 5 }, spawnFn);
    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    assert.equal(calls[0].opts.windowsHide, true, "windowsHide must be true");
    assert.deepEqual(calls[0].opts.stdio, ["ignore", "pipe", "pipe"], "stdio must be preserved");
  });

  it("tools.ts runDirectWrite passes windowsHide with stdio intact", async () => {
    const { spawnFn, calls } = makeCapturingSpawn({});
    await runDirectWrite({ literal: "a captured turn" }, spawnFn);
    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    assert.equal(calls[0].opts.windowsHide, true, "windowsHide must be true");
    assert.deepEqual(calls[0].opts.stdio, ["ignore", "pipe", "pipe"], "stdio must be preserved");
  });

  it("tools.ts runDirectRecall passes windowsHide with stdio intact", async () => {
    const { spawnFn, calls } = makeCapturingSpawn({ hits: [] });
    await runDirectRecall({ cue: "probe" }, spawnFn);
    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    assert.equal(calls[0].opts.windowsHide, true, "windowsHide must be true");
    assert.deepEqual(calls[0].opts.stdio, ["ignore", "pipe", "pipe"], "stdio must be preserved");
  });

  it("tools.ts runDirectTemporal passes windowsHide with stdio intact", async () => {
    const { spawnFn, calls } = makeCapturingSpawn({});
    await runDirectTemporal({ cue: "probe" }, spawnFn);
    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    assert.equal(calls[0].opts.windowsHide, true, "windowsHide must be true");
    assert.deepEqual(calls[0].opts.stdio, ["ignore", "pipe", "pipe"], "stdio must be preserved");
  });

  it("tools.ts runDirectSearch passes windowsHide with stdio intact", async () => {
    const { spawnFn, calls } = makeCapturingSpawn({});
    await runDirectSearch({ query: "probe" }, spawnFn);
    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    assert.equal(calls[0].opts.windowsHide, true, "windowsHide must be true");
    assert.deepEqual(calls[0].opts.stdio, ["ignore", "pipe", "pipe"], "stdio must be preserved");
  });

  it("tools.ts runBankFallback passes windowsHide with stdio intact", async () => {
    const { spawnFn, calls } = makeCapturingSpawn({ hits: [] });
    await runBankFallback("probe", 5, spawnFn);
    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    assert.equal(calls[0].opts.windowsHide, true, "windowsHide must be true");
    assert.deepEqual(calls[0].opts.stdio, ["ignore", "pipe", "pipe"], "stdio must be preserved");
  });
});
