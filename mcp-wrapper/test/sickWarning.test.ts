
import { strict as assert } from "node:assert";
import { afterEach, describe, it } from "node:test";
import { EventEmitter } from "node:events";

import {
  emitSickWarningIfNeeded,
  probeDaemonDoctor,
} from "../src/sickWarning.js";

type StreamLike = EventEmitter & { setEncoding: (enc: string) => void };
type ProcLike = EventEmitter & {
  stdout: StreamLike;
  stderr: StreamLike;
  kill: () => void;
};

function makeMockProc(opts: {
  exitCode?: number | null;
  emitErrorBeforeClose?: boolean;
}): ProcLike {
  const proc = new EventEmitter() as ProcLike;
  const stdout = new EventEmitter() as StreamLike;
  stdout.setEncoding = () => {};
  proc.stdout = stdout;
  const stderr = new EventEmitter() as StreamLike;
  stderr.setEncoding = () => {};
  proc.stderr = stderr;
  proc.kill = () => {};
  if (opts.emitErrorBeforeClose) {
    setImmediate(() => proc.emit("error", new Error("spawn ENOENT")));
  } else if (opts.exitCode !== undefined) {
    setImmediate(() => proc.emit("close", opts.exitCode));
  }
  return proc;
}

const originalStderrWrite = process.stderr.write.bind(process.stderr);
const stderrLines: string[] = [];

function captureStderr(): void {
  process.stderr.write = ((line: string | Uint8Array) => {
    stderrLines.push(typeof line === "string" ? line : line.toString("utf-8"));
    return true;
  }) as typeof process.stderr.write;
}

afterEach(() => {
  process.stderr.write = originalStderrWrite;
  stderrLines.length = 0;
});

describe("probeDaemonDoctor", () => {
  it("exit 0 -> no stderr line", async () => {
    captureStderr();
    const mockSpawn = (_cmd: string, _args: ReadonlyArray<string>) =>
      makeMockProc({ exitCode: 0 }) as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    const code = await probeDaemonDoctor(mockSpawn as any, 5_000);
    emitSickWarningIfNeeded(code);
    assert.equal(code, 0);
    assert.equal(stderrLines.length, 0, "expected zero stderr writes on PASS");
  });

  it("exit 1 -> exactly one stderr line with exit=1", async () => {
    captureStderr();
    const mockSpawn = () =>
      makeMockProc({ exitCode: 1 }) as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    const code = await probeDaemonDoctor(mockSpawn as any, 5_000);
    emitSickWarningIfNeeded(code);
    assert.equal(code, 1);
    assert.equal(stderrLines.length, 1, "expected one stderr write on FAIL");
    const line = stderrLines[0];
    assert.ok(line.includes("iai-mcp warning"), `missing label: ${line}`);
    assert.ok(line.includes("daemon doctor"), `missing subject: ${line}`);
    assert.ok(line.includes("exit=1"), `missing exit code: ${line}`);
    assert.ok(line.endsWith("\n"), `line must end with newline: ${line!}`);
  });

  it("exit 2 -> exactly one stderr line with exit=2", async () => {
    captureStderr();
    const mockSpawn = () =>
      makeMockProc({ exitCode: 2 }) as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    const code = await probeDaemonDoctor(mockSpawn as any, 5_000);
    emitSickWarningIfNeeded(code);
    assert.equal(code, 2);
    assert.equal(stderrLines.length, 1);
    assert.ok(
      stderrLines[0].includes("exit=2"),
      `expected exit=2 in line: ${stderrLines[0]}`,
    );
  });

  it("spawn error -> silent, probe resolves null", async () => {
    captureStderr();
    const mockSpawn = () =>
      makeMockProc({ emitErrorBeforeClose: true }) as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    const code = await probeDaemonDoctor(mockSpawn as any, 5_000);
    emitSickWarningIfNeeded(code);
    assert.equal(code, null, "expected null on spawn error");
    assert.equal(stderrLines.length, 0, "expected silent degrade on spawn error");
  });

  it("timeout -> silent, probe resolves null", async () => {
    captureStderr();
    const mockSpawn = () =>
      makeMockProc({}) as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    const code = await probeDaemonDoctor(mockSpawn as any, 50);
    emitSickWarningIfNeeded(code);
    assert.equal(code, null, "expected null on timeout");
    assert.equal(stderrLines.length, 0, "expected silent degrade on timeout");
  });

  it("invokes spawn with `iai-mcp` and args `[\"doctor\"]` (NOT `[\"daemon\", \"doctor\"]`)", async () => {
    captureStderr();
    const calls: Array<{ cmd: string; args: string[] }> = [];
    const mockSpawn = (cmd: string, args: ReadonlyArray<string>) => {
      calls.push({ cmd, args: [...args] });
      return makeMockProc({ exitCode: 0 }) as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    };
    const prevCli = process.env.IAI_MCP_CLI;
    delete process.env.IAI_MCP_CLI;
    try {
      await probeDaemonDoctor(mockSpawn as any, 5_000);
    } finally {
      if (prevCli !== undefined) {
        process.env.IAI_MCP_CLI = prevCli;
      }
    }
    assert.equal(calls.length, 1);
    assert.equal(calls[0].cmd, "iai-mcp");
    assert.deepEqual(
      calls[0].args,
      ["doctor"],
      `argv must be ["doctor"], got ${JSON.stringify(calls[0].args)}`,
    );
  });
});

describe("emitSickWarningIfNeeded", () => {
  it("null exit code is a silent no-op", () => {
    captureStderr();
    emitSickWarningIfNeeded(null);
    assert.equal(stderrLines.length, 0);
  });

  it("exit 0 is a silent no-op", () => {
    captureStderr();
    emitSickWarningIfNeeded(0);
    assert.equal(stderrLines.length, 0);
  });
});
