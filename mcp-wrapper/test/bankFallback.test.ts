
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { EventEmitter } from "node:events";

import { runBankFallback, invokeTool } from "../src/tools.js";
import type { PythonCoreBridge } from "../src/bridge.js";

describe("runBankFallback", () => {
  it("spawns iai-mcp bank-recall and tags the response with _source", async () => {
    const calls: Array<{ cmd: string; args: string[] }> = [];
    const mockSpawn = (cmd: string, args: ReadonlyArray<string>) => {
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
        stdout.emit(
          "data",
          JSON.stringify({
            hits: [
              {
                record_id: "abc",
                score: 0.5,
                reason: "bank-substring-match (processed)",
                literal_surface: "hello carrot",
                adjacent_suggestions: [],
                valid_from: null,
                valid_to: null,
              },
            ],
            anti_hits: [],
            activation_trace: [],
            budget_used: 0,
            cue_mode: "verbatim",
            patterns_observed: [],
            _knobs_applied: {},
          }),
        );
        proc.emit("close", 0);
      });
      return proc as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    };
    const result = await runBankFallback("carrot", 20, mockSpawn as any);
    assert.ok(result, "expected non-null fallback payload");
    assert.equal(result!["_source"], "bank-fallback");
    assert.equal((result!["hits"] as any[]).length, 1);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].cmd, "iai-mcp");
    assert.deepEqual(calls[0].args, [
      "bank-recall",
      "--query",
      "carrot",
      "--limit",
      "20",
      "--json",
    ]);
  });

  it("returns null when subprocess exits non-zero", async () => {
    const mockSpawn = () => {
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
      setImmediate(() => proc.emit("close", 1));
      return proc as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    };
    const result = await runBankFallback("anything", 5, mockSpawn as any);
    assert.equal(result, null);
  });
});

describe("invokeTool memory_recall budget-vs-limit decoupling", () => {
  it(
    "invokeTool budget-vs-limit decoupling: when direct-store + bridge both fail, bank receives limit=BANK_FALLBACK_LIMIT (20), NOT budget_tokens",
    async () => {
      const spawnCalls: Array<{ cmd: string; args: string[] }> = [];
      let spawnCount = 0;

      const mockSpawnFn = (cmd: string, args: ReadonlyArray<string>) => {
        const callIndex = spawnCount++;
        spawnCalls.push({ cmd, args: [...args] });

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
          if (callIndex === 0) {
            proc.emit("close", 1);
          } else {
            stdout.emit(
              "data",
              JSON.stringify({
                hits: [],
                anti_hits: [],
                activation_trace: [],
                budget_used: 0,
                cue_mode: "verbatim",
                patterns_observed: [],
                _knobs_applied: {},
              }),
            );
            proc.emit("close", 0);
          }
        });
        return proc as unknown as ReturnType<
          typeof import("node:child_process").spawn
        >;
      };

      const mockBridge = {
        call: async () => { throw new Error("socket dead"); },
      } as unknown as PythonCoreBridge;

      const prevFallback = process.env["IAI_MCP_BANK_FALLBACK"];
      delete process.env["IAI_MCP_BANK_FALLBACK"];
      try {
        await invokeTool(
          mockBridge,
          "memory_recall",
          { cue: "test", budget_tokens: 9999 },
          mockSpawnFn as any,
        );
      } finally {
        if (prevFallback !== undefined) {
          process.env["IAI_MCP_BANK_FALLBACK"] = prevFallback;
        }
      }

      assert.ok(spawnCalls.length >= 2, `expected at least 2 spawns; got ${spawnCalls.length}`);

      const bankArgs = spawnCalls[1].args;
      const limitIdx = bankArgs.indexOf("--limit");
      assert.ok(limitIdx >= 0, "--limit flag must be present in bank-recall argv");
      assert.equal(
        bankArgs[limitIdx + 1],
        "20",
        "invokeTool must pass BANK_FALLBACK_LIMIT (20) as --limit to bank, not budget_tokens",
      );
      assert.notEqual(
        bankArgs[limitIdx + 1],
        "9999",
        "--limit must not equal budget_tokens (9999)",
      );
    },
  );
});

describe("invokeTool memory_search dispatch", () => {
  it("socket-first: a live bridge serves memory_search directly", async () => {
    const calls: Array<{ method: string }> = [];
    const mockBridge = {
      call: async (method: string) => {
        calls.push({ method });
        return { hits: [{ surface: "x" }], frame: "hints" };
      },
    } as unknown as PythonCoreBridge;

    const out = (await invokeTool(
      mockBridge,
      "memory_search",
      { query: "HippoDB", k: 4 },
      (() => { throw new Error("spawn must not be reached"); }) as any,
    )) as Record<string, unknown>;
    assert.equal(calls.length, 1);
    assert.equal(calls[0]!.method, "memory_search");
    assert.ok(Array.isArray(out["hits"]));
  });

  it("daemon-down: memory_search falls back to the direct CLI", async () => {
    const spawnCalls: Array<{ cmd: string; args: string[] }> = [];
    const mockSpawnFn = (cmd: string, args: ReadonlyArray<string>) => {
      spawnCalls.push({ cmd, args: [...args] });
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
        stdout.emit("data", JSON.stringify({ hits: [], frame: "hints" }));
        proc.emit("close", 0);
      });
      return proc as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    };
    const mockBridge = {
      call: async () => {
        throw new Error("connect ECONNREFUSED /tmp/.daemon.sock");
      },
    } as unknown as PythonCoreBridge;

    const out = (await invokeTool(
      mockBridge,
      "memory_search",
      { query: "HippoDB", k: 4 },
      mockSpawnFn as any,
    )) as Record<string, unknown>;
    assert.equal(spawnCalls.length, 1);
    assert.equal(spawnCalls[0]!.args[0], "search");
    assert.equal(out["_source"], "direct-store");
  });
});
