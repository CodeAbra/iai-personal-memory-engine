
import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { mkdtemp, readFile, readdir, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  WrapperLifecycle,
  kickstartArgs,
  titleIsDaemonOfStore,
} from "../src/lifecycle.js";

async function makeTmp(prefix: string): Promise<string> {
  return await mkdtemp(join(tmpdir(), `iai-mcp-lifecycle-${prefix}-`));
}

async function cleanupTmp(dir: string): Promise<void> {
  await rm(dir, { recursive: true, force: true });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}


describe("WrapperLifecycle.ensureDaemonAlive", () => {
  it("does NOT invoke subprocess when socket is reachable", async () => {
    const tmp = await makeTmp("alive");
    try {
      let kickstarts = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {
          kickstarts += 1;
        },
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(kickstarts, 0, "kickstart must not be invoked when socket is alive");
      await assert.rejects(stat(join(tmp, "wake.signal")));
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("invokes launchctl kickstart on darwin when socket is unreachable", async () => {
    const tmp = await makeTmp("kickstart");
    try {
      let kickstarts = 0;
      let signalExistedAtKickstart = false;
      const wakeSignalPath = join(tmp, "wake.signal");
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath,
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => false,
        probeAttempts: 1,
        probeDelayMs: 0,
        spawnKickstart: async () => {
          kickstarts += 1;
          try {
            signalExistedAtKickstart = (await stat(wakeSignalPath)).isFile();
          } catch {
            signalExistedAtKickstart = false;
          }
        },
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(kickstarts, 1, "kickstart must be invoked exactly once on darwin");
      const sigStat = await stat(wakeSignalPath);
      assert.ok(
        sigStat.isFile(),
        "wake.signal must be written on darwin — kickstart alone never wakes the FSM out of HIBERNATION",
      );
      assert.equal(
        signalExistedAtKickstart,
        true,
        "wake.signal must land BEFORE kickstart so the booting daemon finds it",
      );
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("still writes wake.signal when kickstart fails on darwin", async () => {
    const tmp = await makeTmp("fallback");
    try {
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => false,
        probeAttempts: 1,
        probeDelayMs: 0,
        spawnKickstart: async () => {
          throw new Error("kickstart simulated failure");
        },
      });
      await lifecycle.ensureDaemonAlive();
      const sigStat = await stat(join(tmp, "wake.signal"));
      assert.ok(sigStat.isFile(), "wake.signal must exist after kickstart failure");
      const raw = await readFile(join(tmp, "wake.signal"), "utf-8");
      const parsed = JSON.parse(raw);
      assert.ok(typeof parsed.requested_at === "string");
      assert.ok(typeof parsed.wrapper_pid === "number");
      assert.ok(typeof parsed.wrapper_uuid === "string");
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("on non-macos writes wake.signal and never spawns subprocess", async () => {
    const tmp = await makeTmp("linux");
    try {
      let kickstarts = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "linux",
        socketReachable: async () => false,
        probeAttempts: 1,
        probeDelayMs: 0,
        spawnKickstart: async () => {
          kickstarts += 1;
        },
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(kickstarts, 0, "subprocess must never be invoked on non-darwin");
      const sigStat = await stat(join(tmp, "wake.signal"));
      assert.ok(sigStat.isFile(), "wake.signal must exist on non-darwin path");
    } finally {
      await cleanupTmp(tmp);
    }
  });
});


describe("WrapperLifecycle.scheduleAutoHeal", () => {
  it("spawns doctor --auto when the socket is still dead after the grace window", async () => {
    const tmp = await makeTmp("heal-dead");
    try {
      let doctorSpawns = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => false,
        probeAttempts: 1,
        probeDelayMs: 0,
        spawnKickstart: async () => {},
        spawnDoctorAuto: async () => {
          doctorSpawns += 1;
        },
        doctorAutoGraceMs: 20,
      });
      lifecycle.scheduleAutoHeal();
      await sleep(80);
      assert.equal(doctorSpawns, 1, "doctor --auto must be spawned once when daemon stays dead");
      await lifecycle.cleanupHeartbeat();
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("does NOT spawn doctor --auto when the socket recovered within the grace window", async () => {
    const tmp = await makeTmp("heal-alive");
    try {
      let doctorSpawns = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {},
        spawnDoctorAuto: async () => {
          doctorSpawns += 1;
        },
        doctorAutoGraceMs: 20,
      });
      lifecycle.scheduleAutoHeal();
      await sleep(80);
      assert.equal(doctorSpawns, 0, "a reachable daemon must not trigger the heal reflex");
      await lifecycle.cleanupHeartbeat();
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("cleanupHeartbeat cancels a pending heal timer", async () => {
    const tmp = await makeTmp("heal-cancel");
    try {
      let doctorSpawns = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => false,
        probeAttempts: 1,
        probeDelayMs: 0,
        spawnKickstart: async () => {},
        spawnDoctorAuto: async () => {
          doctorSpawns += 1;
        },
        doctorAutoGraceMs: 30,
      });
      lifecycle.scheduleAutoHeal();
      await lifecycle.cleanupHeartbeat();
      await sleep(80);
      assert.equal(doctorSpawns, 0, "a cancelled heal timer must not fire");
    } finally {
      await cleanupTmp(tmp);
    }
  });
});


describe("WrapperLifecycle.registerHeartbeat", () => {
  it("creates heartbeat file with correct schema", async () => {
    const tmp = await makeTmp("hb-schema");
    try {
      const heartbeatPath = join(tmp, "wrappers", "heartbeat-12345-abc.json");
      const lifecycle = new WrapperLifecycle({
        pid: 12345,
        uuid: "abc",
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath,
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {},
        refreshIntervalMs: 60_000,
      });
      await lifecycle.registerHeartbeat();
      try {
        const raw = await readFile(heartbeatPath, "utf-8");
        const parsed = JSON.parse(raw);
        assert.equal(parsed.pid, 12345);
        assert.equal(parsed.uuid, "abc");
        assert.ok(typeof parsed.started_at === "string");
        assert.ok(typeof parsed.last_refresh === "string");
        assert.ok(typeof parsed.wrapper_version === "string");
        assert.equal(parsed.schema_version, 1);
      } finally {
        await lifecycle.cleanupHeartbeat();
      }
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("refresh timer updates last_refresh", async () => {
    const tmp = await makeTmp("hb-refresh");
    try {
      const heartbeatPath = join(tmp, "wrappers", "heartbeat-1-x.json");
      const lifecycle = new WrapperLifecycle({
        pid: 1,
        uuid: "x",
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath,
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {},
        refreshIntervalMs: 10,
      });
      await lifecycle.registerHeartbeat();
      try {
        const before = JSON.parse(await readFile(heartbeatPath, "utf-8"));
        await sleep(60);
        const after = JSON.parse(await readFile(heartbeatPath, "utf-8"));
        assert.equal(before.started_at, after.started_at);
        assert.notEqual(before.last_refresh, after.last_refresh);
      } finally {
        await lifecycle.cleanupHeartbeat();
      }
    } finally {
      await cleanupTmp(tmp);
    }
  });
});


describe("WrapperLifecycle.cleanupHeartbeat", () => {
  it("deletes heartbeat file and clears timer", async () => {
    const tmp = await makeTmp("cleanup");
    try {
      const heartbeatPath = join(tmp, "wrappers", "heartbeat-1-x.json");
      const lifecycle = new WrapperLifecycle({
        pid: 1,
        uuid: "x",
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath,
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {},
        refreshIntervalMs: 10,
      });
      await lifecycle.registerHeartbeat();
      const sigBefore = await stat(heartbeatPath);
      assert.ok(sigBefore.isFile());

      await lifecycle.cleanupHeartbeat();
      await assert.rejects(stat(heartbeatPath), "heartbeat file must be gone after cleanup");

      await sleep(60);
      await assert.rejects(stat(heartbeatPath), "no refresh tick after cleanup");

      await lifecycle.cleanupHeartbeat();
    } finally {
      await cleanupTmp(tmp);
    }
  });
});


describe("WrapperLifecycle security invariants", () => {
  it("source contains no shell-true option and no shell-interpreting subprocess variants", async () => {
    const here = fileURLToPath(new URL(".", import.meta.url));
    const srcDir = join(here, "..", "src");
    const files = await readdir(srcDir);
    const tsFiles = files.filter((f) => f.endsWith(".ts"));
    assert.ok(tsFiles.length > 0, "expected at least one .ts file in src/");

    const E = String.fromCharCode(0x65);
    const X = String.fromCharCode(0x78);
    const C = String.fromCharCode(0x63);
    const SHELL_INTERP_TOKEN = E + X + E + C;
    const SHELL_OPTION_TOKEN = "shell";
    const shellOptionRegex = new RegExp(
      `\\b${SHELL_OPTION_TOKEN}\\s*:\\s*true\\b`,
    );
    const bareCallRegex = new RegExp(
      `(?:^|[^A-Za-z0-9_])${SHELL_INTERP_TOKEN}\\s*\\(`,
    );
    const dottedCallRegex = new RegExp(
      `\\bchild_process\\s*\\.\\s*${SHELL_INTERP_TOKEN}\\s*\\(`,
    );

    const forbidden: { file: string; pattern: string; line: number }[] = [];
    for (const f of tsFiles) {
      const path = join(srcDir, f);
      const content = await readFile(path, "utf-8");
      const lines = content.split("\n");
      lines.forEach((line, idx) => {
        const trimmed = line.trim();
        const codePortion = (trimmed.split("//")[0] ?? "").trim();
        if (codePortion.length === 0) {
          return;
        }
        if (shellOptionRegex.test(codePortion)) {
          forbidden.push({
            file: f,
            pattern: "shell-true option",
            line: idx + 1,
          });
        }
        if (dottedCallRegex.test(codePortion)) {
          forbidden.push({
            file: f,
            pattern: "child_process.<shell-interp-call>",
            line: idx + 1,
          });
        }
        if (bareCallRegex.test(codePortion)) {
          forbidden.push({
            file: f,
            pattern: "bare <shell-interp-call>",
            line: idx + 1,
          });
        }
      });
    }

    assert.deepEqual(
      forbidden,
      [],
      `Forbidden subprocess pattern in mcp-wrapper/src/: ${JSON.stringify(forbidden, null, 2)}`,
    );
  });
});

describe("WrapperLifecycle death evidence", () => {
  it("retries the probe before declaring the daemon dead", async () => {
    const tmp = await makeTmp("retry");
    try {
      let probes = 0;
      let kickstarts = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        // A daemon grinding a consolidation step at full CPU misses the
        // first accept and answers the next one; one miss is not death.
        socketReachable: async () => {
          probes += 1;
          return probes > 1;
        },
        daemonPidAlive: async () => false,
        spawnKickstart: async () => {
          kickstarts += 1;
        },
        probeAttempts: 3,
        probeDelayMs: 0,
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(probes, 2, "the probe must be retried after a single miss");
      assert.equal(kickstarts, 0, "a busy daemon must never be kickstarted");
      await assert.rejects(
        stat(join(tmp, "wake.signal")),
        "no wake signal is needed for a daemon that answered",
      );
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("never kickstarts while the recorded daemon pid is alive", async () => {
    const tmp = await makeTmp("pidalive");
    try {
      let kickstarts = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => false,
        daemonPidAlive: async () => true,
        spawnKickstart: async () => {
          kickstarts += 1;
        },
        probeAttempts: 2,
        probeDelayMs: 0,
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(
        kickstarts,
        0,
        "an unreachable but LIVE daemon is busy or booting, never dead",
      );
      const sigStat = await stat(join(tmp, "wake.signal"));
      assert.ok(sigStat.isFile(), "the wake signal is the whole intervention");
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("kickstarts only when both the socket and the pid are gone", async () => {
    const tmp = await makeTmp("bothgone");
    try {
      let kickstarts = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => false,
        daemonPidAlive: async () => false,
        spawnKickstart: async () => {
          kickstarts += 1;
        },
        probeAttempts: 2,
        probeDelayMs: 0,
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(kickstarts, 1);
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("kickstart argv carries no -k: a wake must never terminate", () => {
    const args = kickstartArgs(501);
    assert.deepEqual(args, ["kickstart", "gui/501/com.iai-mcp.daemon"]);
    assert.ok(!args.includes("-k"), "-k SIGTERMs the running instance");
  });

  it("pid identity: strict when the title names a store, lenient without", () => {
    const store = "/Users/alice/.iai-mcp";
    assert.equal(
      titleIsDaemonOfStore(`iai lilli (iai_mcp.daemon) store=${store}\n`, store),
      true,
    );
    assert.equal(
      titleIsDaemonOfStore(
        `iai lilli (iai_mcp.daemon) store=/tmp/other-store\n`,
        store,
      ),
      false,
      "another store's daemon on a recycled pid must not suppress the wake",
    );
    assert.equal(
      titleIsDaemonOfStore("python -m iai_mcp.daemon\n", store),
      true,
      "a daemon whose setproctitle failed must still count as alive",
    );
    assert.equal(
      titleIsDaemonOfStore("/usr/bin/some-other-process\n", store),
      false,
    );
  });
});
