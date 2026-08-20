
import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, rename, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);


export const HEARTBEAT_REFRESH_INTERVAL_MS = 30_000;

export const DOCTOR_AUTO_GRACE_MS = 10_000;

// A single short connect probe is NOT evidence of death: a daemon grinding a
// sleep-pipeline step at full CPU can miss a 1 s accept without being down,
// and the old single-probe → kickstart -k path SIGTERMed a healthy busy
// daemon. Death requires every probe in the series to fail AND the recorded
// daemon pid to be gone.
export const SOCKET_PROBE_TIMEOUT_MS = 3_000;

export const SOCKET_PROBE_ATTEMPTS = 3;

export const SOCKET_PROBE_DELAY_MS = 2_000;

export const HEARTBEAT_SCHEMA_VERSION = 1;

export const WRAPPER_VERSION = "3.0.5";

const LAUNCHCTL_BIN = "/bin/launchctl";

const LAUNCHD_LABEL = "com.iai-mcp.daemon";

const KICKSTART_TIMEOUT_MS = 5_000;

const PS_BIN = "/bin/ps";

const PS_TIMEOUT_MS = 2_000;

const DAEMON_TITLE_MARKER = "iai_mcp.daemon";


interface HeartbeatPayload {
  pid: number;
  uuid: string;
  started_at: string;
  last_refresh: string;
  wrapper_version: string;
  schema_version: number;
}


export function defaultSocketPath(): string {
  return join(homedir(), ".iai-mcp", ".daemon.sock");
}

export function defaultWakeSignalPath(): string {
  return join(homedir(), ".iai-mcp", "wake.signal");
}

export function defaultHeartbeatPath(pid: number, uuid: string): string {
  return join(homedir(), ".iai-mcp", "wrappers", `heartbeat-${pid}-${uuid}.json`);
}


export interface WrapperLifecycleOptions {
  pid?: number;
  uuid?: string;
  socketPath?: string;
  wakeSignalPath?: string;
  heartbeatPath?: string;
  platform?: NodeJS.Platform;
  socketReachable?: () => Promise<boolean>;
  daemonPidAlive?: () => Promise<boolean>;
  spawnKickstart?: () => Promise<void>;
  spawnDoctorAuto?: () => Promise<void>;
  refreshIntervalMs?: number;
  doctorAutoGraceMs?: number;
  probeAttempts?: number;
  probeDelayMs?: number;
}

export class WrapperLifecycle {
  private readonly pid: number;
  private readonly uuid: string;
  private readonly socketPath: string;
  private readonly wakeSignalPath: string;
  private readonly heartbeatPath: string;
  private readonly platform: NodeJS.Platform;
  private readonly socketReachable: () => Promise<boolean>;
  private readonly daemonPidAlive: () => Promise<boolean>;
  private readonly spawnKickstart: () => Promise<void>;
  private readonly spawnDoctorAuto: () => Promise<void>;
  private readonly refreshIntervalMs: number;
  private readonly doctorAutoGraceMs: number;
  private readonly probeAttempts: number;
  private readonly probeDelayMs: number;

  private readonly startedAt: string;
  private timer: NodeJS.Timeout | null = null;
  private healTimer: NodeJS.Timeout | null = null;

  constructor(opts: WrapperLifecycleOptions = {}) {
    this.pid = opts.pid ?? process.pid;
    this.uuid = opts.uuid ?? randomUUID();
    this.socketPath = opts.socketPath ?? defaultSocketPath();
    this.wakeSignalPath = opts.wakeSignalPath ?? defaultWakeSignalPath();
    this.heartbeatPath =
      opts.heartbeatPath ?? defaultHeartbeatPath(this.pid, this.uuid);
    this.platform = opts.platform ?? process.platform;
    this.socketReachable = opts.socketReachable ?? defaultSocketReachable(this.socketPath);
    this.daemonPidAlive = opts.daemonPidAlive ?? defaultDaemonPidAlive(this.socketPath);
    this.spawnKickstart = opts.spawnKickstart ?? defaultSpawnKickstart();
    this.spawnDoctorAuto = opts.spawnDoctorAuto ?? defaultSpawnDoctorAuto();
    this.refreshIntervalMs = opts.refreshIntervalMs ?? HEARTBEAT_REFRESH_INTERVAL_MS;
    this.doctorAutoGraceMs = opts.doctorAutoGraceMs ?? DOCTOR_AUTO_GRACE_MS;
    this.probeAttempts = opts.probeAttempts ?? SOCKET_PROBE_ATTEMPTS;
    this.probeDelayMs = opts.probeDelayMs ?? SOCKET_PROBE_DELAY_MS;
    this.startedAt = isoNow();
  }

  scheduleAutoHeal(): void {
    // Reflex arc: if the daemon is still unreachable after the wake
    // attempt has had its grace window, hand off to `doctor --auto` —
    // the safe unattended heal subset, damped on the doctor side.
    if (this.healTimer !== null) {
      clearTimeout(this.healTimer);
    }
    const timer = setTimeout(() => {
      void (async () => {
        if (await this.probeAliveWithRetries()) {
          return;
        }
        try {
          await this.spawnDoctorAuto();
        } catch {
        }
      })();
    }, this.doctorAutoGraceMs);
    timer.unref();
    this.healTimer = timer;
  }

  async ensureDaemonAlive(): Promise<void> {
    if (await this.probeAliveWithRetries()) {
      return;
    }
    // Signal first, then kickstart: the wake signal is what lets the FSM
    // leave persisted HIBERNATION, and a daemon kickstarted before the file
    // lands would boot, find nothing, and exit after one tick. Kickstart
    // alone never wakes the FSM.
    try {
      await this.writeWakeSignal();
    } catch {
    }
    // A recorded pid that is still alive means the daemon exists and is
    // busy or booting, never dead: kickstarting it would tear down a live
    // process (launchd's kickstart terminates the running instance first).
    // The wake signal above is the whole intervention in that case.
    let pidAlive = false;
    try {
      pidAlive = await this.daemonPidAlive();
    } catch {
      pidAlive = false;
    }
    if (pidAlive) {
      return;
    }
    if (this.platform === "darwin") {
      try {
        await this.spawnKickstart();
      } catch {
      }
    }
  }

  private async probeAliveWithRetries(): Promise<boolean> {
    const attempts = this.probeAttempts > 0 ? this.probeAttempts : 1;
    for (let i = 0; i < attempts; i += 1) {
      let alive = false;
      try {
        alive = await this.socketReachable();
      } catch {
        alive = false;
      }
      if (alive) {
        return true;
      }
      if (i + 1 < attempts && this.probeDelayMs > 0) {
        await sleep(this.probeDelayMs);
      }
    }
    return false;
  }

  async registerHeartbeat(): Promise<void> {
    await this.writeHeartbeat();
    if (this.timer !== null) {
      clearInterval(this.timer);
    }
    const timer = setInterval(() => {
      void this.writeHeartbeat().catch(() => {
      });
    }, this.refreshIntervalMs);
    timer.unref();
    this.timer = timer;
  }

  async cleanupHeartbeat(): Promise<void> {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.healTimer !== null) {
      clearTimeout(this.healTimer);
      this.healTimer = null;
    }
    try {
      await unlink(this.heartbeatPath);
    } catch {
    }
  }


  private async writeHeartbeat(): Promise<void> {
    const payload: HeartbeatPayload = {
      pid: this.pid,
      uuid: this.uuid,
      started_at: this.startedAt,
      last_refresh: isoNow(),
      wrapper_version: WRAPPER_VERSION,
      schema_version: HEARTBEAT_SCHEMA_VERSION,
    };
    const dir = dirname(this.heartbeatPath);
    await mkdir(dir, { recursive: true });
    const tmp = `${this.heartbeatPath}.${this.uuid}.tmp`;
    await writeFile(tmp, JSON.stringify(payload), { encoding: "utf-8" });
    await rename(tmp, this.heartbeatPath);
  }

  private async writeWakeSignal(): Promise<void> {
    const dir = dirname(this.wakeSignalPath);
    await mkdir(dir, { recursive: true });
    const payload = JSON.stringify({
      requested_at: isoNow(),
      wrapper_pid: this.pid,
      wrapper_uuid: this.uuid,
    });
    const tmp = `${this.wakeSignalPath}.${this.uuid}.tmp`;
    await writeFile(tmp, payload, { encoding: "utf-8" });
    await rename(tmp, this.wakeSignalPath);
  }
}


function isoNow(): string {
  return new Date().toISOString();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    t.unref?.();
  });
}

function defaultSocketReachable(socketPath: string): () => Promise<boolean> {
  return async () => {
    const { createConnection } = await import("node:net");
    return await new Promise<boolean>((resolve) => {
      let settled = false;
      const settle = (v: boolean): void => {
        if (settled) return;
        settled = true;
        try {
          socket.destroy();
        } catch {
        }
        resolve(v);
      };
      const socket = createConnection({ path: socketPath });
      socket.setTimeout(SOCKET_PROBE_TIMEOUT_MS);
      socket.once("connect", () => settle(true));
      socket.once("error", () => settle(false));
      socket.once("timeout", () => settle(false));
    });
  };
}

function defaultDaemonPidAlive(socketPath: string): () => Promise<boolean> {
  // Identity-verified liveness: the recorded pid must still name a daemon
  // process of THIS store. A bare kill(pid, 0) would accept a recycled pid
  // and suppress the wake forever; matching the process title cannot.
  return async () => {
    const { readFile, realpath } = await import("node:fs/promises");
    const storeDir = dirname(socketPath);
    const statePath = join(storeDir, ".daemon-state.json");
    let pid = 0;
    try {
      const raw = await readFile(statePath, { encoding: "utf-8" });
      const parsed = JSON.parse(raw) as { daemon_pid?: unknown };
      pid = typeof parsed.daemon_pid === "number" ? parsed.daemon_pid : 0;
    } catch {
      return false;
    }
    if (!Number.isInteger(pid) || pid <= 0) {
      return false;
    }
    try {
      const { stdout } = await execFileAsync(PS_BIN, ["-o", "command=", "-p", String(pid)], {
        timeout: PS_TIMEOUT_MS,
      });
      // The daemon embeds its store RESOLVED, so compare resolved paths —
      // otherwise a symlinked HOME makes the strict branch silently inert.
      let resolvedStore = storeDir;
      try {
        resolvedStore = await realpath(storeDir);
      } catch {
        resolvedStore = storeDir;
      }
      return titleIsDaemonOfStore(stdout, resolvedStore);
    } catch {
      return false;
    }
  };
}

// Deliberately lenient where evidence is missing and strict where it is
// present: a title carrying `store=` must name THIS store (so another
// store's daemon on a recycled pid cannot suppress the wake), but a title
// without it still counts — the daemon's setproctitle call is best-effort,
// and treating a title-less daemon as dead would reintroduce the kill this
// whole path exists to prevent.
export function titleIsDaemonOfStore(psOutput: string, storeRoot: string): boolean {
  if (!psOutput.includes(DAEMON_TITLE_MARKER)) {
    return false;
  }
  const marker = "store=";
  const at = psOutput.indexOf(marker);
  if (at < 0) {
    return true;
  }
  const claimed = psOutput.slice(at + marker.length).trim();
  return claimed === storeRoot || claimed.startsWith(`${storeRoot} `);
}

// No `-k`: that flag TERMINATES the running instance first, and a wrapper
// never has the evidence to justify a kill — by the time this runs the
// caller has established the daemon is unreachable AND its pid is gone.
export function kickstartArgs(uid: number): string[] {
  return ["kickstart", `gui/${uid}/${LAUNCHD_LABEL}`];
}

function defaultSpawnKickstart(): () => Promise<void> {
  return async () => {
    const uid = typeof process.getuid === "function" ? process.getuid() : 0;
    await execFileAsync(LAUNCHCTL_BIN, kickstartArgs(uid), {
      timeout: KICKSTART_TIMEOUT_MS,
    });
  };
}

function defaultSpawnDoctorAuto(): () => Promise<void> {
  return async () => {
    const { spawn } = await import("node:child_process");
    // Fire-and-forget: the reflex must never hold the MCP session open or
    // surface heal noise into it. `iai-mcp` resolves from PATH; a missing
    // binary is swallowed by the caller's catch.
    const child = spawn("iai-mcp", ["doctor", "--auto"], {
      detached: true,
      stdio: "ignore",
    });
    child.on("error", () => {
    });
    child.unref();
  };
}
