
import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, rename, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";
import {
  type ConnectTarget,
  createDaemonConnection,
  getDaemonConnectTarget,
  IS_WINDOWS,
} from "./ipc.js";

const execFileAsync = promisify(execFile);

const SCHTASKS_TASK_NAME = "iai-mcp-daemon";


export const HEARTBEAT_REFRESH_INTERVAL_MS = 30_000;

export const HEARTBEAT_SCHEMA_VERSION = 1;

export const WRAPPER_VERSION = "1.0.0";

const LAUNCHCTL_BIN = "/bin/launchctl";

const LAUNCHD_LABEL = "com.iai-mcp.daemon";

const KICKSTART_TIMEOUT_MS = 5_000;


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
  spawnKickstart?: () => Promise<void>;
  refreshIntervalMs?: number;
}

export interface EnsureDaemonAliveOptions {
  waitForReachableMs?: number;
}

export function shouldWakeDefaultDaemonForToolCall(
  platform: NodeJS.Platform = process.platform,
): boolean {
  if (process.env.IAI_MCP_DISABLE_DAEMON_AUTOSTART === "1") {
    return false;
  }
  if (process.env.IAI_DAEMON_SOCKET_PATH) {
    return false;
  }
  return platform === "darwin" || platform === "win32";
}

export class WrapperLifecycle {
  private readonly pid: number;
  private readonly uuid: string;
  private readonly socketPath: string;
  private readonly wakeSignalPath: string;
  private readonly heartbeatPath: string;
  private readonly platform: NodeJS.Platform;
  private readonly socketReachable: () => Promise<boolean>;
  private readonly spawnKickstart: () => Promise<void>;
  private readonly refreshIntervalMs: number;

  private readonly startedAt: string;
  private timer: NodeJS.Timeout | null = null;

  constructor(opts: WrapperLifecycleOptions = {}) {
    this.pid = opts.pid ?? process.pid;
    this.uuid = opts.uuid ?? randomUUID();
    this.socketPath = opts.socketPath ?? defaultSocketPath();
    this.wakeSignalPath = opts.wakeSignalPath ?? defaultWakeSignalPath();
    this.heartbeatPath =
      opts.heartbeatPath ?? defaultHeartbeatPath(this.pid, this.uuid);
    this.platform = opts.platform ?? process.platform;
    this.socketReachable = opts.socketReachable ?? defaultSocketReachable(this.socketPath);
    this.spawnKickstart = opts.spawnKickstart ?? defaultSpawnKickstart();
    this.refreshIntervalMs = opts.refreshIntervalMs ?? HEARTBEAT_REFRESH_INTERVAL_MS;
    this.startedAt = isoNow();
  }

  async ensureDaemonAlive(opts: EnsureDaemonAliveOptions = {}): Promise<void> {
    let alive = false;
    try {
      alive = await this.socketReachable();
    } catch {
      alive = false;
    }
    if (alive) {
      return;
    }
    // macOS: launchctl kickstart. Windows: schtasks /Run the daemon task.
    // Both are best-effort; fall through to the wake-signal sentinel on
    // failure or on Linux (where systemd/scripts own daemon startup).
    if (this.platform === "darwin" || this.platform === "win32") {
      try {
        await this.writeWakeSignal();
      } catch {
      }
      try {
        await this.spawnKickstart();
        await this.waitUntilReachable(opts.waitForReachableMs);
        return;
      } catch {
      }
    }
    try {
      await this.writeWakeSignal();
    } catch {
    }
    await this.waitUntilReachable(opts.waitForReachableMs);
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

  private async waitUntilReachable(timeoutMs?: number): Promise<void> {
    const deadlineMs = timeoutMs ?? 0;
    if (deadlineMs <= 0) {
      return;
    }
    const deadline = Date.now() + deadlineMs;
    while (Date.now() < deadline) {
      try {
        if (await this.socketReachable()) {
          return;
        }
      } catch {
      }
      await sleep(100);
    }
  }
}


function isoNow(): string {
  return new Date().toISOString();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function defaultSocketReachable(socketPath: string): () => Promise<boolean> {
  return async () => {
    // POSIX: probe the (possibly injected) Unix socket path. Windows: probe
    // the TCP loopback endpoint from the daemon port file.
    const target: ConnectTarget | null = IS_WINDOWS
      ? getDaemonConnectTarget()
      : socketPath;
    if (target === null) return false;
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
      const socket = createDaemonConnection(target);
      socket.setTimeout(1_000);
      socket.once("connect", () => settle(true));
      socket.once("error", () => settle(false));
      socket.once("timeout", () => settle(false));
    });
  };
}

function defaultSpawnKickstart(): () => Promise<void> {
  if (IS_WINDOWS) {
    return async () => {
      await execFileAsync("schtasks", ["/Run", "/TN", SCHTASKS_TASK_NAME], {
        timeout: KICKSTART_TIMEOUT_MS,
      });
    };
  }
  return async () => {
    const uid = typeof process.getuid === "function" ? process.getuid() : 0;
    const args = ["kickstart", "-k", `gui/${uid}/${LAUNCHD_LABEL}`];
    await execFileAsync(LAUNCHCTL_BIN, args, {
      timeout: KICKSTART_TIMEOUT_MS,
    });
  };
}
