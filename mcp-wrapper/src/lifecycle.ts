
import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, rename, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);


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

  async ensureDaemonAlive(): Promise<void> {
    let alive = false;
    try {
      alive = await this.socketReachable();
    } catch {
      alive = false;
    }
    if (alive) {
      return;
    }
    if (this.platform === "darwin") {
      try {
        await this.spawnKickstart();
        return;
      } catch {
      }
    }
    try {
      await this.writeWakeSignal();
    } catch {
    }
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
}


function isoNow(): string {
  return new Date().toISOString();
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
      socket.setTimeout(1_000);
      socket.once("connect", () => settle(true));
      socket.once("error", () => settle(false));
      socket.once("timeout", () => settle(false));
    });
  };
}

function defaultSpawnKickstart(): () => Promise<void> {
  return async () => {
    const uid = typeof process.getuid === "function" ? process.getuid() : 0;
    const args = ["kickstart", "-k", `gui/${uid}/${LAUNCHD_LABEL}`];
    await execFileAsync(LAUNCHCTL_BIN, args, {
      timeout: KICKSTART_TIMEOUT_MS,
    });
  };
}
