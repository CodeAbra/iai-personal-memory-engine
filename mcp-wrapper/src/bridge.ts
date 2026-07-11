
import * as crypto from "node:crypto";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";

function getDaemonSocketPath(): string {
  return process.env.IAI_DAEMON_SOCKET_PATH
    ?? path.join(os.homedir(), ".iai-mcp", ".daemon.sock");
}
const SOCKET_CONNECT_TIMEOUT_MS = 5000;
const ERR_DAEMON_UNREACHABLE = -32002;

export class DaemonUnreachableError extends Error {
  public code: number;
  constructor(message: string) {
    super(message);
    this.name = "DaemonUnreachableError";
    this.code = ERR_DAEMON_UNREACHABLE;
  }
}

interface RpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params: Record<string, unknown>;
}

interface RpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string };
}

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
}

export class PythonCoreBridge {
  private sock: net.Socket | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private buffer = "";
  private reconnectAttempted = false;
  private reconnectPromise: Promise<void> | null = null;
  private startPromise: Promise<void> | null = null;
  private parseErrorStreak = 0;
  private static readonly PARSE_ERROR_REJECT_THRESHOLD = 4;

  constructor(
    private readonly pythonCmd: string = process.env.IAI_MCP_PYTHON ?? "python3",
  ) {}

  async start(): Promise<void> {
    if (this.sock) return;
    if (this.startPromise) return this.startPromise;
    this.startPromise = this._doStart();
    try {
      await this.startPromise;
    } catch (err) {
      this.startPromise = null;
      throw err;
    }
  }

  private async _doStart(): Promise<void> {
    this.reconnectAttempted = false;

    let sock: net.Socket;
    try {
      sock = await this.connectWithTimeout(
        getDaemonSocketPath(),
        SOCKET_CONNECT_TIMEOUT_MS,
      );
    } catch (e) {
      throw new DaemonUnreachableError(
        "iai-mcp daemon not running. "
        + "Run: launchctl load -w ~/Library/LaunchAgents/com.iai-mcp.daemon.plist "
        + "or run scripts/install.sh"
      );
    }
    this.sock = sock;
    this.attachSocketHandlers();
  }

  private connectWithTimeout(
    socketPath: string,
    timeoutMs: number,
  ): Promise<net.Socket> {
    return new Promise((resolve, reject) => {
      const sock = net.createConnection(socketPath);
      // Keep a pending/abandoned connect attempt from pinning the event loop
      // (e.g. an in-flight reconnect after socket death). A live connected
      // socket re-refs below so real RPC still holds the process open.
      sock.unref();
      const t = setTimeout(() => {
        try { sock.destroy(); } catch {  }
        reject(new Error("connect_timeout"));
      }, timeoutMs);
      t.unref();
      sock.once("connect", () => {
        clearTimeout(t);
        sock.ref();
        resolve(sock);
      });
      sock.once("error", (e) => {
        clearTimeout(t);
        reject(e);
      });
    });
  }

  private attachSocketHandlers(): void {
    if (!this.sock) return;
    this.sock.on("data", (chunk: Buffer) => this.handleData(chunk));
    this.sock.on("close", () => this.handleSocketDeath("closed"));
    this.sock.on("error", (e: Error) => this.handleSocketDeath(`error: ${e.message}`));
  }

  private handleData(chunk: Buffer): void {
    this.buffer += chunk.toString("utf-8");
    let nl: number;
    while ((nl = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, nl).trim();
      this.buffer = this.buffer.slice(nl + 1);
      if (!line) continue;
      this.handleLine(line);
    }
  }

  private handleLine(line: string): void {
    let msg: RpcResponse;
    try {
      msg = JSON.parse(line) as RpcResponse;
    } catch {
      this.parseErrorStreak += 1;
      if (
        this.parseErrorStreak >= PythonCoreBridge.PARSE_ERROR_REJECT_THRESHOLD
        && this.pending.size > 0
      ) {
        const oldestId = Math.min(...this.pending.keys());
        const handler = this.pending.get(oldestId);
        if (handler) {
          this.pending.delete(oldestId);
          handler.reject(
            new Error(
              `parse_error: ${PythonCoreBridge.PARSE_ERROR_REJECT_THRESHOLD} consecutive non-JSON lines on daemon socket; rejecting stale RPC id=${oldestId}`,
            ),
          );
        }
        try {
          process.stderr.write(
            `${JSON.stringify({
              event: "bridge_ndjson_parse_error_streak",
              threshold: PythonCoreBridge.PARSE_ERROR_REJECT_THRESHOLD,
              rejected_rpc_id: oldestId,
            })}\n`,
          );
        } catch {  }
        this.parseErrorStreak = 0;
      }
      return;
    }
    this.parseErrorStreak = 0;
    const handler = this.pending.get(msg.id);
    if (!handler) return;
    this.pending.delete(msg.id);
    if (msg.error) {
      handler.reject(new Error(msg.error.message));
    } else {
      handler.resolve(msg.result);
    }
  }

  private handleSocketDeath(why: string): void {
    const err = new Error(`daemon_unreachable: socket ${why} (code ${ERR_DAEMON_UNREACHABLE})`);
    for (const [, p] of this.pending) p.reject(err);
    this.pending.clear();
    this.sock = null;
    this.startPromise = null;

    if (this.reconnectAttempted) return;
    this.reconnectAttempted = true;

    this.reconnectPromise = (async () => {
      try {
        const testDelayMs = Number(
          process.env.IAI_MCP_RECONNECT_TEST_DELAY_MS ?? "0",
        );
        if (testDelayMs > 0) {
          await new Promise<void>((r) => setTimeout(r, testDelayMs));
        }
        this.sock = await this.connectWithTimeout(
          getDaemonSocketPath(),
          SOCKET_CONNECT_TIMEOUT_MS,
        );
        this.attachSocketHandlers();
      } catch {
      } finally {
        this.reconnectPromise = null;
      }
    })();
  }

  async call<T = unknown>(
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<T> {
    if (this.reconnectPromise) {
      await this.reconnectPromise;
    }
    if (!this.sock) {
      throw new Error(`daemon_unreachable: bridge not connected (code ${ERR_DAEMON_UNREACHABLE})`);
    }
    const id = this.nextId++;
    const req: RpcRequest = { jsonrpc: "2.0", id, method, params };
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      try {
        this.sock!.write(JSON.stringify(req) + "\n");
      } catch (e) {
        this.pending.delete(id);
        reject(e as Error);
      }
    });
  }

  disconnect(): void {
    // Suppress the reconnect that destroying the socket below would otherwise
    // trigger via the "close" handler — an explicit teardown must not spawn a
    // fresh connect attempt.
    this.reconnectAttempted = true;
    if (this.sock) {
      try { this.sock.end(); } catch {  }
      try { this.sock.destroy(); } catch {  }
      this.sock = null;
    }
    this.startPromise = null;
    for (const [, p] of this.pending) {
      p.reject(new Error("bridge_disconnected"));
    }
    this.pending.clear();
  }

  isConnected(): boolean {
    return this.sock !== null;
  }
}


export function sessionOpenSocketPath(): string {
  const env = process.env.IAI_DAEMON_SOCKET_PATH;
  if (env) return env;
  return path.join(os.homedir(), ".iai-mcp", ".daemon.sock");
}


export function newSessionId(): string {
  return crypto.randomUUID();
}


export function emitSessionOpen(sessionId: string): Promise<void> {
  return new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    try {
      const socketPath = sessionOpenSocketPath();
      const sock = net.createConnection(socketPath, () => {
        const msg =
          JSON.stringify({
            type: "session_open",
            session_id: sessionId,
            ts: new Date().toISOString(),
          }) + "\n";
        sock.write(msg, () => {
          sock.end();
        });
      });
      sock.on("error", () => finish());
      sock.on("close", () => finish());
      sock.setTimeout(2000, () => {
        try {
          sock.destroy();
        } catch {
        }
        finish();
      });
    } catch {
      finish();
    }
  });
}
