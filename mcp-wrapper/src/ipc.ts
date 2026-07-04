
/**
 * Platform-agnostic IPC transport, mirroring the Python `iai_mcp._ipc` module.
 *
 *   POSIX:   Unix-domain socket  ->  ~/.iai-mcp/.daemon.sock
 *   Windows: TCP loopback         ->  127.0.0.1:<port>, port read from
 *                                     ~/.iai-mcp/.daemon.port
 *
 * The base dir is ~/.iai-mcp (os.homedir()) to match `_ipc._BASE_DIR`, which
 * uses Path.home() regardless of IAI_MCP_STORE.
 */
import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";

export const IS_WINDOWS = process.platform === "win32";

export type ConnectTarget = string | { host: string; port: number };

function daemonBaseDir(): string {
  return path.join(os.homedir(), ".iai-mcp");
}

export function daemonSocketPath(): string {
  return path.join(daemonBaseDir(), ".daemon.sock");
}

export function daemonPortFile(): string {
  return path.join(daemonBaseDir(), ".daemon.port");
}

export function readDaemonPort(): number | null {
  try {
    const txt = fs.readFileSync(daemonPortFile(), "utf-8").trim();
    const port = Number.parseInt(txt, 10);
    return Number.isFinite(port) && port > 0 ? port : null;
  } catch {
    return null;
  }
}

/**
 * Windows-only auth token file, mirroring `_ipc.TOKEN_FILE` on the Python
 * side. The daemon requires every TCP client to send this token as the
 * first line on a fresh connection (see `_make_authenticated_handler` /
 * `_send_token_sync` in `_ipc.py`) — a plain connect is not enough.
 * IAI_DAEMON_TOKEN_PATH overrides for tests.
 */
export function daemonTokenPath(): string {
  const env = process.env.IAI_DAEMON_TOKEN_PATH;
  if (env) return env;
  return path.join(daemonBaseDir(), ".daemon.token");
}

export function readDaemonToken(): string | null {
  try {
    const txt = fs.readFileSync(daemonTokenPath(), "utf-8").trim();
    return txt.length > 0 ? txt : null;
  } catch {
    return null;
  }
}

/**
 * Resolve the daemon IPC endpoint.
 *   POSIX   -> Unix-domain socket path (string)
 *   Windows -> { host: "127.0.0.1", port } from the port file
 * Returns null when the endpoint cannot be determined (on Windows: port file
 * absent => daemon not running). IAI_DAEMON_SOCKET_PATH overrides on POSIX.
 */
export function getDaemonConnectTarget(): ConnectTarget | null {
  const env = process.env.IAI_DAEMON_SOCKET_PATH;
  if (env) return env;
  // Test-only escape hatch so the TCP+auth-token path can be exercised
  // without depending on the real process.platform (IS_WINDOWS is a
  // module-load-time constant and can't be swapped mid-test-run).
  const tcpPort = process.env.IAI_DAEMON_TCP_PORT;
  if (tcpPort) {
    const port = Number.parseInt(tcpPort, 10);
    if (Number.isFinite(port) && port > 0) {
      return { host: process.env.IAI_DAEMON_TCP_HOST ?? "127.0.0.1", port };
    }
  }
  if (IS_WINDOWS) {
    const port = readDaemonPort();
    return port === null ? null : { host: "127.0.0.1", port };
  }
  return daemonSocketPath();
}

export function daemonUnreachableHint(): string {
  if (IS_WINDOWS) {
    return (
      "iai-mcp daemon not running. "
      + 'Start it with: schtasks /Run /TN "iai-mcp-daemon" '
      + "(or: iai-mcp daemon install)."
    );
  }
  if (process.platform === "darwin") {
    return (
      "iai-mcp daemon not running. "
      + "Run: launchctl load -w ~/Library/LaunchAgents/com.iai-mcp.daemon.plist "
      + "or run scripts/install.sh"
    );
  }
  return (
    "iai-mcp daemon not running. "
    + "Run: systemctl --user start iai-mcp-daemon or run scripts/install.sh"
  );
}

/**
 * Open a net.Socket to the daemon for either transport. Accepts the union
 * target returned by getDaemonConnectTarget so callers stay platform-agnostic.
 */
export function createDaemonConnection(
  target: ConnectTarget,
  connectListener?: () => void,
): net.Socket {
  if (typeof target === "string") {
    return connectListener
      ? net.createConnection(target, connectListener)
      : net.createConnection(target);
  }
  return connectListener
    ? net.createConnection(target.port, target.host, connectListener)
    : net.createConnection(target.port, target.host);
}

/**
 * Perform the auth-token handshake a TCP connection requires before the
 * daemon will process any RPC traffic on it. Unix-socket targets (POSIX)
 * need no handshake and are a no-op here.
 *
 * Must be called synchronously right after connect, before anything else
 * is written to the socket — the daemon's `_make_authenticated_handler`
 * requires the token to be the very first line, and closes the connection
 * outright (no error response) if it isn't.
 */
export function authenticateConnection(sock: net.Socket, target: ConnectTarget): void {
  if (typeof target === "string") return;
  const token = readDaemonToken();
  if (token === null) {
    throw new Error(
      `Daemon auth token not found: ${daemonTokenPath()} missing. The daemon may need a restart to regenerate it.`,
    );
  }
  sock.write(token + "\n");
}
