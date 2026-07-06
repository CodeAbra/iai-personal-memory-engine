
import { describe, it, afterEach } from "node:test";
import { strict as assert } from "node:assert";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as net from "node:net";

import { PythonCoreBridge, DaemonUnreachableError } from "../src/bridge.js";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function makeTmpDir(): Promise<string> {
  return await mkdtemp(join(tmpdir(), "iai-bridge-test-"));
}

class MockDaemon {
  private server: net.Server;
  private connections: net.Socket[] = [];
  readonly socketPath: string;
  public requestHandler: (req: { id: number; method: string; params: unknown }) => unknown;

  constructor(dir: string) {
    this.socketPath = join(dir, ".daemon.sock");
    this.requestHandler = (req) => ({ ok: true, method: req.method });
    this.server = net.createServer((conn) => {
      this.connections.push(conn);
      let buf = "";
      conn.on("data", (chunk) => {
        buf += chunk.toString("utf-8");
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          try {
            const req = JSON.parse(line);
            const result = this.requestHandler(req);
            const resp = JSON.stringify({ jsonrpc: "2.0", id: req.id, result });
            conn.write(resp + "\n");
          } catch (e) {
            const req = JSON.parse(line);
            conn.write(
              JSON.stringify({
                jsonrpc: "2.0",
                id: req.id,
                error: { code: -32603, message: String(e) },
              }) + "\n",
            );
          }
        }
      });
    });
  }

  async listen(): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(this.socketPath, () => resolve());
    });
  }

  async close(): Promise<void> {
    for (const c of this.connections) {
      try { c.destroy(); } catch {  }
    }
    this.connections = [];
    return new Promise((resolve) => {
      this.server.close(() => resolve());
    });
  }

  disconnectAll(): void {
    for (const c of this.connections) {
      try { c.destroy(); } catch {  }
    }
    this.connections = [];
  }
}

function makeBridge(socketPath: string): PythonCoreBridge {
  process.env.IAI_DAEMON_SOCKET_PATH = socketPath;
  return new PythonCoreBridge();
}

let tmpDirs: string[] = [];
let daemons: MockDaemon[] = [];
let bridges: PythonCoreBridge[] = [];

afterEach(async () => {
  for (const b of bridges) { try { b.disconnect(); } catch {  } }
  bridges = [];
  for (const d of daemons) { try { await d.close(); } catch {  } }
  daemons = [];
  for (const dir of tmpDirs) { await rm(dir, { recursive: true, force: true }); }
  tmpDirs = [];
  delete process.env.IAI_DAEMON_SOCKET_PATH;
  delete process.env.IAI_MCP_RECONNECT_TEST_DELAY_MS;
  delete process.env.IAI_DAEMON_TCP_PORT;
  delete process.env.IAI_DAEMON_TCP_HOST;
  delete process.env.IAI_DAEMON_TOKEN_PATH;
});

async function setup(): Promise<{ daemon: MockDaemon; bridge: PythonCoreBridge }> {
  const dir = await makeTmpDir();
  tmpDirs.push(dir);
  const daemon = new MockDaemon(dir);
  daemons.push(daemon);
  await daemon.listen();
  const bridge = makeBridge(daemon.socketPath);
  bridges.push(bridge);
  return { daemon, bridge };
}


describe("PythonCoreBridge: basic connectivity", () => {
  it("connects to daemon socket and reports isConnected", async () => {
    const { bridge } = await setup();
    assert.equal(bridge.isConnected(), false);
    await bridge.start();
    assert.equal(bridge.isConnected(), true);
  });

  it("start() is idempotent", async () => {
    const { bridge } = await setup();
    await bridge.start();
    await bridge.start();
    assert.equal(bridge.isConnected(), true);
  });

  it("throws DaemonUnreachableError on missing socket", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const bridge = makeBridge(join(dir, "nonexistent.sock"));
    bridges.push(bridge);
    await assert.rejects(
      () => bridge.start(),
      (err: Error) => err instanceof DaemonUnreachableError,
    );
  });

  it("disconnect clears state", async () => {
    const { bridge } = await setup();
    await bridge.start();
    bridge.disconnect();
    assert.equal(bridge.isConnected(), false);
  });
});


describe("PythonCoreBridge: JSON-RPC calls", () => {
  it("sends request and receives response", async () => {
    const { daemon, bridge } = await setup();
    daemon.requestHandler = (req) => ({ echo: req.method, args: req.params });
    await bridge.start();
    const res = await bridge.call<{ echo: string; args: unknown }>("test_method", { key: "value" });
    assert.equal(res.echo, "test_method");
    assert.deepEqual(res.args, { key: "value" });
  });

  it("rejects on daemon error response", async () => {
    const { daemon, bridge } = await setup();
    daemon.requestHandler = () => { throw new Error("intentional_error"); };
    await bridge.start();
    await assert.rejects(
      () => bridge.call("failing_method"),
      (err: Error) => err.message.includes("intentional_error"),
    );
  });

  it("rejects with daemon_unreachable when not connected", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const bridge = makeBridge(join(dir, "none.sock"));
    bridges.push(bridge);
    await assert.rejects(
      () => bridge.call("anything"),
      (err: Error) => err.message.includes("daemon_unreachable"),
    );
  });
});


describe("PythonCoreBridge: concurrent requests", () => {
  it("handles 10 concurrent requests correctly", async () => {
    const { daemon, bridge } = await setup();
    daemon.requestHandler = (req) => ({ id: req.id, method: req.method });
    await bridge.start();

    const promises = Array.from({ length: 10 }, (_, i) =>
      bridge.call<{ id: number; method: string }>(`method_${i}`, { idx: i }),
    );
    const results = await Promise.all(promises);

    for (let i = 0; i < 10; i++) {
      assert.equal(results[i].method, `method_${i}`);
    }
  });

  it("handles slow responses without blocking fast ones", async () => {
    const { daemon, bridge } = await setup();
    const server = (daemon as unknown as { server: net.Server }).server;

    daemon.requestHandler = (req) => ({ method: req.method });
    await bridge.start();

    const fast = bridge.call<{ method: string }>("fast");
    const slow = bridge.call<{ method: string }>("slow");

    const results = await Promise.all([fast, slow]);
    assert.equal(results[0].method, "fast");
    assert.equal(results[1].method, "slow");
  });
});


describe("PythonCoreBridge: handleSocketDeath", () => {
  it("rejects pending requests on socket close", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const socketPath = join(dir, ".daemon.sock");
    const connections: net.Socket[] = [];

    const server = net.createServer((conn) => {
      connections.push(conn);
      conn.on("data", () => {  });
    });
    await new Promise<void>((r) => server.listen(socketPath, r));

    const bridge = makeBridge(socketPath);
    bridges.push(bridge);
    try {
      await bridge.start();

      const pending = bridge.call("hanging_method");
      // Attach the rejection expectation BEFORE the socket dies below, so the
      // synchronous reject in handleSocketDeath is always observed and never
      // escapes as an unhandledRejection during the gap before the assertion.
      const rejection = assert.rejects(
        () => pending,
        (err: Error) => err.message.includes("daemon_unreachable"),
      );
      await sleep(50);

      for (const c of connections) c.destroy();
      await sleep(50);

      await rejection;
    } finally {
      // The socket death above triggers one reconnect to the still-listening
      // server; disconnect the bridge and drop every server-side connection
      // before close() so it is not left waiting on a live handle.
      bridge.disconnect();
      for (const c of connections) { try { c.destroy(); } catch {  } }
      await new Promise<void>((r) => server.close(r));
    }
  });

  it("reconnects once after socket death", async () => {
    const { daemon, bridge } = await setup();
    process.env.IAI_MCP_RECONNECT_TEST_DELAY_MS = "100";
    daemon.requestHandler = (req) => ({ ok: true, method: req.method });
    await bridge.start();

    daemon.disconnectAll();
    await sleep(200);

    const res = await bridge.call<{ ok: boolean }>("after_reconnect");
    assert.equal(res.ok, true);
  });

  it("stays degraded after second death (no infinite reconnect loop)", async () => {
    const { daemon, bridge } = await setup();
    process.env.IAI_MCP_RECONNECT_TEST_DELAY_MS = "50";
    daemon.requestHandler = (req) => ({ ok: true });
    await bridge.start();

    daemon.disconnectAll();
    await sleep(100);
    assert.equal(bridge.isConnected(), true);

    daemon.disconnectAll();
    await sleep(100);
    assert.equal(bridge.isConnected(), false);
  });
});


describe("PythonCoreBridge: framing errors", () => {
  it("tolerates non-JSON lines from daemon (e.g. stray prints)", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const socketPath = join(dir, ".daemon.sock");

    const server = net.createServer((conn) => {
      let buf = "";
      conn.on("data", (chunk) => {
        buf += chunk.toString();
        const nl = buf.indexOf("\n");
        if (nl >= 0) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          const req = JSON.parse(line);
          conn.write("WARNING: something unexpected\n");
          conn.write("DEBUG: another line\n");
          conn.write(JSON.stringify({ jsonrpc: "2.0", id: req.id, result: { ok: true } }) + "\n");
        }
      });
    });
    await new Promise<void>((r) => server.listen(socketPath, r));

    const bridge = makeBridge(socketPath);
    bridges.push(bridge);
    await bridge.start();

    const res = await bridge.call<{ ok: boolean }>("test");
    assert.equal(res.ok, true);

    bridge.disconnect();
    await new Promise<void>((r) => server.close(r));
  });

  it("rejects after 4 consecutive parse errors (threshold)", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const socketPath = join(dir, ".daemon.sock");

    const server = net.createServer((conn) => {
      let buf = "";
      conn.on("data", (chunk) => {
        buf += chunk.toString();
        const nl = buf.indexOf("\n");
        if (nl >= 0) {
          buf = buf.slice(nl + 1);
          for (let i = 0; i < 5; i++) {
            conn.write(`GARBAGE LINE ${i}\n`);
          }
        }
      });
    });
    await new Promise<void>((r) => server.listen(socketPath, r));

    const bridge = makeBridge(socketPath);
    bridges.push(bridge);
    await bridge.start();

    await assert.rejects(
      () => bridge.call("doomed"),
      (err: Error) => err.message.includes("parse_error"),
    );

    bridge.disconnect();
    await new Promise<void>((r) => server.close(r));
  });

  it("handles partial JSON lines (chunked delivery)", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const socketPath = join(dir, ".daemon.sock");

    const server = net.createServer((conn) => {
      let buf = "";
      conn.on("data", (chunk) => {
        buf += chunk.toString();
        const nl = buf.indexOf("\n");
        if (nl >= 0) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          const req = JSON.parse(line);
          const resp = JSON.stringify({ jsonrpc: "2.0", id: req.id, result: { chunked: true } });
          const full = resp + "\n";
          const chunk1 = full.slice(0, 10);
          const chunk2 = full.slice(10, 30);
          const chunk3 = full.slice(30);
          conn.write(chunk1);
          setTimeout(() => conn.write(chunk2), 10);
          setTimeout(() => conn.write(chunk3), 20);
        }
      });
    });
    await new Promise<void>((r) => server.listen(socketPath, r));

    const bridge = makeBridge(socketPath);
    bridges.push(bridge);
    await bridge.start();

    const res = await bridge.call<{ chunked: boolean }>("chunked_test");
    assert.equal(res.chunked, true);

    bridge.disconnect();
    await new Promise<void>((r) => server.close(r));
  });
});


describe("PythonCoreBridge: rapid connection flaps", () => {
  it("survives rapid connect/disconnect cycles", async () => {
    const { daemon, bridge } = await setup();
    daemon.requestHandler = (req) => ({ cycle: req.method });

    for (let i = 0; i < 5; i++) {
      await bridge.start();
      assert.equal(bridge.isConnected(), true);
      bridge.disconnect();
      assert.equal(bridge.isConnected(), false);
    }
  });

  it("concurrent start() calls resolve to same connection", async () => {
    const { bridge } = await setup();

    const [r1, r2, r3] = await Promise.allSettled([
      bridge.start(),
      bridge.start(),
      bridge.start(),
    ]);

    assert.equal(r1.status, "fulfilled");
    assert.equal(r2.status, "fulfilled");
    assert.equal(r3.status, "fulfilled");
    assert.equal(bridge.isConnected(), true);
  });
});


// Simulates the Windows daemon's `_make_authenticated_handler`: the first
// line on every connection must equal `expectedToken`, or the connection is
// closed immediately with nothing written back — mirroring _ipc.py exactly,
// including the "silently closed, no error frame" behavior that produced the
// `daemon_unreachable: socket closed` symptom this test guards against.
class MockAuthDaemon {
  private server: net.Server;
  private connections: net.Socket[] = [];
  private readonly expectedToken: string;
  public requestHandler: (req: { id: number; method: string; params: unknown }) => unknown;

  constructor(expectedToken: string) {
    this.expectedToken = expectedToken;
    this.requestHandler = (req) => ({ ok: true, method: req.method });
    this.server = net.createServer((conn) => {
      this.connections.push(conn);
      let authenticated = false;
      let buf = "";
      conn.on("data", (chunk) => {
        buf += chunk.toString("utf-8");
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!authenticated) {
            authenticated = true;
            if (line !== this.expectedToken) {
              conn.destroy();
              return;
            }
            continue;
          }
          if (!line) continue;
          try {
            const req = JSON.parse(line);
            const result = this.requestHandler(req);
            conn.write(JSON.stringify({ jsonrpc: "2.0", id: req.id, result }) + "\n");
          } catch (e) {
            const req = JSON.parse(line);
            conn.write(
              JSON.stringify({ jsonrpc: "2.0", id: req.id, error: { code: -32603, message: String(e) } }) + "\n",
            );
          }
        }
      });
    });
  }

  async listen(): Promise<number> {
    return new Promise((resolve) => {
      this.server.listen(0, "127.0.0.1", () => {
        resolve((this.server.address() as net.AddressInfo).port);
      });
    });
  }

  async close(): Promise<void> {
    for (const c of this.connections) { try { c.destroy(); } catch {  } }
    this.connections = [];
    return new Promise((resolve) => { this.server.close(() => resolve()); });
  }
}

describe("PythonCoreBridge: TCP auth-token handshake (Windows transport)", () => {
  it("authenticates with the token before any RPC traffic and completes calls normally", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const tokenPath = join(dir, ".daemon.token");
    const token = "test-token-abc123";
    await writeFile(tokenPath, token, "utf-8");

    const daemon = new MockAuthDaemon(token);
    const port = await daemon.listen();

    process.env.IAI_DAEMON_TCP_PORT = String(port);
    process.env.IAI_DAEMON_TOKEN_PATH = tokenPath;
    const bridge = new PythonCoreBridge();
    bridges.push(bridge);

    await bridge.start();
    assert.equal(bridge.isConnected(), true);

    const res = await bridge.call<{ echo: string }>("ping");
    assert.equal(res.method, "ping");

    await daemon.close();
  });

  it("fails as daemon_unreachable when the token file is missing", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const missingTokenPath = join(dir, ".daemon.token"); // never written

    const daemon = new MockAuthDaemon("irrelevant-since-no-token-sent");
    const port = await daemon.listen();

    process.env.IAI_DAEMON_TCP_PORT = String(port);
    process.env.IAI_DAEMON_TOKEN_PATH = missingTokenPath;
    const bridge = new PythonCoreBridge();
    bridges.push(bridge);

    await assert.rejects(
      () => bridge.start(),
      (err: Error) => err instanceof DaemonUnreachableError,
    );
    assert.equal(bridge.isConnected(), false);

    await daemon.close();
  });

  it("gets disconnected by the daemon when the token is wrong (pre-fix regression case)", async () => {
    const dir = await makeTmpDir();
    tmpDirs.push(dir);
    const tokenPath = join(dir, ".daemon.token");
    await writeFile(tokenPath, "wrong-token", "utf-8");

    const daemon = new MockAuthDaemon("actual-expected-token");
    const port = await daemon.listen();

    process.env.IAI_DAEMON_TCP_PORT = String(port);
    process.env.IAI_DAEMON_TOKEN_PATH = tokenPath;
    const bridge = new PythonCoreBridge();
    bridges.push(bridge);

    // The mismatched-token socket is accepted at the TCP level and only
    // closed after the daemon reads the (wrong) first line, so start()
    // itself resolves; the failure shows up as the daemon severing the
    // connection right after, exactly like the bug this test targets.
    await bridge.start();
    await assert.rejects(
      () => bridge.call("ping"),
      (err: Error) => err.message.includes("daemon_unreachable"),
    );

    await daemon.close();
  });
});
