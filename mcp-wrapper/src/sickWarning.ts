
import { spawn, type SpawnOptions } from "node:child_process";

const DOCTOR_PROBE_TIMEOUT_MS = 5_000;

export async function probeDaemonDoctor(
  spawnFn: typeof spawn = spawn,
  timeoutMs: number = DOCTOR_PROBE_TIMEOUT_MS,
): Promise<number | null> {
  const cli = process.env.IAI_MCP_CLI ?? "iai-mcp";
  const args = ["doctor"];
  return new Promise<number | null>((resolve) => {
    const opts: SpawnOptions = { stdio: ["ignore", "ignore", "ignore"] };
    let proc: ReturnType<typeof spawn>;
    try {
      proc = spawnFn(cli, args, opts);
    } catch {
      resolve(null);
      return;
    }
    const t = setTimeout(() => {
      try {
        proc.kill();
      } catch {
      }
      resolve(null);
    }, timeoutMs);
    proc.on("error", () => {
      clearTimeout(t);
      resolve(null);
    });
    proc.on("close", (code: number | null) => {
      clearTimeout(t);
      resolve(code);
    });
  });
}

export function emitSickWarningIfNeeded(exitCode: number | null): void {
  if (exitCode === null || exitCode === 0) {
    return;
  }
  const msg =
    `iai-mcp warning: daemon doctor reports FAIL (exit=${exitCode}). ` +
    "Run `iai-mcp doctor` for details. Memory tools may fall back to bank-recall.\n";
  try {
    process.stderr.write(msg);
  } catch {
  }
}
