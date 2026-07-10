#!/usr/bin/env python3
from __future__ import annotations


import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="arousal_budget A/B ship-gate harness",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 137])
    parser.add_argument(
        "--scale", default="honest",
        choices=["smoke", "mvp", "honest", "stress"],
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Ship-gate threshold (delta_rescue >= threshold -> keep)",
    )
    parser.add_argument(
        "--store-dir", default=None,
        help=(
            "Bench-isolated IAI_MCP_STORE path passed through to inner bench. "
            "Default lets the inner bench use its DEFAULT_STORE_DIR "
            "(/tmp/iai-mcp-bench-claude/store)."
        ),
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worktree_root = Path(__file__).resolve().parent.parent

    env = os.environ.copy()
    env.pop("IAI_MCP_AROUSAL_USE_SHADOW", None)
    cmd = [
        sys.executable,
        str(worktree_root / "bench" / "contradiction_longitudinal_claude.py"),
        "--output-dir", str(args.output_dir),
        "--scale", args.scale,
        "--seeds", *(str(s) for s in args.seeds),
    ]
    if args.store_dir:
        cmd += ["--store-dir", args.store_dir]
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        return rc

    return subprocess.run(
        [
            sys.executable,
            str(worktree_root / "bench" / "analyze_arousal_ab.py"),
            str(args.output_dir),
            "--threshold", str(args.threshold),
        ],
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
