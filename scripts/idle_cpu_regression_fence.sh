#!/usr/bin/env bash
# scripts/idle_cpu_regression_fence.sh — idle-CPU regression fence.
#
# Contract: 30-min `python -m iai_mcp.daemon` run with first_turn_pending = 1
# shows process CPU < 5% sampled every 30s.
#
# Usage:
#   scripts/idle_cpu_regression_fence.sh        # 30-min run, samples every 30s
#   IAI_FENCE_DURATION_MIN=5 scripts/idle_cpu_regression_fence.sh  # short run
#
# Pre-condition: daemon must already be running. The script does NOT spawn
# the daemon and does NOT manage launchd — D7.2-26 + D7.2-27 keep daemon
# lifecycle entirely under user discretion. To start the daemon manually
# before running this fence, run:
#
#   python -m iai_mcp.daemon &
#
# (development / manual subprocess path; foreground or background). The
# fence treats the daemon as a black box and only reads its self-CPU% via
# psutil, so any startup mechanism that yields a `iai_mcp.daemon` process
# will work.
#
# Exit codes:
#   0 — all samples < THRESHOLD_PCT sustained
#   1 — at least one sample >= THRESHOLD_PCT
#   2 — daemon not running / pgrep returned 0 matches
#   3 — psutil / Python error
set -eu

DURATION_MIN="${IAI_FENCE_DURATION_MIN:-30}"
SAMPLE_INTERVAL_SEC="${IAI_FENCE_SAMPLE_INTERVAL_SEC:-30}"
THRESHOLD_PCT="${IAI_FENCE_THRESHOLD_PCT:-5.0}"

# Locate the daemon PID. We use pgrep -f for the explicit module form.
DAEMON_PID=$(pgrep -f "iai_mcp.daemon" | head -1 || true)
if [ -z "$DAEMON_PID" ]; then
  echo "ERROR: no iai_mcp.daemon process found." >&2
  echo "Start it manually before running this fence:" >&2
  echo "  python -m iai_mcp.daemon &" >&2
  exit 2
fi

echo "idle-CPU regression fence"
echo "  daemon PID:        $DAEMON_PID"
echo "  duration:          ${DURATION_MIN}min"
echo "  sample interval:   ${SAMPLE_INTERVAL_SEC}s"
echo "  threshold:         ${THRESHOLD_PCT}%"
echo

SAMPLES_TAKEN=0
OVER_THRESHOLD=0
MAX_SEEN=0
DURATION_SEC=$((DURATION_MIN * 60))
START_TS=$(date +%s)

while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TS))
  if [ $ELAPSED -ge $DURATION_SEC ]; then
    break
  fi

  # Use python+psutil for cross-platform self-CPU% read.
  CPU=$(python3 -c "
import psutil, sys
try:
    p = psutil.Process($DAEMON_PID)
    p.cpu_percent(interval=None)
    import time
    time.sleep(1.0)
    print(p.cpu_percent(interval=None))
except Exception as e:
    sys.stderr.write(f'psutil error: {e}\n')
    sys.exit(3)
")
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "  sample fail: psutil error" >&2
    exit 3
  fi

  SAMPLES_TAKEN=$((SAMPLES_TAKEN + 1))
  printf "  t=%4ds  cpu=%5.1f%%\n" "$ELAPSED" "$CPU"

  # awk float comparison (bash doesn't do floats natively).
  OVER=$(awk -v cpu="$CPU" -v thr="$THRESHOLD_PCT" 'BEGIN { print (cpu > thr) ? 1 : 0 }')
  if [ "$OVER" = "1" ]; then
    OVER_THRESHOLD=$((OVER_THRESHOLD + 1))
  fi

  MAX_SEEN=$(awk -v cur="$CPU" -v prev="$MAX_SEEN" 'BEGIN { print (cur > prev) ? cur : prev }')

  sleep "$SAMPLE_INTERVAL_SEC"
done

echo
echo "Summary:"
echo "  total samples:      $SAMPLES_TAKEN"
echo "  over threshold:     $OVER_THRESHOLD"
echo "  max observed CPU%:  $MAX_SEEN"

if [ $OVER_THRESHOLD -gt 0 ]; then
  echo "FAIL: $OVER_THRESHOLD/$SAMPLES_TAKEN samples exceeded ${THRESHOLD_PCT}%."
  exit 1
else
  echo "PASS: all samples under threshold."
  exit 0
fi
