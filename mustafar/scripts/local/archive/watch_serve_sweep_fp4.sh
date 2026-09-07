#!/usr/bin/env bash
# Background monitor for the fp4 serving-capacity sweep.
# Tracks: packed leg (driver 2073424) -> native leg (orchestrator 2081836) -> ALL DONE.
# Logs heartbeats + transitions to logs/sweep-monitor.log; exits 0 on clean
# completion, 1 on FATAL/stall so the launching agent is notified.
set -u
DRIVER=2073424
ORCH=2081836
LOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/sweep-monitor.log
PKLOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/sweep-packed-run.log
ORLOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/sweep-orchestrator.log
NVLOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/sweep-native-run.log
RES=/home/jovyan/winstonxcai/flash-optimizations/mustafar/results/serve-sweep-fp4
MAX_ROUNDS=400   # ~6.5h at 60s
: > "$LOG"

say () { echo "[$(date +%H:%M:%S)Z] $*" | tee -a "$LOG"; }
health () { curl -fsS -m 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1 && echo healthy || echo DOWN; }
npoints () { find "$RES/$1" -name measured.jsonl 2>/dev/null | wc -l; }
fatal_packed () { grep -q "FATAL" "$PKLOG" 2>/dev/null; }
fatal_native () { grep -q "FATAL" "$NVLOG" 2>/dev/null || grep -q "FATAL" "$ORLOG" 2>/dev/null; }

say "monitor start: driver=$DRIVER orchestrator=$ORCH (packed points done so far: $(npoints packed))"
round=0
# ---------- PHASE 1: packed leg ----------
while kill -0 "$DRIVER" 2>/dev/null; do
  round=$((round+1))
  if [ "$round" -gt "$MAX_ROUNDS" ]; then say "FATAL: watchdog timeout in packed phase"; exit 1; fi
  if fatal_packed; then say "FATAL: packed driver reported FATAL"; tail -20 "$PKLOG" | tee -a "$LOG"; exit 1; fi
  H=$(health 30211)
  if [ "$H" = DOWN ]; then
    say "WARN: server 30211 DOWN during packed phase (driver still alive)"; sleep 20
    H2=$(health 30211); [ "$H2" = DOWN ] && say "WARN: 30211 still down after 20s; last driver log:" && tail -5 "$PKLOG" | tee -a "$LOG"
  fi
  sleep 55
done
say "packed driver exited. points done: $(npoints packed). packed log tail:"
tail -4 "$PKLOG" | tee -a "$LOG"
if fatal_packed || ! grep -q "==== \[packed\] done" "$PKLOG" 2>/dev/null; then
  say "WARN: packed leg did not log 'done' (may have died mid-point); continuing to watch native"
fi

# ---------- PHASE 2: native leg ----------
say "waiting for native boot / orchestrator run..."
round=0
while kill -0 "$ORCH" 2>/dev/null; do
  round=$((round+1))
  if [ "$round" -gt "$MAX_ROUNDS" ]; then say "FATAL: watchdog timeout in native phase"; exit 1; fi
  if fatal_native; then say "FATAL: native reported FATAL"; tail -20 "$NVLOG" 2>/dev/null | tee -a "$LOG"; exit 1; fi
  if [ -f "$NVLOG" ] && [ -s "$NVLOG" ]; then
    # only warn once per 10 rounds about health
    if [ $((round % 10)) -eq 0 ]; then
      H=$(health 30212)
      say "native progress: health=$H points=$(npoints native) native-log-lines=$(wc -l < "$NVLOG")"
    fi
  fi
  sleep 55
done
say "orchestrator exited."
tail -6 "$ORLOG" | tee -a "$LOG"
say "FINAL: packed points=$(npoints packed) native points=$(npoints native)"
if grep -q "ALL DONE" "$ORLOG" 2>/dev/null; then say "SWEEP COMPLETE (clean)"; exit 0; fi
say "SWEEP COMPLETE (check orchestrator log for rc)"; exit 0
