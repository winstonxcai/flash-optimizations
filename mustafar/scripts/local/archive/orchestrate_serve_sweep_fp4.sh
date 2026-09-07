#!/usr/bin/env bash
# Chained orchestrator for the fp4 serving-capacity sweep (run in background).
# Waits for the already-running packed-leg driver (PID 2073424) to finish its
# 8 points, frees GPUs 4-7, then boots+runs the native leg (4 points).
# Logs: logs/sweep-native-run.log ; final notify when both legs are done (~4h).
set -u
LOG_BASE=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs
DRIVER=/home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts/local/run_serve_sweep_fp4.sh
PACKED_PID=2073424

echo "[orchestrator] $(date -u +%H:%M:%S)Z waiting for packed driver $PACKED_PID"
while kill -0 "$PACKED_PID" 2>/dev/null; do sleep 60; done
echo "[orchestrator] $(date -u +%H:%M:%S)Z packed driver exited"

# frees GPUs (packed driver already kills its 30211 server at exit; belt-and-braces)
ps -eo pid,args | grep "[s]glang.launch_server" | grep -E -- "--port 3021[12]" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 5

echo "[orchestrator] $(date -u +%H:%M:%S)Z running native leg"
bash "$DRIVER" native > "$LOG_BASE/sweep-native-run.log" 2>&1
RC=$?
echo "[orchestrator] $(date -u +%H:%M:%S)Z native leg done rc=$RC"
ps -eo pid,args | grep "[s]glang.launch_server" | grep -E -- "--port 3021[12]" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
echo "[orchestrator] ALL DONE $(date -u +%H:%M:%S)Z rc=$RC"
exit "$RC"
