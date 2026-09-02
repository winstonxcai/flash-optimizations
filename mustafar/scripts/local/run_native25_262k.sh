#!/usr/bin/env bash
# NATIVE (pristine, untouched csa) rerun of the 25 NEW packed-set Sangfor tasks at
# --context-length 262144. Two TP4 native servers on 10.72.1.175: A=30211 (GPUs 4-7),
# B=30212 (GPUs 0-3), both pristine stock (git-checkout, no MUSTAFAR patch). Same 25
# tasks as the Stage-1 packed run (dsv4-packed-262k-*_20260901). The two packed-collapse
# tasks run FIRST, one per server: f268ef1 on B, 09299ad2 on A; then the rest.
# RID dsv4-native25-262k-<task>_20260902.
set -u
cd /data/zyj/YJYBench
STAMP=20260902
MASTER=/data/zyj/YJYBench/results/dsv4-native25-262k_master.log
CFG_A=/data/zyj/YJYBench/test_env/docker_env_config_dsv4-windowed.json
CFG_B=/data/zyj/YJYBench/test_env/docker_env_config_dsv4-windowed-30212.json

# --- server A (30211) ---
# The two packed-collapse tasks (f268ef1, 09299ad2) run FIRST, one per server.
A_TASKS="
aiyycp_sales-flow_09299ad2|medium|go|24/24
apex_chat-agent_9347a21|hard|python|56/56
apex_gpt-train-data-collector_1dbcd396|hard|go|109/111
apex_source-tracing-investigation_17ae176f|hard|go|42/42
apex_source-tracing-investigation_ab21ecf0|hard|go|27/28
aiyycp_sales-audit-platform_2439f30d|medium|go|24/24
aiyycp_sales-audit-platform_ebac64e2|medium|go|31/34
aiyycp_sales-auth_daf3ea25|medium|go|24/25
aiyycp_sales-flow_033981bd|medium|go|29/29
aiyycp_sales-flow_191d12be|medium|go|16/37
apex_soar-app_282ef229|easy|python|11/11
apex_soar-app_4b0d01bf|easy|python|7/8
apex_soar-app_8073b35d|easy|python|11/12
apex_soar-app_989a23c5|easy|python|14/14
"
# --- server B (30212) ---
B_TASKS="
apex_chat-agent_f268ef1|hard|python|32/32
apex_soar-app_9207ca23|hard|python|21/23
apex_source-tracing-investigation_a4432711|hard|go|20/20
sri_ap-gpt_0cd2c2ac|hard|python|17/107
aiyycp_sales-audit-platform_53266d85|medium|go|23/23
aiyycp_sales-audit-platform_ef78d2c0|medium|go|14/15
aiyycp_sales-conversation_fa2bb019|medium|go|25/25
aiyycp_sales-flow_edb6ec00|medium|go|23/24
apex_soar-app_4896a623|easy|python|10/10
apex_soar-app_76e6e4f8|easy|python|5/15
apex_soar-app_969ed0d4|easy|python|8/8
"

run_chain() {
  local TAG="$1" CFG="$2" LIST="$3"
  while IFS='|' read -r INST DIFF LANG CLOUD; do
    [ -z "$INST" ] && continue
    [[ "$INST" == \#* ]] && continue
    RID="dsv4-native25-262k-$(echo "$INST" | sed 's/[^a-z0-9-]/_/g')_$STAMP"
    LOG="/data/zyj/YJYBench/results/${RID}_launch.log"
    echo "[$(date '+%H:%M:%S')] [$TAG] START $INST (diff=$DIFF lang=$LANG cloud=$CLOUD) rid=$RID" | tee -a "$MASTER"
    .venv/bin/python -m yjybench.cli \
      --benchmark Sangfor-Bench --mode e2e --max_workers 1 --timeout 18000 \
      --exp_name test --docker_env_config "$CFG" --agent_type cc --agent_mode vibe \
      --sangforbench_prompt_source claude_result-tasks.md \
      --instance_ids "$INST" --run_id "$RID" > "$LOG" 2>&1
    RC=$?
    echo "[$(date '+%H:%M:%S')] [$TAG] DONE $INST rc=$RC" | tee -a "$MASTER"
  done <<< "$LIST"
}

echo "NATIVE25_START $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
run_chain A "$CFG_A" "$A_TASKS" &
PID_A=$!
run_chain B "$CFG_B" "$B_TASKS" &
PID_B=$!
wait "$PID_A" "$PID_B"
echo "ALL_25_DONE $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
