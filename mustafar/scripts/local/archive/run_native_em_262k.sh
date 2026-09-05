#!/usr/bin/env bash
# NATIVE-CSA baseline rerun of the easy+medium Sangfor tasks at --context-length 262144.
# Server: 10.72.1.175:30212 (GPUs 0-3, native SGLANG_OPT_TOPMAG=0). Parallel leg to the
# TopMag50@262144 easy+medium run on 30211/4-7. Same 17 tasks, same harness; only the
# server window differs from the 135168 cratered runs.
# ORDER: mediums that hit the 135k context-400s first (the confounded set), then the
# other mediums, then easy. RID dsv4-native-em-262k-<task>_20260901.
set -u
cd /data/zyj/YJYBench
STAMP=20260901
MASTER=/data/zyj/YJYBench/results/dsv4-native-em-262k_master.log
COMMON="--benchmark Sangfor-Bench --mode e2e --max_workers 1 --timeout 18000 --exp_name test --docker_env_config /data/zyj/YJYBench/test_env/docker_env_config_dsv4-windowed-30212.json --agent_type cc --agent_mode vibe --sangforbench_prompt_source claude_result-tasks.md"

# task  | instance_id                         | diff | lang | cloud-native
TASKS="
# --- medium, 400-hit at 135k (6) ---
gcjs_kube-log-check-recover_5b6a23ad|medium|go|101/254
sri_s1_00ce55e2|medium|python|126/126
aiyycp_sales-flow_d7329e44|medium|go|3/74
sri_chat-agent_86ce36d3|medium|python|0/62
sri_chat-agent_b2f8ec64|medium|python|75/75
sri_s1_d060bef0|medium|python|118/131
# --- other medium (4) ---
sri_esecgpt_cf8ba0fb|medium|go|268/268
fy_gptanalystagent_fb3d6a3d|medium|python|111/111
gcjs_go-zero_22ab9e7d|medium|go|48/48
sri_ap-gpt_0dd68d23|medium|python|119/122
# --- easy (7) ---
gcjs_kube-log-check-recover_c6a12bfe|easy|go|122/122
gcjs_kube-log-check-recover_fc67bfda|easy|go|131/132
tw_esecgpt_4966005|easy|python|40/40
sri_chat-agent_035a16f0|easy|python|25/27
tw_esecgpt_6741243f|easy|python|40/40
gcjs_kube-log-check-recover_e04abbb7|easy|go|73/74
mss_drme-service_2a2095f8|easy|python|35/35
"

echo "EM_START $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
while IFS='|' read -r INST DIFF LANG NATIVE; do
  [ -z "$INST" ] && continue
  [[ "$INST" == \#* ]] && continue
  RID="dsv4-native-em-262k-$(echo "$INST" | sed 's/[^a-z0-9-]/_/g')_$STAMP"
  LOG="/data/zyj/YJYBench/results/${RID}_launch.log"
  echo "[$(date '+%H:%M:%S')] START $INST (diff=$DIFF lang=$LANG cloud=$NATIVE) rid=$RID" | tee -a "$MASTER"
  .venv/bin/python -m yjybench.cli $COMMON --instance_ids "$INST" --run_id "$RID" > "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%H:%M:%S')] DONE $INST rc=$RC" | tee -a "$MASTER"
done <<< "$TASKS"
echo "ALL_17_DONE $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
