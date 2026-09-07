#!/usr/bin/env bash
# NATIVE-CSA control rerun of the 8 HARD Sangfor tasks (local server, no TopMag).
# Purpose: separate TopMag50's hard-bucket regression from cloud-vs-local / n=1 noise.
# Native baseline for each = cloud run task_20260825_195126_744 (the same native
# numbers the TopMag25 run compared against).
set -u
cd /data/zyj/YJYBench
STAMP=20260830
MASTER=/data/zyj/YJYBench/results/dsv4-hardnative_master.log
COMMON="--benchmark Sangfor-Bench --mode e2e --max_workers 1 --timeout 18000 --exp_name test --docker_env_config /data/zyj/YJYBench/test_env/docker_env_config_dsv4-windowed.json --agent_type cc --agent_mode vibe --sangforbench_prompt_source claude_result-tasks.md"

# task  | instance_id                         | diff | lang | cloud-native
TASKS="
sri_swe-bench_5f5a7df7|hard|python|116/116
sri_esecgpt_48486b59|hard|go|75/227
sri_esecgpt_80fa3321|hard|go|267/267
sri_s1_cec32c82|hard|python|76/176
sri_swe-bench_fea293e6|hard|python|86/86
sri_ap-gpt_2bcf1160|hard|python|16/164
tw_esecgpt_f291630|hard|go|243/243
sri_ap-gpt_d7527749|hard|python|1/137
"

echo "HARD_START $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
while IFS='|' read -r INST DIFF LANG NATIVE; do
  [ -z "$INST" ] && continue
  RID="dsv4-hardnative-$(echo "$INST" | sed 's/[^a-z0-9-]/_/g')_$STAMP"
  LOG="/data/zyj/YJYBench/results/${RID}_launch.log"
  echo "[$(date '+%H:%M:%S')] START $INST (diff=$DIFF lang=$LANG cloud=$NATIVE) rid=$RID" | tee -a "$MASTER"
  .venv/bin/python -m yjybench.cli $COMMON --instance_ids "$INST" --run_id "$RID" > "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%H:%M:%S')] DONE $INST rc=$RC" | tee -a "$MASTER"
done <<< "$TASKS"
echo "ALL_8_DONE $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
