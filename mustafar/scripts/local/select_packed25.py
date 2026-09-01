#!/usr/bin/env python3
"""Select 25 NEW Sangfor-Bench tasks for the Stage-1 packed eval.

Even split of cloud-native pass/fail, per-difficulty same as prior 25 run
(easy 4P/3F, medium 5P/5F, hard 4P/4F -> 13P/12F total; 7 easy / 10 medium /
8 hard). Excludes the previous 25 already run on TopMag50.
"""
import json
import sys
from pathlib import Path

CLOUD = Path("/data/zyj/YJYBench/results/test/"
             "Sangfor-Bench_cc_vibe_DeepSeek-V4-Flash-Local_task_20260825_195126_744")
BENCH = Path("/data/zyj/YJYBench/bench/sangforbench")

PREV25 = set("""
aiyycp_sales-flow_d7329e44 fy_gptanalystagent_fb3d6a3d gcjs_go-zero_22ab9e7d
gcjs_kube-log-check-recover_5b6a23ad gcjs_kube-log-check-recover_c6a12bfe
gcjs_kube-log-check-recover_e04abbb7 gcjs_kube-log-check-recover_fc67bfda
mss_drme-service_2a2095f8 sri_ap-gpt_0dd68d23 sri_ap-gpt_2bcf1160
sri_ap-gpt_d7527749 sri_chat-agent_035a16f0 sri_chat-agent_86ce36d3
sri_chat-agent_b2f8ec64 sri_esecgpt_48486b59 sri_esecgpt_80fa3321
sri_esecgpt_cf8ba0fb sri_s1_00ce55e2 sri_s1_cec32c82 sri_s1_d060bef0
sri_swe-bench_5f5a7df7 sri_swe-bench_fea293e6 tw_esecgpt_4966005
tw_esecgpt_6741243f tw_esecgpt_f291630
""".split())

# per-difficulty target: (pass, fail)
TARGET = {"easy": (4, 3), "medium": (5, 5), "hard": (4, 4)}
MIN_TESTS = 8  # skip degenerate/empty test tasks (0/0, 0/1)

rows = []
for d in sorted(CLOUD.iterdir()):
    if not d.is_dir():
        continue
    inst = d.name
    if inst in PREV25:
        continue
    tr = d / "test_result.json"
    if not tr.exists():
        continue
    data = json.loads(tr.read_text())
    bj = BENCH / f"{inst}.json"
    diff = None
    lang = data.get("language")
    if bj.exists():
        b = json.loads(bj.read_text())
        diff = b.get("difficulty")
    passed = data.get("tests", {}).get("passed", 0)
    total = data.get("tests", {}).get("total", 0)
    pr = data.get("pass_rate", 0.0)
    if total < MIN_TESTS:
        continue
    rows.append({
        "inst": inst, "diff": diff, "lang": lang,
        "pass": passed, "total": total, "pass_rate": pr,
        "ok": pr >= 100.0,
    })

print(f"candidates (post prev-25): {len(rows)}", file=sys.stderr)
from collections import Counter
print("by diff:", dict(Counter(r["diff"] for r in rows)), file=sys.stderr)
print("ok/fail:", dict(Counter(r["ok"] for r in rows)), file=sys.stderr)

# Deterministic pick: sort by (diff, inst) and take head for pass and fail buckets.
chosen = []
for diff, (np_, nf) in TARGET.items():
    pool = [r for r in rows if r["diff"] == diff]
    passers = sorted([r for r in pool if r["ok"]], key=lambda r: r["inst"])
    failers = sorted([r for r in pool if not r["ok"]], key=lambda r: r["inst"])
    for r in passers[:np_]:
        chosen.append(r)
    for r in failers[:nf]:
        chosen.append(r)
    # print avail for the record
    print(f"[{diff}] avail pass={len(passers)} fail={len(failers)}", file=sys.stderr)

print(f"chosen: {len(chosen)}", file=sys.stderr)
from collections import Counter
print("chosen by diff:", dict(Counter(r["diff"] for r in chosen)), file=sys.stderr)
print("chosen ok/fail:", dict(Counter(r["ok"] for r in chosen)), file=sys.stderr)

# emit a pipe-delimited table in run order: hard first, then medium, then easy
order = {"hard": 0, "medium": 1, "easy": 2}
for r in sorted(chosen, key=lambda r: (order[r["diff"]], r["inst"])):
    print(f"{r['inst']}|{r['diff']}|{r['lang']}|{r['pass']}/{r['total']}|{'P' if r['ok'] else 'F'}")
