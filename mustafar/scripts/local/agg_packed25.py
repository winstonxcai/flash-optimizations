#!/usr/bin/env python3
"""Collect the 25 Stage-1 packed Sangfor results and compare vs cloud-native."""
import json
from pathlib import Path

CLOUD = Path("/data/zyj/YJYBench/results/test/"
             "Sangfor-Bench_cc_vibe_DeepSeek-V4-Flash-Local_task_20260825_195126_744")
PACKED = Path("/data/zyj/YJYBench/results/test")
BENCH = Path("/data/zyj/YJYBench/bench/sangforbench")

INSTS = """apex_chat-agent_9347a21|hard|python
apex_chat-agent_f268ef1|hard|python
apex_gpt-train-data-collector_1dbcd396|hard|go
apex_soar-app_9207ca23|hard|python
apex_source-tracing-investigation_17ae176f|hard|go
apex_source-tracing-investigation_a4432711|hard|go
apex_source-tracing-investigation_ab21ecf0|hard|go
sri_ap-gpt_0cd2c2ac|hard|python
aiyycp_sales-audit-platform_2439f30d|medium|go
aiyycp_sales-audit-platform_53266d85|medium|go
aiyycp_sales-audit-platform_ebac64e2|medium|go
aiyycp_sales-audit-platform_ef78d2c0|medium|go
aiyycp_sales-auth_daf3ea25|medium|go
aiyycp_sales-conversation_fa2bb019|medium|go
aiyycp_sales-flow_033981bd|medium|go
aiyycp_sales-flow_09299ad2|medium|go
aiyycp_sales-flow_191d12be|medium|go
aiyycp_sales-flow_edb6ec00|medium|go
apex_soar-app_282ef229|easy|python
apex_soar-app_4896a623|easy|python
apex_soar-app_4b0d01bf|easy|python
apex_soar-app_76e6e4f8|easy|python
apex_soar-app_8073b35d|easy|python
apex_soar-app_969ed0d4|easy|python
apex_soar-app_989a23c5|easy|python""".splitlines()


def load(path):
    return json.loads(path.read_text())


def summarize(d):
    t = d.get("tests", {})
    return t.get("passed", 0), t.get("total", 0), d.get("pass_rate", 0.0), d.get("build_error", False)


rows = []
for line in INSTS:
    inst, diff, lang = line.split("|")
    cloud_d = load(CLOUD / inst / "test_result.json")
    cpass, ctot, cpr, _ = summarize(cloud_d)
    # find the packed result dir by RID prefix
    cands = [p for p in PACKED.iterdir()
             if p.is_dir() and p.name.startswith(f"Sangfor-Bench_cc_vibe_deepseek-v4-flash_dsv4-packed-262k-{inst}_")]
    if not cands:
        print(f"{inst}: NO PACKED RESULT DIR")
        continue
    pr = cands[0] / inst / "test_result.json"
    if not pr.exists():
        print(f"{inst}: no test_result.json in {cands[0].name}")
        continue
    pd = load(pr)
    ppass, ptot, ppr, pbuild = summarize(pd)
    rows.append({
        "inst": inst, "diff": diff, "lang": lang,
        "cpass": cpass, "ctot": ctot, "cpr": cpr,
        "ppass": ppass, "ptot": ptot, "ppr": ppr, "pbuild": pbuild,
    })

print(f"{'instance':42s} {'diff':6s} {'lang':6s} {'native':>7s} {'packed':>7s} {'Δpp':>6s}")
for r in rows:
    dn = f"{r['cpass']}/{r['ctot']}"
    dn2 = f"{r['ppass']}/{r['ptot']}"
    delta = r["ppr"] - r["cpr"]
    flag = " BUILD" if r["pbuild"] else ""
    print(f"{r['inst']:42s} {r['diff']:6s} {r['lang']:6s} {dn:>7s} {dn2:>7s} {delta:+6.1f}{flag}")

# aggregates
from collections import defaultdict
agg = defaultdict(lambda: {"pp": 0, "pt": 0, "cp": 0, "ct": 0})
for r in rows:
    a = agg["all"]
    a["pp"] += r["ppass"]; a["pt"] += r["ptot"]
    a["cp"] += r["cpass"]; a["ct"] += r["ctot"]
    b = agg[r["diff"]]
    b["pp"] += r["ppass"]; b["pt"] += r["ptot"]
    b["cp"] += r["cpass"]; b["ct"] += r["ctot"]
print("\nweighted means (Σ passed / Σ total):")
for k in ("all", "hard", "medium", "easy"):
    a = agg[k]
    print(f"  {k:6s} packed {100*a['pp']/a['pt']:6.1f}  native {100*a['cp']/a['ct']:6.1f}")
# suite-size-change rows (honesty)
print("\nsuite-size changes (native total != packed total):")
for r in rows:
    if r["ctot"] != r["ptot"]:
        print(f"  {r['inst']}: native {r['ctot']} tests -> packed {r['ptot']} tests")

# concentration analysis: aggregate excluding the two severe regressions
SEVERE = {"apex_chat-agent_f268ef1", "aiyycp_sales-flow_09299ad2"}
print("\nconcentration (excl. f268ef1 + 09299ad2):")
for k in ("all", "hard", "medium", "easy"):
    sub = [r for r in rows if (k == "all" or r["diff"] == k) and r["inst"] not in SEVERE]
    pp = sum(r["ppass"] for r in sub); pt = sum(r["ptot"] for r in sub)
    cp = sum(r["cpass"] for r in sub); ct = sum(r["ctot"] for r in sub)
    print(f"  {k:6s} packed {100*pp/pt:6.1f}  native {100*cp/ct:6.1f}  d={100*pp/pt-100*cp/ct:+6.1f}")

# per-task delta in test cases (same-suite rows only)
print("\ntest-case deltas (same suite, packed-native):")
same = [r for r in rows if r["ctot"] == r["ptot"]]
gain = sum(1 for r in same if r["ppass"] > r["cpass"])
loss = sum(1 for r in same if r["ppass"] < r["cpass"])
flat = sum(1 for r in same if r["ppass"] == r["cpass"])
print(f"  same-suite rows: {len(same)}  +tc {gain}  -tc {loss}  flat {flat}")
for r in sorted(same, key=lambda x: x["ppass"] - x["cpass"]):
    d = r["ppass"] - r["cpass"]
    if d:
        print(f"    {r['inst']:42s} native={r['cpass']}/{r['ctot']} packed={r['ppass']}/{r['ptot']} d={d:+}")
print("\nrow pass-rate deltas: pack-ok+{}, native-ok+{}".format(
    sum(1 for r in rows if r["ppr"] >= 100),
    sum(1 for r in rows if r["cpr"] >= 100)))
