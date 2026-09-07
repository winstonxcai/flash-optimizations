#!/usr/bin/env bash
# =====================================================================
# lswb-row.sh <run_dir|tag> -- print the 7 recorded SLO-run fields from a
# business_replay summary.json.
#
#   run_dir may be a <ts>/ dir directly under results/lswb-replay/<tag>/,
#   a tag dir (newest <ts> used), or results/lswb-replay root (newest tag+ts).
#
# Prints one line:
#   slo=PASS|FAIL valid=<bool> done=<completed> ttft_p90=<s> e2e_p90=<s>
#   l1=<device_hit> l2=<host_hit> uncached=<prompt-total_cached> succ=<rate>
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

ARG=${1:-}
[ -n "$ARG" ] || { echo "usage: $0 <run_dir|tag>"; exit 1; }

BASE="$RESULTS_HOST/lswb-replay"
if [ -f "$ARG/client/summary.json" ]; then
  S="$ARG/client/summary.json"
elif [ -f "$BASE/$ARG/client/summary.json" ]; then
  S="$BASE/$ARG/client/summary.json"
elif [ -d "$BASE/$ARG" ]; then
  S=$(ls -1dt "$BASE/$ARG"/20*/client/summary.json 2>/dev/null | head -1)
  [ -n "$S" ] || { echo "no summary under $BASE/$ARG"; exit 1; }
elif [ -d "$BASE" ]; then
  S=$(ls -1dt "$BASE"/*/20*/client/summary.json 2>/dev/null | head -1)
  [ -n "$S" ] || { echo "no summary under $BASE"; exit 1; }
else
  echo "cannot resolve $ARG"; exit 1
fi

/usr/bin/python3 -c "
import json, sys
s = json.load(open('$S'))
c = s.get('cache', {})
ttft = s['ttft_ms']['p90'] / 1000.0
e2e  = s['latency_ms']['p90'] / 1000.0
pass_ = bool(s.get('valid')) and ttft < 10.0
uncached = c.get('prompt_tokens', 0) - c.get('total_cached_tokens', 0)
print('dir=%s' % ('$S'.rsplit('/client',1)[0].rsplit('/',1)[-1]))
print('slo=%s valid=%s done=%d ttft_p90=%.2fs e2e_p90=%.2fs l1=%.4f l2=%.4f uncached=%d succ=%.4f' % (
    'PASS' if pass_ else 'FAIL', bool(s.get('valid')),
    s.get('counts', {}).get('successful', 0), ttft, e2e,
    c.get('device_hit_rate', 0.0), c.get('host_hit_rate', 0.0),
    uncached, s.get('success_rate', 0.0)))
" 2>&1 || { echo "parse failed on $S"; exit 1; }
