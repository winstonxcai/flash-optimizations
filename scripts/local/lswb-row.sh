#!/usr/bin/env bash
# =====================================================================
# lswb-row.sh <run_dir|tag> -- print the 7 recorded SLO-run fields from a
# business_replay summary.json.
# Author: Winston Cai.
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
else
  if [ -d "$ARG" ]; then
    RUN_DIR="$ARG"
  elif [ -d "$BASE/$ARG" ]; then
    RUN_DIR="$BASE/$ARG"
  else
    echo "cannot resolve $ARG"; exit 1
  fi
  # Search across tags only when the caller explicitly selects the results root.
  # File identity also recognizes relative paths, trailing slashes and symlinks.
  if [ "$RUN_DIR" -ef "$BASE" ]; then
    S=$(ls -1dt "$RUN_DIR"/*/20*/client/summary.json 2>/dev/null | head -1)
  else
    S=$(ls -1dt "$RUN_DIR"/20*/client/summary.json 2>/dev/null | head -1)
  fi
  [ -n "$S" ] || { echo "no summary under $RUN_DIR"; exit 1; }
fi

"${PYTHON:-python3}" -c "
import json, sys
summary_path = sys.argv[1]
with open(summary_path) as stream:
    s = json.load(stream)
c = s.get('cache', {})
slo = s.get('slo')
ttft_stats = s['ready_ttft_ms'] if slo else s['ttft_ms']
e2e_stats = s['ready_latency_ms'] if slo else s['latency_ms']
ttft = ttft_stats['p90'] / 1000.0 if ttft_stats['p90'] is not None else float('nan')
e2e = e2e_stats['p90'] / 1000.0 if e2e_stats['p90'] is not None else float('nan')
pass_ = bool(s.get('valid')) and (slo['status'] == 'pass' if slo else ttft < 10.0)
uncached = c.get('prompt_tokens', 0) - c.get('total_cached_tokens', 0)
rate = lambda name: c[name] if c.get(name) is not None else float('nan')
done = s.get('counts', {}).get('successful_in_window' if slo else 'successful', 0)
print('dir=%s' % (summary_path.rsplit('/client',1)[0].rsplit('/',1)[-1]))
if slo:
    print('ttft_basis=ready_to_first_output slo_status=%s threshold_ms=%s censored=%s' % (
        slo['status'], slo['threshold_ms'], slo['right_censored']))
print('slo=%s valid=%s done=%d ttft_p90=%.2fs e2e_p90=%.2fs l1=%.4f l2=%.4f uncached=%d succ=%.4f' % (
    'PASS' if pass_ else 'FAIL', bool(s.get('valid')),
    done, ttft, e2e,
    rate('device_hit_rate'), rate('host_hit_rate'),
    uncached, s.get('success_rate', 0.0)))
" "$S" 2>&1 || { echo "parse failed on $S"; exit 1; }
