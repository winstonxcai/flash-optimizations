#!/usr/bin/env bash
# =====================================================================
# run-all.sh -- the whole xKV GPU leg series as ONE unattended job.
#
# Submit through the GPU scheduler, never bare -- the queue is what keeps this
# from colliding with another tenant's job (which is exactly what happened when
# the calibration was launched outside it):
#
#   gpuq run --project xkv --gpus 4 --gpu-ids 0,1,2,3 \
#     --cwd /home/jovyan/winstonxcai/flash-optimizations \
#     --expected-duration 6h --workload-key xkv-serving --detach \
#     -- bash xkv/scripts/local/run-all.sh
#
# The scheduler exports GPUQ_ALLOCATED_GPUS and CUDA_VISIBLE_DEVICES for the
# granted device set; this script reads them so it never assumes 0-3. Run by
# hand, it falls back to $GPUS, then to the env.sh default.
#
# Steps, in order. A failing step stops the series (later steps depend on
# earlier ones), and the failure plus everything already collected stays under
# $RUN_ROOT:
#   0 prep      apply + verify the xkv patch on the fork tree
#   1 selftest  triton recon vs the torch reference (no server needed)
#   2 basis     refit the rank-192 basis from live prefill traffic
#   3 qa        boot each leg: capture proof (xkv decode graphs) + fixed-prompt QA
#   4 fair      64k / C=32 native vs xkv, report protocol
#   5 graphab   64k / C=32 xkv, decode graphs full vs disabled
#
# Env overrides: CTX (65536), C (32), NUM_PROMPTS/FIT_LEN (basis calibration),
# SKIP (space-separated step names to skip), RUN_ROOT.
# =====================================================================
set -u
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$DIR/env.sh"

CTX=${CTX:-65536}
C=${C:-32}
FIT_PROMPTS=${NUM_PROMPTS:-16}
FIT_LEN=${FIT_LEN:-$CTX}
SKIP=${SKIP:-}
# Prefer the scheduler's output dir when it gave us one, so the artifacts and the
# queue's own record of the job point at the same place.
RUN_ROOT=${RUN_ROOT:-${GPUQ_OUTPUT_DIR:-$RESULTS_HOST/runall-$(ts)}}
RESULT="$RUN_ROOT/RESULT.txt"

# --------------------------- GPU allocation ---------------------------
# Prefer what the scheduler granted. CUDA_VISIBLE_DEVICES is the same list (the
# container runs --gpus all, so host ids == container ids); GPUQ_ALLOCATED_GPUS
# is the scheduler's own record of it and wins if the two ever disagree.
ALLOC=${GPUQ_ALLOCATED_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
if [ -n "$ALLOC" ]; then
  GPUS=$ALLOC
  export GPUS
fi
export CUDA_VISIBLE_DEVICES=$GPUS   # children (docker, bench clients) see only these

mkdir -p "$RUN_ROOT"

say () { echo "$@" | tee -a "$RESULT"; }
step_on () { case " $SKIP " in *" $1 "*) return 1 ;; esac; return 0; }

say "== xkv run-all $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "   gpus=$GPUS tp=$TP port=$PORT ctx=$CTX C=$C"
say "   job=${GPUQ_JOB_ID:-<none>} project=${GPUQ_PROJECT:-<none>} run_root=$RUN_ROOT"

# --------------------------- 0. patch prep ----------------------------
if step_on prep; then
  say ""
  say "== [0/5] prep: apply + verify the xkv patch (fork tree, v0.5.18 anchors)"
  # `patch` is idempotent and re-runs the read-only anchor drift census against
  # the pristine tree first, so a drifted anchor fails here rather than inside a
  # bench leg an hour later.
  if bash "$DIR/container.sh" patch >> "$RUN_ROOT/prep.log" 2>&1; then
    say "   patch OK"
  else
    say "   patch FAILED -- see $RUN_ROOT/prep.log"; tail -30 "$RUN_ROOT/prep.log"; exit 1
  fi
fi

# --------------------------- 1. selftest ------------------------------
if step_on selftest; then
  say ""
  say "== [1/5] selftest: triton reconstruction vs the torch reference"
  if ct "cd $REPO_CT && export CUDA_VISIBLE_DEVICES=$GPUS \
          && python3 -m xkv selftest" > "$RUN_ROOT/selftest.log" 2>&1; then
    say "   selftest OK"; grep -aE "^(ok|FAIL|pass|fail)" "$RUN_ROOT/selftest.log" | sed 's/^/   /' | head -20
  else
    say "   selftest FAILED -- see $RUN_ROOT/selftest.log"; tail -30 "$RUN_ROOT/selftest.log"; exit 1
  fi
fi

# --------------------------- 2. basis refit ---------------------------
if step_on basis; then
  say ""
  say "== [2/5] basis: refit rank-192 from $FIT_PROMPTS x $FIT_LEN-token prefills"
  if bash "$DIR/calibrate.sh" "$FIT_PROMPTS" "$FIT_LEN" > "$RUN_ROOT/calibrate.log" 2>&1; then
    say "   calibrated: $(ls -1 "$BASIS_HOST" 2>/dev/null | wc -l) basis file(s) in $BASIS_HOST"
  else
    say "   calibration FAILED -- see $RUN_ROOT/calibrate.log"; tail -30 "$RUN_ROOT/calibrate.log"; exit 1
  fi
  if ! ls "$BASIS_HOST"/A_*.pt >/dev/null 2>&1; then
    say "   no A_*.pt in $BASIS_HOST -- the store would silently skip every layer"; exit 1
  fi
fi

# --------------------- 3. capture proof + output QA -------------------
# One boot per leg. The xkv boot is the capture proof: with decode graphs
# requested as `full`, capture failure raises during model-runner init and no
# server comes up, so a healthy boot IS the proof -- plus the explicit negative
# grep below, in case a tree ever downgrades that to a logged fallback.
boot_qa () {  # $1=leg
  local leg=$1 dir="$RUN_ROOT/qa" rc=0
  mkdir -p "$dir"
  say "   [$leg] booting for capture proof + QA"
  bash "$DIR/serve.sh" "$leg" > "$dir/boot_$leg.log" 2>&1 || {
    say "   [$leg] SERVER FAILED TO BOOT -- see $dir/boot_$leg.log"
    return 1; }
  local slog="$LOG_HOST/serve_$leg.log"
  say "   [$leg] pool=$(pool_of "$slog")"
  mkdir -p "$dir/$leg" && echo "pool=$(pool_of "$slog")" > "$dir/$leg/pool.txt"

  case "$leg" in
    xkv)
      if grep -qaE "Capture cuda graph failed|CUDA graph capture failed" "$slog"; then
        say "   [xkv] DECODE GRAPHS DID NOT CAPTURE:"; grep -a "Capture cuda graph failed" "$slog" | head -3 | sed 's/^/     /'
        rc=1
      else
        say "   [xkv] decode graphs captured (backend=full, no capture failure in boot log)"
      fi
      ;;
  esac

  ct "python3 $(to_ct "$DIR")/qa_probe.py --port $PORT --tag $leg \
      --out $(to_ct "$dir")/$leg/qa.jsonl" > "$dir/$leg/qa.log" 2>&1 \
    || { say "   [$leg] QA probe errored -- see $dir/$leg/qa.log"; rc=1; }
  bash "$DIR/serve.sh" "$leg" stop >/dev/null 2>&1
  return $rc
}

if step_on qa; then
  say ""
  say "== [3/5] capture proof + fixed-prompt QA (one boot per leg)"
  qa_rc=0
  boot_qa native || qa_rc=1
  boot_qa xkv    || qa_rc=1
  if [ -f "$RUN_ROOT/qa/native/qa.jsonl" ] && [ -f "$RUN_ROOT/qa/xkv/qa.jsonl" ]; then
    say "   --- native vs xkv output QA ---"
    python3 "$DIR/qa_compare.py" "$RUN_ROOT/qa/native/qa.jsonl" "$RUN_ROOT/qa/xkv/qa.jsonl" \
      2>&1 | sed 's/^/   /' | tee -a "$RESULT"
  fi
  # QA is diagnostic: record the finding, but a shaky QA is not a reason to
  # throw away the serving numbers that follow. The report reads both.
  [ $qa_rc -eq 0 ] || say "   (QA/capture reported problems -- recorded, series continues)"
fi

# --------------------------- 4. fair bench ----------------------------
if step_on fair; then
  say ""
  say "== [4/5] fair: native vs xkv at ctx=$CTX C=$C"
  if CTX="$CTX" bash "$DIR/bench-serving.sh" fair "$CTX" "$C" > "$RUN_ROOT/fair.log" 2>&1; then
    grep -aE "^  (native|xkv)" "$RUN_ROOT/fair.log" | sed 's/^/   /' | tee -a "$RESULT"
  else
    say "   fair bench FAILED -- see $RUN_ROOT/fair.log"; tail -30 "$RUN_ROOT/fair.log" | sed 's/^/   /'
  fi
  grep -a "artifacts at" "$RUN_ROOT/fair.log" | sed 's/^/   /' >> "$RESULT"
fi

# -------------------------- 5. graph A/B ------------------------------
if step_on graphab; then
  say ""
  say "== [5/5] graphab: xkv decode graphs full vs disabled at ctx=$CTX C=$C"
  if CTX="$CTX" bash "$DIR/bench-serving.sh" graphab "$CTX" "$C" > "$RUN_ROOT/graphab.log" 2>&1; then
    grep -aE "^  xkv" "$RUN_ROOT/graphab.log" | sed 's/^/   /' | tee -a "$RESULT"
  else
    say "   graphab FAILED -- see $RUN_ROOT/graphab.log"; tail -30 "$RUN_ROOT/graphab.log" | sed 's/^/   /'
  fi
  grep -a "artifacts at" "$RUN_ROOT/graphab.log" | sed 's/^/   /' >> "$RESULT"
fi

say ""
say "== done $(date -u +%Y-%m-%dT%H:%M:%SZ); artifacts under $RUN_ROOT"
