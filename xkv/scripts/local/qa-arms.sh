#!/usr/bin/env bash
# =====================================================================
# qa-arms.sh -- boot the same prompts under each xKV arm, to attribute a
#               generation-quality regression to one component.
#
#   qa-arms.sh [run_root]
#
# The free fair/graphab protocol cannot answer "why is the xkv leg's text
# different": it measures a serving point, not a component. This does four
# short boots of the same three greedy prompts, each removing one thing:
#
#   native         stock tree, no xkv code on the path at all.
#   xkv            the full low-rank path (fork tree, store on, triton recon).
#   xkv-nostore    fork tree with the store gated off. Isolates the *patch* --
#                  if this does not match native, the bug is not in compression.
#   xkv-torchrecon fork tree, store on, but the torch reconstruct instead of the
#                  fused triton one. Isolates the triton kernel from the maths.
#
# Each arm's completions land in <run_root>/<arm>/qa.jsonl with its boot log, so
# `qa_compare.py native/xkv.jsonl` can be run between any pair afterwards.
# =====================================================================
set -u
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$DIR/env.sh"

RUN=${1:-$RESULTS_HOST/qa-arms-$(ts)}
mkdir -p "$RUN"

# name|EXTRA_ENVS. The tree is chosen from the part before the first '-', so
# every xkv-* arm serves the patched fork.
ARMS=(
  "native|"
  "xkv|"
  "xkv-nostore|SGLANG_OPT_LOWRANK_KV_STORE=0"
  "xkv-torchrecon|XKV_RECON_TRITON=0"
)

for spec in "${ARMS[@]}"; do
  name=${spec%%|*}; extra=${spec#*|}; leg=${name%%-*}
  echo "== [$name] booting (leg=$leg extra='$extra')"
  mkdir -p "$RUN/$name"
  if ! EXTRA_ENVS="$extra" bash "$DIR/serve.sh" "$leg" > "$RUN/$name/boot.log" 2>&1; then
    echo "   [$name] BOOT FAILED -- see $RUN/$name/boot.log"
    cp "$LOG_HOST/serve_$leg.log" "$RUN/$name/serve.log" 2>/dev/null || true
    bash "$DIR/serve.sh" "$leg" stop >/dev/null 2>&1
    continue
  fi
  cp "$LOG_HOST/serve_$leg.log" "$RUN/$name/serve.log" 2>/dev/null || true
  echo "   [$name] pool=$(pool_of "$RUN/$name/serve.log")"
  echo "pool=$(pool_of "$RUN/$name/serve.log")" > "$RUN/$name/pool.txt"
  ct "python3 $(to_ct "$DIR")/qa_probe.py --port $PORT --tag $name \
      --out $(to_ct "$RUN")/$name/qa.jsonl" > "$RUN/$name/probe.log" 2>&1 \
    || echo "   [$name] probe reported an error -- see $RUN/$name/probe.log"
  bash "$DIR/serve.sh" "$leg" stop >/dev/null 2>&1
done

echo
echo "== native vs each arm (greedy, so agreement should be near-total)"
for spec in "${ARMS[@]}"; do
  name=${spec%%|*}
  [ "$name" = native ] && continue
  [ -f "$RUN/$name/qa.jsonl" ] || continue
  echo "--- native vs $name"
  python3 "$DIR/qa_compare.py" "$RUN/native/qa.jsonl" "$RUN/$name/qa.jsonl" \
    | sed 's/^/    /'
done
echo "== artifacts in $RUN"
