#!/usr/bin/env bash
# =====================================================================
# bench-max.sh <ctx>
#   Maximum-concurrency serving comparison: each leg measured at its OWN
#   allocator ceiling for <ctx> (Native at C_nat, Packed at C_pck), so the
#   larger packed pool gets to show what extra concurrency it buys.
#   Thin wrapper over bench-serving.sh max. See that file for the protocol.
#   Results: <RESULTS_HOST>/serving/max-ctx<ctx>-<ts>/
# =====================================================================
set -u
DIR=$(dirname "$0")
bash "$DIR/bench-serving.sh" max "$@"
