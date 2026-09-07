#!/usr/bin/env bash
# =====================================================================
# bench-fair.sh <ctx> [C_fair]
#   Fair-concurrency serving comparison: Native vs Packed at the SAME
#   concurrency C, where C defaults to Native's allocator ceiling for <ctx>
#   (pass an explicit C_fair to compare at an arbitrary concurrency).
#   Thin wrapper over bench-serving.sh fair. See that file for the protocol.
#   Results: <RESULTS_HOST>/serving/fair-ctx<ctx>-<ts>/
# =====================================================================
set -u
DIR=$(dirname "$0")
bash "$DIR/bench-serving.sh" fair "$@"
