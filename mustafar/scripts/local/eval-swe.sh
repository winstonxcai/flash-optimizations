#!/usr/bin/env bash
# =====================================================================
# eval-swe.sh <instance-list> [run-id]
#   SWE-bench_Verified agentic eval (a list of GitHub issue + repo task
#   instances) against an ALREADY-RUNNING local server (serve.sh native|packed).
#   Agents run on the remote YJYBench box and resolve tasks by writing+testing
#   patches. See agentic-eval.sh for the full contract (BASE_URL override, etc.).
#   Results: remote $EVAL_YJY/results/<run-id>/
# =====================================================================
set -u
bash "$(dirname "$0")/agentic-eval.sh" swe "$@"
