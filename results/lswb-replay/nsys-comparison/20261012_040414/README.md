# Profiled replay comparison

Author: Winston Cai.

These five runs used `scripts/local/bench-lswb.sh` for 1200 seconds each, with Nsight start/stop requests at 300/360 and 900/960 seconds. Capture was inactive during model loading and graph creation. `remote-run-record.json` indexes the canonical summaries, measured capture API acknowledgment times, source identity, and remote trace locations; `summary.csv` is derived from the canonical summaries.

All runs missed the 99% success requirement with `missing_output_event` failures. Their cause is unresolved. These are candidate profiled runs, not an acceptance speed comparison. For both Packed Triton and Packed Kernel, c28 reused that implementation’s c24 server/cache; Native restarted with the same token pool size. Trace files are saved and nonempty but their contents have not been analyzed.

Canonical small artifacts are in the five timestamped run directories. Raw requests/responses and ten large `.nsys-rep` files remain on the VDI at the paths recorded in JSON. Do not substitute this aggregate for those raw artifacts when diagnosing failures.

Reproduction entry point (after launching the chosen mode with `serve.sh <mode> profile` and configuring the replay environment):

```bash
bash scripts/local/bench-lswb.sh <tag> 30212 <concurrency> 1200 300:360,900:960
```

The server was stopped after collection. The GPU lease was renewed to 12:08 Shanghai time on October 12, 2026; that timestamp is a recorded state, not a guarantee of a later renewal.

The Kernel runs used parent source `34dc2e227e4747e4f235157919b8a14ede0a9fdf`, the same fork pins and Docker image as the earlier runs, and GPUs 0–3. Its reports 1/2 belong to c24 and 3/4 to c28. All four were verified nonempty after collection. Capture acknowledgment delay is recorded; trace boundaries remain to be inspected. The existing serving and replay scripts handled the launch and windows without code changes.

Diagnosis after comparison with `native-slo-c24/20260909_081547`: all five recent launches used decode graphs capped at batch 15, so c24/c28 decode fell back to eager. These runs are misconfigured evidence and cannot establish implementation speed. Original Native c24 measured 277.98 output tokens/s and 22.32 ms median TPOT with 99.19% success. Dataset and runner hashes match; the adapter hash changed and remains a comparison limitation. The correction defaults standalone serving to extended graphs and checks resolved server graph coverage before replay. Validation repeats Native c24 unprofiled for 1200s using the existing entry point, then stops the server.
