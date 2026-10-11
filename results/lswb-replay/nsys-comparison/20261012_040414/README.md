# Profiled replay comparison

Author: Winston Cai.

These three runs used `scripts/local/bench-lswb.sh` for 1200 seconds each, with Nsight start/stop requests at 300/360 and 900/960 seconds. Capture was inactive during model loading and graph creation. `remote-run-record.json` indexes the canonical summaries, measured capture API acknowledgment times, source identity, and remote trace locations; `summary.csv` is derived from the canonical summaries.

All runs missed the 99% success requirement with `missing_output_event` failures. Their cause is unresolved. These are candidate profiled runs, not an acceptance speed comparison. Packed c28 reused the c24 server/cache; Native restarted with the same token pool size. Trace files are saved and nonempty but their contents have not been analyzed.

Canonical small artifacts are in the three timestamped run directories. Raw requests/responses and six large `.nsys-rep` files remain on the VDI at the paths recorded in JSON. Do not substitute this aggregate for those raw artifacts when diagnosing failures.

Reproduction entry point (after launching the chosen mode with `serve.sh <mode> profile` and configuring the replay environment):

```bash
bash scripts/local/bench-lswb.sh <tag> 30212 <concurrency> 1200 300:360,900:960
```

The server was stopped after collection. The GPU lease was renewed to 09:20 Shanghai time on October 12, 2026; that timestamp is a recorded state, not a guarantee of a later renewal.
