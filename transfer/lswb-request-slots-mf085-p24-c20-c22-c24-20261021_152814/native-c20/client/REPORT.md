# Business Replay Report

- Valid: `false`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 995 / 984`
- Success rate: `98.8945%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `0.803 req/s`
- Mean / P90 TTFT: `7646.767 / 12123.899 ms`
- Mean / P90 TPOT: `117.056 / 285.586 ms`
- Mean / P90 E2E: `23457.632 / 60189.616 ms`
- GPU / CPU / total KV cache hit rate: `90.5383% / 0.0000% / 90.5383%`
- KV cache detail coverage: `964 / 964`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'fail', 'ready_requests': 999, 'within_threshold': 692, 'violations': 307, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
