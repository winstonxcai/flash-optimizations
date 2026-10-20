# Business Replay Report

- Valid: `false`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 333 / 327`
- Success rate: `98.1982%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `0.259 req/s`
- Mean / P90 TTFT: `15203.871 / 27836.301 ms`
- Mean / P90 TPOT: `389.392 / 713.268 ms`
- Mean / P90 E2E: `57571.111 / 120799.937 ms`
- GPU / CPU / total KV cache hit rate: `60.3768% / 0.0000% / 60.3768%`
- KV cache detail coverage: `311 / 311`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'fail', 'ready_requests': 341, 'within_threshold': 28, 'violations': 313, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
