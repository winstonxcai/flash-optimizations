Author: Winston Cai.

Question: does additional Packed capacity preserve shared prefixes and improve throughput?
Matrix: Native / Packed Triton / Packed Kernel; G16/G32/G40; C16.
Each cell: 1280 requests, target 131072-token prefix, 512-token question,
256 output tokens, seed 42, uniform shuffled groups, no request warmup.
Prefix cache is flushed before each cell; server remains resident.
Baseline: Native at the same workload and concurrency; one screening run per cell.
Acceptance: correct configuration, unchanged sources/inputs, complete requests,
exact output lengths and valid cache reporting. Throughput gains are not assumed.
Stop: first failure within each mode, 90 minutes per cell, 9600 seconds per job.
Entry point: run.sh prepare | verify | run native|triton|kernel.
