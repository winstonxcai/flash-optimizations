# Mustafar Stage 2A: 128k Decode Serving on DeepSeek-V4-Flash-0731

## Setup

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Runtime: SGLang v0.5.17 with FlashInfer MXFP4 MoE
- Hardware: 4×H100, tensor parallelism 4
- Workload: 131,072 input tokens and 2,048 output tokens per request
- Load: concurrency 8, infinite request rate, identical random seed
- CUDA graphs: full decode graphs for batch sizes 1–16; prefill graphs disabled

## Serving results

| Mode | Requests/s | Output tokens/s | Total tokens/s | Median TTFT | Median TPOT |
|---|---:|---:|---:|---:|---:|
| Untouched native | **0.08388** | **171.80** | **11,166.73** | **40,742.59 ms** | **26.659 ms** |
| Stage 1 packed | 0.08153 | 166.98 | 10,853.77 | 40,956.64 ms | 27.853 ms |
| Stage 2A packed | 0.08194 | 167.81 | 10,907.46 | 40,952.76 ms | 27.623 ms |

Relative to untouched native, Stage 1 delivered 2.80% lower throughput and
4.48% higher median TPOT. Stage 2A recovered part of the reconstruction
overhead, but remained 2.32% lower in throughput and 3.61% higher in median
TPOT. Median TTFT was effectively unchanged: both packed modes were about
0.52% slower than native.

The untouched-native measurement used one warm-up wave and one measured wave
of eight requests. The existing Stage 1 and Stage 2A measurements used one
warm-up wave and three measured waves. This is sufficient for a directional
comparison, but it does not establish run-to-run variance.

All measured requests completed with exactly 2,048 output tokens and empty
error fields. Decode CUDA-graph replay was confirmed. The native run exposed
4,291,328 logical KV-token slots, compared with 5,198,080 for both packed
modes, giving the packed representation 1.2113× as much logical capacity.
