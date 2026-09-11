#pragma once

#include <cuda_runtime.h>

// Both passes share one launch signature so the host never needs to know the
// kernel template. `in`/`out` are bf16/fp32 for the scores pass and
// bf16/bf16 for the output pass; see bindings.cpp for the tensor contract.
void mustafar_sparse_launch(cudaStream_t stream, const void* in, void* out,
                                const void* values, const void* bitmaps,
                                const void* scales, const void* physical,
                                const void* raw, const void* topk_lengths,
                                const void* freqs, int num_queries,
                                int selected_k, int num_heads, float sm_scale,
                                bool scores_pass);
