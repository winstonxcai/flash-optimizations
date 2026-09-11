// PyBind surface for the sparse MLA passes.
//
// The softmax and its log-sum-exp live on the Python side
// (mustafar.sparse.sparse_forward), mirroring dhjoo98/mustafar's own
// reference wiring, so this module exposes only the two SpMM-shaped passes.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "sparse.h"

namespace {

void check_inputs(const torch::Tensor& values, const torch::Tensor& bitmaps,
                  const torch::Tensor& scales, const torch::Tensor& physical,
                  const torch::Tensor& raw,
                  const torch::Tensor& topk_lengths,
                  const torch::Tensor& freqs) {
  TORCH_CHECK(values.is_cuda() && bitmaps.is_cuda() && scales.is_cuda() &&
                  physical.is_cuda() && raw.is_cuda() &&
                  topk_lengths.is_cuda() && freqs.is_cuda(),
              "all sparse MLA inputs must be on CUDA");
  TORCH_CHECK(values.scalar_type() == at::kByte, "values must be uint8");
  TORCH_CHECK(bitmaps.scalar_type() == at::kLong ||
                  bitmaps.scalar_type() == at::kUInt64,
              "bitmaps must be int64/uint64");
  TORCH_CHECK(scales.scalar_type() == at::kByte, "scales must be uint8");
  TORCH_CHECK(physical.scalar_type() == at::kInt,
              "physical indices must be int32");
  TORCH_CHECK(raw.scalar_type() == at::kInt, "raw indices must be int32");
  TORCH_CHECK(topk_lengths.scalar_type() == at::kInt,
              "topk lengths must be int32");
  TORCH_CHECK(freqs.scalar_type() == at::kFloat,
              "freqs must be a float32 view_as_real of complex64");
  TORCH_CHECK(values.is_contiguous() && bitmaps.is_contiguous() &&
                  scales.is_contiguous() && physical.is_contiguous() &&
                  raw.is_contiguous() && topk_lengths.is_contiguous() &&
                  freqs.is_contiguous(),
              "all sparse MLA inputs must be contiguous");
  TORCH_CHECK(freqs.dim() == 3 && freqs.size(2) == 2,
              "freqs must be [max_position + 1, rope_pairs, 2]");
  TORCH_CHECK(values.size(-1) == 256, "values rows must be 256 bytes");
  TORCH_CHECK(bitmaps.size(-1) == 8 && scales.size(-1) == 8,
              "bitmap and scale rows must be 8 words/bytes");
}

// freqs arrives as a float32 [max_position+1, rope_pairs, 2] view of the
// complex64 table, i.e. exactly a float2 array. Torch allocations are
// 512-byte aligned and each pair is 8 bytes, so the reinterpret is aligned.
const float2* as_float2(const torch::Tensor& freqs) {
  return reinterpret_cast<const float2*>(freqs.data_ptr<float>());
}

}  // namespace

// Scores: [num_queries, num_heads, selected_k] float32. Already scaled by
// sm_scale, matching flash_mla_sparse_fwd's internal scaling.
torch::Tensor sparse_scores(torch::Tensor q, torch::Tensor values,
                                torch::Tensor bitmaps, torch::Tensor scales,
                                torch::Tensor physical, torch::Tensor raw,
                                torch::Tensor topk_lengths, torch::Tensor freqs,
                                double sm_scale) {
  check_inputs(values, bitmaps, scales, physical, raw, topk_lengths, freqs);
  TORCH_CHECK(q.is_cuda() && q.is_contiguous() &&
                  q.scalar_type() == at::kBFloat16,
              "q must be a contiguous bf16 CUDA tensor");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512, "q must be [queries, heads, 512]");
  TORCH_CHECK(physical.dim() == 2, "physical indices must be [queries, selected_k]");
  TORCH_CHECK(physical.size(0) == q.size(0) && raw.sizes() == physical.sizes(),
              "physical/raw must match the query count");

  const int num_queries = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int selected_k = static_cast<int>(physical.size(1));
  TORCH_CHECK(num_heads % 16 == 0, "num_heads must be a multiple of 16");

  auto scores = torch::empty({num_queries, num_heads, selected_k},
                             q.options().dtype(at::kFloat));
  if (num_queries == 0 || selected_k == 0) return scores;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mustafar_sparse_launch(
      stream, q.data_ptr(), scores.data_ptr(), values.data_ptr(),
      bitmaps.data_ptr(), scales.data_ptr(), physical.data_ptr(),
      raw.data_ptr(), topk_lengths.data_ptr(), as_float2(freqs), num_queries,
      selected_k, num_heads, static_cast<float>(sm_scale), /*scores_pass=*/true);
  return scores;
}

// Output: [num_queries, num_heads, 512] bf16. `p` must already be the
// topk-masked softmax (zero beyond each row's topk_length) so that invalid
// slots contribute nothing to the sum.
torch::Tensor sparse_output(torch::Tensor p, torch::Tensor values,
                                torch::Tensor bitmaps, torch::Tensor scales,
                                torch::Tensor physical, torch::Tensor raw,
                                torch::Tensor topk_lengths,
                                torch::Tensor freqs) {
  check_inputs(values, bitmaps, scales, physical, raw, topk_lengths, freqs);
  TORCH_CHECK(p.is_cuda() && p.is_contiguous() &&
                  p.scalar_type() == at::kBFloat16,
              "p must be a contiguous bf16 CUDA tensor");
  TORCH_CHECK(p.dim() == 3 && p.size(2) == 512,
              "p must be [queries, heads, selected_k] with selected_k == 512");
  TORCH_CHECK(physical.dim() == 2 && physical.size(0) == p.size(0) &&
                  physical.size(1) == p.size(2),
              "physical must be [queries, selected_k] matching p");

  const int num_queries = static_cast<int>(p.size(0));
  const int num_heads = static_cast<int>(p.size(1));
  const int selected_k = static_cast<int>(p.size(2));
  TORCH_CHECK(num_heads % 16 == 0, "num_heads must be a multiple of 16");

  auto out = torch::empty({num_queries, num_heads, 512}, p.options());
  if (num_queries == 0 || selected_k == 0) return out;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mustafar_sparse_launch(
      stream, p.data_ptr(), out.data_ptr(), values.data_ptr(),
      bitmaps.data_ptr(), scales.data_ptr(), physical.data_ptr(),
      raw.data_ptr(), topk_lengths.data_ptr(), as_float2(freqs), num_queries,
      selected_k, num_heads, 0.0f, /*scores_pass=*/false);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Direct 328-byte packed sparse MLA passes for DSV4-Flash";
  m.def("sparse_scores", &sparse_scores,
        "c4 scores: S = sm_scale * Q . K(T) from packed records");
  m.def("sparse_output", &sparse_output,
        "c4 output: O = P . V from packed records");
}
