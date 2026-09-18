#include <torch/extension.h>

void packed_to_native_cuda(
    const torch::Tensor& values,
    const torch::Tensor& bitmaps,
    const torch::Tensor& scales,
    const torch::Tensor& physical_indices,
    const torch::Tensor& raw_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& freq_pairs,
    const torch::Tensor& native_out,
    int64_t page_size,
    int64_t bytes_per_page);

void packed_to_native_cuda_optimized(
    const torch::Tensor& values,
    const torch::Tensor& bitmaps,
    const torch::Tensor& scales,
    const torch::Tensor& physical_indices,
    const torch::Tensor& raw_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& freq_pairs,
    const torch::Tensor& native_out,
    int64_t page_size,
    int64_t bytes_per_page);

void packed_to_native_cuda_early_rope(
    const torch::Tensor& values,
    const torch::Tensor& bitmaps,
    const torch::Tensor& scales,
    const torch::Tensor& physical_indices,
    const torch::Tensor& raw_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& freq_pairs,
    const torch::Tensor& native_out,
    int64_t page_size,
    int64_t bytes_per_page);

void packed_to_native_cuda_geometry(
    const torch::Tensor& values,
    const torch::Tensor& bitmaps,
    const torch::Tensor& scales,
    const torch::Tensor& physical_indices,
    const torch::Tensor& raw_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& freq_pairs,
    const torch::Tensor& native_out,
    int64_t page_size,
    int64_t bytes_per_page);

void packed_to_native_cuda_combined(
    const torch::Tensor& values,
    const torch::Tensor& bitmaps,
    const torch::Tensor& scales,
    const torch::Tensor& physical_indices,
    const torch::Tensor& raw_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& freq_pairs,
    const torch::Tensor& native_out,
    int64_t page_size,
    int64_t bytes_per_page);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "packed_to_native",
      &packed_to_native_cuda,
      "Packed to FlashMLA-native reconstruction (CUDA)");
  module.def(
      "packed_to_native_optimized",
      &packed_to_native_cuda_optimized,
      "Optimized packed to FlashMLA-native reconstruction (CUDA)");
  module.def(
      "packed_to_native_early_rope",
      &packed_to_native_cuda_early_rope,
      "Benchmark-only optimized reconstruction with early RoPE loads (CUDA)");
  module.def(
      "packed_to_native_geometry",
      &packed_to_native_cuda_geometry,
      "Optimized reconstruction with fixed K=512/page_size=16 geometry (CUDA)");
  module.def(
      "packed_to_native_combined",
      &packed_to_native_cuda_combined,
      "Optimized reconstruction with fixed geometry and early RoPE loads (CUDA)");
}
