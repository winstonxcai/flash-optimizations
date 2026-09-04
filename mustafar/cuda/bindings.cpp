#include <torch/extension.h>

void packed_c4_to_native_cuda(
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
      "packed_c4_to_native",
      &packed_c4_to_native_cuda,
      "Packed C4 to FlashMLA-native reconstruction (CUDA)");
}
