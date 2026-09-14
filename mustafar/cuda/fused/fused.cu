#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "packed_abi.cuh"

namespace {

using mustafar::packed::kBitmapWords;
using mustafar::packed::kHeadDim;
using mustafar::packed::kKeptValues;
using mustafar::packed::kNopeDim;

constexpr int kNativeValueBytes = 576;
constexpr int kNativeScaleBytes = 8;
constexpr int kWarpsPerBlock = 4;
constexpr int kWarpSize = 32;

template <typename index_t>
__device__ __forceinline__ int64_t load_index(
    const index_t* data, int64_t offset) {
  return static_cast<int64_t>(data[offset]);
}

__device__ __forceinline__ float decode_e4m3fn(uint8_t code) {
  const int sign = code >> 7;
  const int exponent = (code >> 3) & 0xF;
  const int mantissa = code & 0x7;
  float value;
  if (exponent == 0) {
    value = ldexpf(static_cast<float>(mantissa), -9);
  } else if (exponent == 15 && mantissa == 7) {
    value = 0.0f;
  } else {
    value = ldexpf(static_cast<float>(8 + mantissa), exponent - 10);
  }
  return sign ? -value : value;
}

__device__ __forceinline__ int packed_rank(
    const uint64_t* bitmap, int coordinate) {
  const int word = coordinate >> 6;
  const int lane = coordinate & 63;
  int rank = 0;
  #pragma unroll
  for (int w = 0; w < kBitmapWords; ++w) {
    if (w < word) {
      rank += __popcll(bitmap[w]);
    }
  }
  if (lane != 0) {
    rank += __popcll(bitmap[word] >> (64 - lane));
  }
  return rank;
}

__device__ __forceinline__ bool coordinate_is_kept(
    const uint64_t* bitmap, int coordinate) {
  const int word = coordinate >> 6;
  const int lane = coordinate & 63;
  return (bitmap[word] & (uint64_t{1} << (63 - lane))) != 0;
}

__device__ __forceinline__ uint8_t load_packed_byte(
    uint64_t bitmap_word, int prefix, const uint8_t* packed_values,
    int coordinate) {
  const int lane = coordinate & 63;
  if ((bitmap_word & (uint64_t{1} << (63 - lane))) == 0) {
    return 0;
  }
  const int rank = prefix + (lane == 0
      ? 0
      : __popcll(bitmap_word >> (64 - lane)));
  return packed_values[rank];
}

template <typename index_t>
__global__ void packed_to_native_kernel(
    const uint8_t* __restrict__ values,
    const uint64_t* __restrict__ bitmaps,
    const uint8_t* __restrict__ scales,
    const index_t* __restrict__ physical_indices,
    const index_t* __restrict__ raw_indices,
    const index_t* __restrict__ topk_lengths,
    const float* __restrict__ freq_pairs,
    uint8_t* __restrict__ native_out,
    int64_t rows,
    int64_t selected_k,
    int64_t pool_rows,
    int64_t freq_rows,
    int page_size,
    int64_t bytes_per_page) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  const int64_t query = row / selected_k;
  const int64_t rank_in_query = row - query * selected_k;
  const int64_t physical = load_index(physical_indices, row);
  const int64_t raw = load_index(raw_indices, row);
  int64_t topk = load_index(topk_lengths, query);
  topk = topk < 0 ? 0 : (topk > selected_k ? selected_k : topk);

  const int64_t output_page = row / page_size;
  const int64_t output_offset = row - output_page * page_size;
  uint8_t* value_out = native_out + output_page * bytes_per_page
      + output_offset * kNativeValueBytes;
  uint8_t* scale_out = native_out + output_page * bytes_per_page
      + static_cast<int64_t>(page_size) * kNativeValueBytes
      + output_offset * kNativeScaleBytes;

  for (int byte = lane; byte < kNativeValueBytes; byte += kWarpSize) {
    value_out[byte] = 0;
  }
  if (lane < kNativeScaleBytes) {
    scale_out[lane] = 0;
  }

  const bool valid = rank_in_query < topk && physical >= 0
      && physical < pool_rows && raw >= 0 && raw < (freq_rows + 3) / 4
      && raw * int64_t{4} < freq_rows;
  if (!valid) {
    return;
  }

  const uint64_t* bitmap = bitmaps + physical * kBitmapWords;
  const uint8_t* packed_values = values + physical * kKeptValues;
  const uint8_t* packed_scales = scales + physical * kBitmapWords;

  for (int coordinate = lane; coordinate < kNopeDim; coordinate += kWarpSize) {
    if (coordinate_is_kept(bitmap, coordinate)) {
      value_out[coordinate] = packed_values[packed_rank(bitmap, coordinate)];
    }
  }
  if (lane < 7) {
    scale_out[lane] = packed_scales[lane];
  }

  const int coordinate0 = kNopeDim + 2 * lane;
  const int coordinate1 = coordinate0 + 1;
  float x0 = 0.0f;
  float x1 = 0.0f;
  if (coordinate_is_kept(bitmap, coordinate0)) {
    x0 = decode_e4m3fn(packed_values[packed_rank(bitmap, coordinate0)]);
  }
  if (coordinate_is_kept(bitmap, coordinate1)) {
    x1 = decode_e4m3fn(packed_values[packed_rank(bitmap, coordinate1)]);
  }
  const float scale = ldexpf(1.0f, static_cast<int>(packed_scales[7]) - 127);
  x0 *= scale;
  x1 *= scale;

  const int64_t frequency = (raw * int64_t{4} * 32 + lane) * 2;
  const float cosine = freq_pairs[frequency];
  const float sine = freq_pairs[frequency + 1];
  __nv_bfloat16* tail = reinterpret_cast<__nv_bfloat16*>(value_out + kNopeDim);
  tail[2 * lane] = __float2bfloat16_rn(x0 * cosine - x1 * sine);
  tail[2 * lane + 1] = __float2bfloat16_rn(x0 * sine + x1 * cosine);
}

template <typename index_t, bool kEarlyRopeLoads, int kFixedSelectedK = 0,
          int kFixedPageSize = 0>
__global__ void packed_to_native_kernel_optimized(
    const uint8_t* __restrict__ values,
    const uint64_t* __restrict__ bitmaps,
    const uint8_t* __restrict__ scales,
    const index_t* __restrict__ physical_indices,
    const index_t* __restrict__ raw_indices,
    const index_t* __restrict__ topk_lengths,
    const float* __restrict__ freq_pairs,
    uint8_t* __restrict__ native_out,
    int64_t rows,
    int64_t selected_k,
    int64_t pool_rows,
    int64_t freq_rows,
    int page_size,
    int64_t bytes_per_page) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }

  int64_t query;
  int64_t rank_in_query;
  if constexpr (kFixedSelectedK == 512) {
    query = row >> 9;
    rank_in_query = row & 511;
  } else {
    query = row / selected_k;
    rank_in_query = row - query * selected_k;
  }
  const int64_t physical = load_index(physical_indices, row);
  const int64_t raw = load_index(raw_indices, row);
  int64_t topk = load_index(topk_lengths, query);
  topk = topk < 0 ? 0 : (topk > selected_k ? selected_k : topk);

  int64_t output_page;
  int64_t output_offset;
  if constexpr (kFixedPageSize == 16) {
    output_page = row >> 4;
    output_offset = row & 15;
  } else {
    output_page = row / page_size;
    output_offset = row - output_page * page_size;
  }
  uint8_t* value_out = native_out + output_page * bytes_per_page
      + output_offset * kNativeValueBytes;
  uint8_t* scale_out = native_out + output_page * bytes_per_page
      + static_cast<int64_t>(page_size) * kNativeValueBytes
      + output_offset * kNativeScaleBytes;

  const bool valid = rank_in_query < topk && physical >= 0
      && physical < pool_rows && raw >= 0 && raw < (freq_rows + 3) / 4
      && raw * int64_t{4} < freq_rows;
  auto* value_words = reinterpret_cast<uint32_t*>(value_out);
  if (!valid) {
    for (int word = lane; word < kNativeValueBytes / 4; word += kWarpSize) {
      value_words[word] = 0;
    }
    if (lane == 0) {
      *reinterpret_cast<uint64_t*>(scale_out) = 0;
    }
    return;
  }

  const uint64_t* bitmap = bitmaps + physical * kBitmapWords;
  const uint8_t* packed_values = values + physical * kKeptValues;
  const uint8_t* packed_scales = scales + physical * kBitmapWords;

  float cosine = 0.0f;
  float sine = 0.0f;
  float scale = 0.0f;
  if constexpr (kEarlyRopeLoads) {
    const int64_t frequency = (raw * int64_t{4} * 32 + lane) * 2;
    cosine = freq_pairs[frequency];
    sine = freq_pairs[frequency + 1];
    scale = ldexpf(
        1.0f, static_cast<int>(packed_scales[kBitmapWords - 1]) - 127);
  }

  __shared__ uint64_t bitmap_shared[kWarpsPerBlock][kBitmapWords];
  // Preserve v5C's two-word guard and 272-byte per-warp shared-memory stride.
  // The exact historical layout is required for the validated candidate.
  __shared__ uint64_t values_shared[kWarpsPerBlock][kKeptValues / 8 + 2];
  __shared__ int prefix_shared[kWarpsPerBlock][kBitmapWords];

  if (lane < kBitmapWords) {
    bitmap_shared[warp][lane] = bitmap[lane];
  }
  values_shared[warp][lane] =
      reinterpret_cast<const uint64_t*>(packed_values)[lane];
  if (lane < 2) {
    values_shared[warp][kKeptValues / 8 + lane] = 0;
  }
  __syncwarp();

  if (lane == 0) {
    int prefix = 0;
    #pragma unroll
    for (int word = 0; word < kBitmapWords; ++word) {
      prefix_shared[warp][word] = prefix;
      prefix += __popcll(bitmap_shared[warp][word]);
    }
  }
  __syncwarp();

  const uint8_t* staged_values =
      reinterpret_cast<const uint8_t*>(values_shared[warp]);
  if (lane < kNopeDim / 16) {
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
      const int word_index = lane * 4 + group;
      const int coordinate = word_index * 4;
      const int bitmap_word = coordinate >> 6;
      const int bit_in_word = coordinate & 63;
      const uint64_t bitmap_word_bits = bitmap_shared[warp][bitmap_word];
      const int rank = prefix_shared[warp][bitmap_word]
          + (bit_in_word == 0
              ? 0
              : __popcll(bitmap_word_bits >> (64 - bit_in_word)));
      uint32_t packed = 0;
      int retained = 0;
      #pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        const int bit = bit_in_word + byte;
        const bool kept = (bitmap_word_bits
            & (uint64_t{1} << (63 - bit))) != 0;
        uint8_t value = 0;
        if (kept) {
          value = staged_values[rank + retained];
          ++retained;
        }
        packed |= static_cast<uint32_t>(value) << (byte * 8);
      }
      value_words[word_index] = packed;
    }
  }

  if (lane == 0) {
    *reinterpret_cast<uint64_t*>(scale_out) =
        *reinterpret_cast<const uint64_t*>(packed_scales)
        & 0x00FFFFFFFFFFFFFFull;
  }

  if constexpr (!kEarlyRopeLoads) {
    const int64_t frequency = (raw * int64_t{4} * 32 + lane) * 2;
    cosine = freq_pairs[frequency];
    sine = freq_pairs[frequency + 1];
    scale = ldexpf(
        1.0f, static_cast<int>(packed_scales[kBitmapWords - 1]) - 127);
  }

  const int coordinate0 = kNopeDim + 2 * lane;
  const int coordinate1 = coordinate0 + 1;
  const int rope_word = kBitmapWords - 1;
  const uint64_t rope_bitmap = bitmap_shared[warp][rope_word];
  const int rope_prefix = prefix_shared[warp][rope_word];
  float x0 = decode_e4m3fn(load_packed_byte(
      rope_bitmap, rope_prefix, staged_values, coordinate0));
  float x1 = decode_e4m3fn(load_packed_byte(
      rope_bitmap, rope_prefix, staged_values, coordinate1));
  x0 *= scale;
  x1 *= scale;

  const __nv_bfloat16 y0 = __float2bfloat16_rn(x0 * cosine - x1 * sine);
  const __nv_bfloat16 y1 = __float2bfloat16_rn(x0 * sine + x1 * cosine);
  value_words[kNopeDim / 4 + lane] =
      static_cast<uint32_t>(__bfloat16_as_ushort(y0))
      | (static_cast<uint32_t>(__bfloat16_as_ushort(y1)) << 16);
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void packed_to_native_cuda_impl(
    const torch::Tensor& values,
    const torch::Tensor& bitmaps,
    const torch::Tensor& scales,
    const torch::Tensor& physical_indices,
    const torch::Tensor& raw_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& freq_pairs,
    const torch::Tensor& native_out,
    int64_t page_size,
    int64_t bytes_per_page,
    bool optimized,
    bool early_rope,
    bool geometry) {
  check_cuda_contiguous(values, "values");
  check_cuda_contiguous(bitmaps, "bitmaps");
  check_cuda_contiguous(scales, "scales");
  check_cuda_contiguous(physical_indices, "physical_indices");
  check_cuda_contiguous(raw_indices, "raw_indices");
  check_cuda_contiguous(topk_lengths, "topk_lengths");
  check_cuda_contiguous(freq_pairs, "freq_pairs");
  check_cuda_contiguous(native_out, "native_out");
  TORCH_CHECK(values.scalar_type() == at::kByte, "values must be uint8");
  TORCH_CHECK(bitmaps.scalar_type() == at::kUInt64, "bitmaps must be uint64");
  TORCH_CHECK(scales.scalar_type() == at::kByte, "scales must be uint8");
  TORCH_CHECK(freq_pairs.scalar_type() == at::kFloat, "freq_pairs must be float32");
  TORCH_CHECK(native_out.scalar_type() == at::kByte, "native_out must be uint8");
  TORCH_CHECK(physical_indices.scalar_type() == raw_indices.scalar_type(),
              "physical_indices and raw_indices must have the same dtype");
  TORCH_CHECK(physical_indices.scalar_type() == topk_lengths.scalar_type(),
              "indices and topk_lengths must have the same dtype");
  TORCH_CHECK(physical_indices.scalar_type() == at::kInt
                  || physical_indices.scalar_type() == at::kLong,
              "indices must be int32 or int64");
  TORCH_CHECK(physical_indices.dim() == 2, "indices must be [batch, topk]");
  TORCH_CHECK(raw_indices.sizes() == physical_indices.sizes(),
              "physical_indices and raw_indices must have equal shape");
  TORCH_CHECK(topk_lengths.numel() >= physical_indices.size(0),
              "topk_lengths must contain one value per query");
  TORCH_CHECK(values.numel() % kKeptValues == 0,
              "values must contain 256 bytes per packed row");
  const int64_t pool_rows = values.numel() / kKeptValues;
  TORCH_CHECK(bitmaps.numel() == pool_rows * kBitmapWords,
              "bitmaps must contain 8 words per packed row");
  TORCH_CHECK(scales.numel() == pool_rows * kBitmapWords,
              "scales must contain 8 bytes per packed row");
  TORCH_CHECK(freq_pairs.dim() == 3 && freq_pairs.size(1) == 32
                  && freq_pairs.size(2) == 2,
              "freq_pairs must have shape [positions, 32, 2]");
  TORCH_CHECK(page_size > 0 && bytes_per_page >= page_size * 584,
              "invalid native page geometry");
  const auto device = values.device();
  TORCH_CHECK(bitmaps.device() == device && scales.device() == device
                  && physical_indices.device() == device
                  && raw_indices.device() == device
                  && topk_lengths.device() == device
                  && freq_pairs.device() == device
                  && native_out.device() == device,
              "all tensors must be on the same CUDA device");

  const int64_t rows = physical_indices.numel();
  const int64_t output_pages = (rows + page_size - 1) / page_size;
  TORCH_CHECK(native_out.numel() >= output_pages * bytes_per_page,
              "native_out is too small");
  if (rows == 0) {
    return;
  }

  c10::cuda::CUDAGuard device_guard(values.device());
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(values.get_device());
  if (physical_indices.scalar_type() == at::kInt) {
    if (optimized) {
      if (geometry) {
        TORCH_CHECK(physical_indices.size(1) == 512 && page_size == 16,
                    "geometry candidate requires selected_k=512 and page_size=16");
        packed_to_native_kernel_optimized<int32_t, false, 512, 16>
            <<<grid, block, 0, stream>>>(
                values.data_ptr<uint8_t>(), bitmaps.data_ptr<uint64_t>(),
                scales.data_ptr<uint8_t>(), physical_indices.data_ptr<int32_t>(),
                raw_indices.data_ptr<int32_t>(), topk_lengths.data_ptr<int32_t>(),
                freq_pairs.data_ptr<float>(), native_out.data_ptr<uint8_t>(), rows,
                physical_indices.size(1), pool_rows, freq_pairs.size(0),
                static_cast<int>(page_size), bytes_per_page);
      } else if (early_rope) {
        packed_to_native_kernel_optimized<int32_t, true><<<grid, block, 0, stream>>>(
            values.data_ptr<uint8_t>(), bitmaps.data_ptr<uint64_t>(),
            scales.data_ptr<uint8_t>(), physical_indices.data_ptr<int32_t>(),
            raw_indices.data_ptr<int32_t>(), topk_lengths.data_ptr<int32_t>(),
            freq_pairs.data_ptr<float>(), native_out.data_ptr<uint8_t>(), rows,
            physical_indices.size(1), pool_rows, freq_pairs.size(0),
            static_cast<int>(page_size), bytes_per_page);
      } else {
        packed_to_native_kernel_optimized<int32_t, false><<<grid, block, 0, stream>>>(
          values.data_ptr<uint8_t>(), bitmaps.data_ptr<uint64_t>(),
          scales.data_ptr<uint8_t>(), physical_indices.data_ptr<int32_t>(),
          raw_indices.data_ptr<int32_t>(), topk_lengths.data_ptr<int32_t>(),
          freq_pairs.data_ptr<float>(), native_out.data_ptr<uint8_t>(), rows,
          physical_indices.size(1), pool_rows, freq_pairs.size(0),
          static_cast<int>(page_size), bytes_per_page);
      }
    } else {
      packed_to_native_kernel<int32_t><<<grid, block, 0, stream>>>(
          values.data_ptr<uint8_t>(), bitmaps.data_ptr<uint64_t>(),
          scales.data_ptr<uint8_t>(), physical_indices.data_ptr<int32_t>(),
          raw_indices.data_ptr<int32_t>(), topk_lengths.data_ptr<int32_t>(),
          freq_pairs.data_ptr<float>(), native_out.data_ptr<uint8_t>(), rows,
          physical_indices.size(1), pool_rows, freq_pairs.size(0),
          static_cast<int>(page_size), bytes_per_page);
    }
  } else if (optimized) {
    if (geometry) {
      TORCH_CHECK(physical_indices.size(1) == 512 && page_size == 16,
                  "geometry candidate requires selected_k=512 and page_size=16");
      packed_to_native_kernel_optimized<int64_t, false, 512, 16>
          <<<grid, block, 0, stream>>>(
              values.data_ptr<uint8_t>(), bitmaps.data_ptr<uint64_t>(),
              scales.data_ptr<uint8_t>(), physical_indices.data_ptr<int64_t>(),
              raw_indices.data_ptr<int64_t>(), topk_lengths.data_ptr<int64_t>(),
              freq_pairs.data_ptr<float>(), native_out.data_ptr<uint8_t>(), rows,
              physical_indices.size(1), pool_rows, freq_pairs.size(0),
              static_cast<int>(page_size), bytes_per_page);
    } else if (early_rope) {
      packed_to_native_kernel_optimized<int64_t, true><<<grid, block, 0, stream>>>(
          values.data_ptr<uint8_t>(), bitmaps.data_ptr<uint64_t>(),
          scales.data_ptr<uint8_t>(), physical_indices.data_ptr<int64_t>(),
          raw_indices.data_ptr<int64_t>(), topk_lengths.data_ptr<int64_t>(),
          freq_pairs.data_ptr<float>(), native_out.data_ptr<uint8_t>(), rows,
          physical_indices.size(1), pool_rows, freq_pairs.size(0),
          static_cast<int>(page_size), bytes_per_page);
    } else {
      packed_to_native_kernel_optimized<int64_t, false><<<grid, block, 0, stream>>>(
          values.data_ptr<uint8_t>(), bitmaps.data_ptr<uint64_t>(),
          scales.data_ptr<uint8_t>(), physical_indices.data_ptr<int64_t>(),
          raw_indices.data_ptr<int64_t>(), topk_lengths.data_ptr<int64_t>(),
          freq_pairs.data_ptr<float>(), native_out.data_ptr<uint8_t>(), rows,
          physical_indices.size(1), pool_rows, freq_pairs.size(0),
          static_cast<int>(page_size), bytes_per_page);
    }
  } else {
    packed_to_native_kernel<int64_t><<<grid, block, 0, stream>>>(
        values.data_ptr<uint8_t>(), bitmaps.data_ptr<uint64_t>(),
        scales.data_ptr<uint8_t>(), physical_indices.data_ptr<int64_t>(),
        raw_indices.data_ptr<int64_t>(), topk_lengths.data_ptr<int64_t>(),
        freq_pairs.data_ptr<float>(), native_out.data_ptr<uint8_t>(), rows,
        physical_indices.size(1), pool_rows, freq_pairs.size(0),
        static_cast<int>(page_size), bytes_per_page);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int64_t bytes_per_page) {
  packed_to_native_cuda_impl(
      values, bitmaps, scales, physical_indices, raw_indices, topk_lengths,
      freq_pairs, native_out, page_size, bytes_per_page, false, false, false);
}

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
    int64_t bytes_per_page) {
  packed_to_native_cuda_impl(
      values, bitmaps, scales, physical_indices, raw_indices, topk_lengths,
      freq_pairs, native_out, page_size, bytes_per_page, true, false, false);
}

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
    int64_t bytes_per_page) {
  // This is a benchmark-only variant of the optimized body. It moves the
  // frequency-pair and RoPE scale loads before shared-value staging; all
  // storage, arithmetic, masking, and output behavior remains identical.
  packed_to_native_cuda_impl(
      values, bitmaps, scales, physical_indices, raw_indices, topk_lengths,
      freq_pairs, native_out, page_size, bytes_per_page, true, true, false);
}

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
    int64_t bytes_per_page) {
  packed_to_native_cuda_impl(
      values, bitmaps, scales, physical_indices, raw_indices, topk_lengths,
      freq_pairs, native_out, page_size, bytes_per_page, true, false, true);
}

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
    int64_t bytes_per_page) {
  // Combine the two validated micro-optimizations: fixed serving geometry and
  // early RoPE operand loads. Storage, arithmetic, and masking are unchanged.
  packed_to_native_cuda_impl(
      values, bitmaps, scales, physical_indices, raw_indices, topk_lengths,
      freq_pairs, native_out, page_size, bytes_per_page, true, true, true);
}
