// Sparse MLA for the 328-byte TopMag50 packed layout.
//
// The c4 leg of DSV4-Flash attention reads packed records directly: each KV row
// is decompressed from fp8 E4M3 codes plus per-64-tile UE8M0 scales (pruned
// lanes zeroed), the 64-dim RoPE tail is rotated in place, and the row is used
// as BOTH key and value -- d_qk == d_v == 512, so one shared-memory tile serves
// the QK^T and the PV product. No reassembly into the 584-byte native layout.
//
// Reused from dhjoo98/mustafar (Flash-LLM, Apache-2.0): the MSB-first bitmap
// scan convention (coordinate 64*word+lane is bit 63-lane; packed values are in
// ascending coordinate order) and MMA_FP16_M16N8K16. The Flash-LLM SpMM tiling,
// the K_Global-dependent word indexing and the per-tile `idx` offset table are
// deliberately NOT used -- they are tied to a 4096-wide contraction.
//
// Split into two kernels with the softmax and its log-sum-exp between them,
// mirroring dhjoo98's own reference wiring (key formulation -> softmax on the
// host -> value formulation). Fusing the softmax into a single online-softmax
// kernel is a follow-on once this is bit-comparable to the Triton reference.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "../packed_abi.cuh"
#include "MMA_PTX.cuh"  // MMA_FP16_M16N8K16

namespace {

using mustafar::packed::kHeadDim;
using mustafar::packed::kKeptValues;
using mustafar::packed::kBitmapWords;
using mustafar::packed::kNopeDim;
using mustafar::packed::kRopeDim;

constexpr int kHeadsPerBlock = 16;  // one MMA_M tile
constexpr int kKvBlock = 64;        // KV rows resident in shared memory
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kRopePairs = kRopeDim / 2;

// Matches fused.cu:decode_e4m3fn, kept in sync deliberately so both pack
// consumers decode byte-identical values. Duplicated rather than shared because
// fused.cu is the frozen _FUSED fallback and must not be edited.
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

__device__ __forceinline__ uint32_t pack2(half lo, half hi) {
  const __half2 h = __halves2half2(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// Read one accumulator back as a float. The vendored MMA wrapper carries f32
// accumulators in uint32 registers (the PTX mma takes them as "r" operands), so
// the register holds the *bit pattern*: casting it numerically would turn
// 0x3FD00000 into 1.0706e9 instead of 1.625.
__device__ __forceinline__ float acc_float(uint32_t bits) {
  return __uint_as_float(bits);
}

__device__ __forceinline__ half bf16_to_half(__nv_bfloat16 x) {
  return __float2half(__bfloat162float(x));
}

// Decompress 64-channel groups [g_first, g_first+g_count) of one packed record
// into dst[0..511] (row-contiguous fp16). Pruned lanes are written as zero.
//
// `prefix` needs all eight bitmap words regardless of which groups this thread
// owns, so the whole word set is read here and the popcount prefix is recomputed
// per thread. That is 64 bytes of redundant L1 traffic per thread and it removes
// the per-tile `idx`/`NZ_Offset` tables dhjoo98 carries.
__device__ __forceinline__ void decompress_row(half* __restrict__ dst,
                                               const uint8_t* __restrict__ values,
                                               const uint64_t* __restrict__ bitmap,
                                               const uint8_t* __restrict__ scales,
                                               int g_first, int g_count) {
  uint32_t prefix[kBitmapWords];
  uint32_t acc = 0;
#pragma unroll
  for (int w = 0; w < kBitmapWords; ++w) {
    prefix[w] = acc;
    acc += static_cast<uint32_t>(__popcll(bitmap[w]));
  }

#pragma unroll
  for (int i = 0; i < g_count; ++i) {
    const int g = g_first + i;
    const float scale = ldexpf(1.0f, static_cast<int>(scales[g]) - 127);
    const uint8_t* src = values + prefix[g];
    half* out = dst + g * 64;

#pragma unroll
    for (int c = 0; c < 64; ++c) {
      out[c] = __float2half(0.0f);
    }
    uint64_t bmp = bitmap[g];
    const uint32_t nnz = static_cast<uint32_t>(__popcll(bmp));
    for (uint32_t j = 0; j < nnz; ++j) {
      const int pos = __clzll(bmp);
      bmp &= ~(uint64_t{0x8000000000000000ull} >> pos);
      out[pos] = __float2half(decode_e4m3fn(src[j]) * scale);
    }
  }
}

__device__ __forceinline__ void zero_row(half* __restrict__ dst) {
#pragma unroll
  for (int c = 0; c < kHeadDim; ++c) {
    dst[c] = __float2half(0.0f);
  }
}

// Rotate the 64-dim tail in place by position raw*4, matching
// triton/kernels.py:_rope_tail_complex_inplace_kernel exactly. A zeroed lane
// rotates to zero, so a partially pruned tail pair stays correct.
__device__ __forceinline__ void rope_tail(half* __restrict__ row, int raw,
                                          const float2* __restrict__ freqs) {
  if (raw < 0) return;
  const float2* f = freqs + static_cast<size_t>(raw) * 4 * kRopePairs;
  half* tail = row + kNopeDim;
#pragma unroll
  for (int p = 0; p < kRopePairs; ++p) {
    const float c = f[p].x;
    const float s = f[p].y;
    const float x0 = __half2float(tail[2 * p]);
    const float x1 = __half2float(tail[2 * p + 1]);
    tail[2 * p] = __float2half(x0 * c - x1 * s);
    tail[2 * p + 1] = __float2half(x0 * s + x1 * c);
  }
}

// Decompress rows [0, rows) of one KV block into sKV. 128 threads, 64 rows:
// two threads per row, four channel groups each.
__device__ __forceinline__ void fill_kv_block(
    half* __restrict__ sKV, int q_idx, int base, int rows, int selected_k,
    const uint8_t* __restrict__ values, const uint64_t* __restrict__ bitmaps,
    const uint8_t* __restrict__ scales, const int32_t* __restrict__ physical,
    const int32_t* __restrict__ raw, const float2* __restrict__ freqs) {
  const int r = threadIdx.x % kKvBlock;
  const int half_id = threadIdx.x / kKvBlock;
  half* dst = sKV + static_cast<size_t>(r) * kHeadDim;

  if (r < rows) {
    const int p = physical[static_cast<size_t>(q_idx) * selected_k + base + r];
    if (p >= 0) {
      decompress_row(dst, values + static_cast<size_t>(p) * kKeptValues,
                     bitmaps + static_cast<size_t>(p) * kBitmapWords,
                     scales + static_cast<size_t>(p) * kBitmapWords,
                     half_id * 4, 4);
    } else {
      zero_row(dst);
    }
  } else {
    zero_row(dst);
  }

  __syncthreads();
  if (half_id == 0 && r < rows) {
    const int p = physical[static_cast<size_t>(q_idx) * selected_k + base + r];
    if (p >= 0) {
      rope_tail(dst, raw[static_cast<size_t>(q_idx) * selected_k + base + r],
                freqs);
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 1: scores. S[q, h, j] = sm_scale * dot(Q[q, h, :], KV[q, j, :]).
//
// Block = (query, 16-head group). Each warp owns two 8-row n-tiles of the KV
// block; the contraction is the full head dim, so no cross-warp reduction.
// ---------------------------------------------------------------------------
__global__ void sparse_scores_kernel(
    const __nv_bfloat16* __restrict__ q, float* __restrict__ scores,
    const uint8_t* __restrict__ values, const uint64_t* __restrict__ bitmaps,
    const uint8_t* __restrict__ scales, const int32_t* __restrict__ physical,
    const int32_t* __restrict__ raw, const int32_t* __restrict__ topk_lengths,
    const float2* __restrict__ freqs, int num_queries, int selected_k, int num_heads,
    float sm_scale) {
  extern __shared__ __align__(128) half smem[];
  half* sQ = smem;                              // [16][512]
  half* sKV = smem + kHeadsPerBlock * kHeadDim;  // [64][512]

  const int q_idx = blockIdx.x;
  const int h0 = blockIdx.y * kHeadsPerBlock;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane >> 2;
  const int t = lane & 3;

  for (int i = threadIdx.x; i < kHeadsPerBlock * kHeadDim; i += kThreads) {
    const int hh = i / kHeadDim;
    const int dd = i % kHeadDim;
    sQ[i] = bf16_to_half(
        q[static_cast<size_t>(q_idx) * num_heads * kHeadDim +
          static_cast<size_t>(h0 + hh) * kHeadDim + dd]);
  }

  const int tlen = topk_lengths[q_idx];
  float* out = scores + static_cast<size_t>(q_idx) * num_heads * selected_k;

  for (int base = 0; base < tlen; base += kKvBlock) {
    const int rows = min(kKvBlock, tlen - base);
    fill_kv_block(sKV, q_idx, base, rows, selected_k, values, bitmaps, scales,
                  physical, raw, freqs);
    __syncthreads();

#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int n0 = (warp * 2 + nt) * 8;
      const int jbase = base + n0;
      uint32_t c[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int k = 0; k < kHeadDim / 16; ++k) {
        const int k0 = k * 16;
        uint32_t a[4];
        a[0] = pack2(sQ[g * kHeadDim + k0 + 2 * t],
                     sQ[g * kHeadDim + k0 + 2 * t + 1]);
        a[1] = pack2(sQ[(g + 8) * kHeadDim + k0 + 2 * t],
                     sQ[(g + 8) * kHeadDim + k0 + 2 * t + 1]);
        a[2] = pack2(sQ[g * kHeadDim + k0 + 2 * t + 8],
                     sQ[g * kHeadDim + k0 + 2 * t + 9]);
        a[3] = pack2(sQ[(g + 8) * kHeadDim + k0 + 2 * t + 8],
                     sQ[(g + 8) * kHeadDim + k0 + 2 * t + 9]);
        uint32_t b[2];
        const size_t row = static_cast<size_t>(n0 + g) * kHeadDim + k0 + 2 * t;
        b[0] = pack2(sKV[row], sKV[row + 1]);
        b[1] = pack2(sKV[row + 8], sKV[row + 9]);
        MMA_FP16_M16N8K16(c, a, b);
      }
      if (jbase + 2 * t + 1 < selected_k) {
        out[static_cast<size_t>(h0 + g) * selected_k + jbase + 2 * t] =
            acc_float(c[0]) * sm_scale;
        out[static_cast<size_t>(h0 + g) * selected_k + jbase + 2 * t + 1] =
            acc_float(c[1]) * sm_scale;
        out[static_cast<size_t>(h0 + g + 8) * selected_k + jbase + 2 * t] =
            acc_float(c[2]) * sm_scale;
        out[static_cast<size_t>(h0 + g + 8) * selected_k + jbase + 2 * t + 1] =
            acc_float(c[3]) * sm_scale;
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: output. O[q, h, d] = sum_j P[q, h, j] * KV[q, j, d].
//
// Same block shape. The KV block is the OUTER loop so each is decompressed
// once, which means every warp must hold accumulators for its whole slice of
// the head dim: 16 n-tiles of 8 = 128 columns = c[16][4].
// ---------------------------------------------------------------------------
__global__ void sparse_output_kernel(
    const __nv_bfloat16* __restrict__ p, __nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ values, const uint64_t* __restrict__ bitmaps,
    const uint8_t* __restrict__ scales, const int32_t* __restrict__ physical,
    const int32_t* __restrict__ raw, const int32_t* __restrict__ topk_lengths,
    const float2* __restrict__ freqs, int num_queries, int selected_k, int num_heads) {
  extern __shared__ __align__(128) half smem[];
  half* sP = smem;                               // [16][selected_k]
  half* sKV = smem + kHeadsPerBlock * selected_k;  // [64][512]

  const int q_idx = blockIdx.x;
  const int h0 = blockIdx.y * kHeadsPerBlock;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane >> 2;
  const int t = lane & 3;

  for (int i = threadIdx.x; i < kHeadsPerBlock * selected_k; i += kThreads) {
    const int hh = i / selected_k;
    const int jj = i % selected_k;
    sP[i] = bf16_to_half(
        p[static_cast<size_t>(q_idx) * num_heads * selected_k +
          static_cast<size_t>(h0 + hh) * selected_k + jj]);
  }

  uint32_t c[16][4];
#pragma unroll
  for (int nt = 0; nt < 16; ++nt)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[nt][i] = 0u;

  const int tlen = topk_lengths[q_idx];
  for (int base = 0; base < tlen; base += kKvBlock) {
    const int rows = min(kKvBlock, tlen - base);
    fill_kv_block(sKV, q_idx, base, rows, selected_k, values, bitmaps, scales,
                  physical, raw, freqs);
    __syncthreads();

#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int d0 = (warp * 16 + nt) * 8;
#pragma unroll
      for (int k = 0; k < kKvBlock / 16; ++k) {
        const int k0 = k * 16;
        const int j0 = base + k0;
        uint32_t a[4];
        a[0] = pack2(sP[g * selected_k + j0 + 2 * t],
                     sP[g * selected_k + j0 + 2 * t + 1]);
        a[1] = pack2(sP[(g + 8) * selected_k + j0 + 2 * t],
                     sP[(g + 8) * selected_k + j0 + 2 * t + 1]);
        a[2] = pack2(sP[g * selected_k + j0 + 2 * t + 8],
                     sP[g * selected_k + j0 + 2 * t + 9]);
        a[3] = pack2(sP[(g + 8) * selected_k + j0 + 2 * t + 8],
                     sP[(g + 8) * selected_k + j0 + 2 * t + 9]);
        uint32_t b[2];
        b[0] = pack2(sKV[static_cast<size_t>(k0 + 2 * t) * kHeadDim + d0 + g],
                     sKV[static_cast<size_t>(k0 + 2 * t + 1) * kHeadDim + d0 + g]);
        b[1] = pack2(sKV[static_cast<size_t>(k0 + 2 * t + 8) * kHeadDim + d0 + g],
                     sKV[static_cast<size_t>(k0 + 2 * t + 9) * kHeadDim + d0 + g]);
        MMA_FP16_M16N8K16(c[nt], a, b);
      }
    }
    __syncthreads();
  }

  __nv_bfloat16* dst = out + static_cast<size_t>(q_idx) * num_heads * kHeadDim;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    const int d0 = (warp * 16 + nt) * 8;
    dst[static_cast<size_t>(h0 + g) * kHeadDim + d0 + 2 * t] =
        __float2bfloat16(acc_float(c[nt][0]));
    dst[static_cast<size_t>(h0 + g) * kHeadDim + d0 + 2 * t + 1] =
        __float2bfloat16(acc_float(c[nt][1]));
    dst[static_cast<size_t>(h0 + g + 8) * kHeadDim + d0 + 2 * t] =
        __float2bfloat16(acc_float(c[nt][2]));
    dst[static_cast<size_t>(h0 + g + 8) * kHeadDim + d0 + 2 * t + 1] =
        __float2bfloat16(acc_float(c[nt][3]));
  }
}

constexpr int scores_smem_bytes() {
  return (kHeadsPerBlock * kHeadDim + kKvBlock * kHeadDim) * sizeof(half);
}

// sP is [16][selected_k] and selected_k is a runtime argument, so the output
// pass needs at least selected_k * 16 halves of P storage.
int output_smem_bytes_for(int selected_k) {
  const int cols = selected_k > kHeadDim ? selected_k : kHeadDim;
  return (kHeadsPerBlock * cols + kKvBlock * kHeadDim) * sizeof(half);
}

}  // namespace

void mustafar_sparse_launch(
    cudaStream_t stream, const void* in, void* out,
    const void* values, const void* bitmaps, const void* scales,
    const void* physical, const void* raw, const void* topk_lengths,
    const void* freqs, int num_queries, int selected_k, int num_heads,
    float sm_scale, bool scores_pass) {
  const dim3 grid(num_queries, num_heads / kHeadsPerBlock);
  const dim3 block(kThreads);

  if (scores_pass) {
    const int smem = scores_smem_bytes();
    static int configured_scores = 0;
    if (smem > configured_scores) {
      cudaFuncSetAttribute(sparse_scores_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      configured_scores = smem;
    }
    sparse_scores_kernel<<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(in), static_cast<float*>(out),
        static_cast<const uint8_t*>(values),
        static_cast<const uint64_t*>(bitmaps),
        static_cast<const uint8_t*>(scales),
        static_cast<const int32_t*>(physical), static_cast<const int32_t*>(raw),
        static_cast<const int32_t*>(topk_lengths),
        static_cast<const float2*>(freqs), num_queries, selected_k, num_heads,
        sm_scale);
  } else {
    const int smem = output_smem_bytes_for(selected_k);
    static int configured_output = 0;
    if (smem > configured_output) {
      cudaFuncSetAttribute(sparse_output_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      configured_output = smem;
    }
    sparse_output_kernel<<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(in),
        static_cast<__nv_bfloat16*>(out),
        static_cast<const uint8_t*>(values),
        static_cast<const uint64_t*>(bitmaps),
        static_cast<const uint8_t*>(scales),
        static_cast<const int32_t*>(physical), static_cast<const int32_t*>(raw),
        static_cast<const int32_t*>(topk_lengths),
        static_cast<const float2*>(freqs), num_queries, selected_k, num_heads);
  }
}
