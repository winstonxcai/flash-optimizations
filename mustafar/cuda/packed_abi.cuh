#pragma once

#include <cstddef>
#include <cstdint>

namespace mustafar::packed {

inline constexpr int kHeadDim = 512;
inline constexpr int kNopeDim = 448;
inline constexpr int kRopeDim = 64;
inline constexpr int kKeptValues = 256;
inline constexpr int kBitmapWords = 8;
inline constexpr int kScaleBytes = 8;
inline constexpr int kValueBytes = 256;
inline constexpr int kBitmapBytes = kBitmapWords * sizeof(std::uint64_t);
inline constexpr int kRecordBytes = kValueBytes + kBitmapBytes + kScaleBytes;

// Coordinate 64*word+lane is represented by bit 63-lane. Packed values are
// stored in monotonically increasing coordinate order.
__host__ __device__ constexpr std::uint64_t bitmap_mask(int lane) {
  return std::uint64_t{1} << (63 - lane);
}

static_assert(kHeadDim == kNopeDim + kRopeDim);
static_assert(kBitmapWords * 64 == kHeadDim);
static_assert(kRecordBytes == 328);

}  // namespace mustafar::packed
