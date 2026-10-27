#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "turbo/distance/common/symphony_qg_distance.h"

namespace candidate81 {
constexpr size_t kDimension = 768;
constexpr size_t kLanes = 32;
constexpr size_t kCandidates = 137;
constexpr size_t kBlockBytes = 3328;
constexpr size_t kCompactBytes = kDimension / 8;

// Each input pointer addresses one center. The block bank excludes its file
// header.
inline void assemble_selected_block(const char *blocks, const uint8_t *compact,
                                    const uint8_t *valid,
                                    const uint32_t *indices, char *output) {
  if (!blocks || !compact || !valid || !indices || !output)
    throw std::invalid_argument("candidate81 null assembly input");
  std::array<char, kBlockBytes> result{};
  std::array<uint8_t, kLanes * kCompactBytes> selected{};
  rabitqlib::QGBatchDataMap<float> dst(result.data(), kDimension);
  for (size_t lane = 0; lane < kLanes; ++lane) {
    const size_t choice = indices[lane];
    if (choice >= kCandidates || valid[choice * kLanes + lane] != 1)
      throw std::invalid_argument("candidate81 invalid selected index");
    rabitqlib::ConstQGBatchDataMap<float> src(blocks + choice * kBlockBytes,
                                              kDimension);
    dst.f_add()[lane] = src.f_add()[lane];
    dst.f_rescale()[lane] = src.f_rescale()[lane];
    std::copy_n(compact + (choice * kLanes + lane) * kCompactBytes,
                kCompactBytes, selected.data() + lane * kCompactBytes);
  }
  rabitqlib::fastscan::pack_codes(kDimension, selected.data(), kLanes,
                                  dst.bin_code());
  std::memcpy(output, result.data(), result.size());
}

inline bool verify_selected_block(const char *blocks, const uint8_t *compact,
                                  const uint8_t *valid, const uint32_t *indices,
                                  const char *output) {
  if (!output)
    throw std::invalid_argument("candidate81 null verification block");
  std::array<char, kBlockBytes> reconstructed{};
  assemble_selected_block(blocks, compact, valid, indices,
                          reconstructed.data());
  return std::memcmp(reconstructed.data(), output, reconstructed.size()) == 0;
}
}  // namespace candidate81
