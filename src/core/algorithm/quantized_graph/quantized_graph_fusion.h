// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "quantized_graph_bucket.h"

namespace zvec::core {
// Research-only versioned copy of the frozen LIFO bucket. Version checks must
// happen before consulting the bucket lower bound for adaptive termination.
class QuantizedGraphFusionBucket {
 public:
  static constexpr size_t N = QG_BUCKET_COUNT;
  static_assert(N >= 64 && N <= 4096 && N % 64 == 0);
#if defined(QG_QUEUE_STATS)
  size_t count = 0, peak = 0;
  size_t memory_bytes() const {
    size_t n = sizeof(*this) + touched_.capacity() * sizeof(size_t);
    for (const auto &b : bins_) n += b.capacity() * sizeof(uint64_t);
    return n;
  }
#endif
  void reset(size_t) {
#if defined(QG_QUEUE_STATS)
    count = peak = 0;
#endif
    for (size_t b : touched_) bins_[b].clear();
    touched_.clear();
    bits_.fill(0);
    summary_ = 0;
    width_ = 1.0f;
#if defined(QG_BUCKET_CACHEMIN)
    first_ = N;
#endif
  }
  void set_scale(float distance) {
    width_ = std::max(distance, 1e-6f) * QG_BUCKET_WIDTH;
  }
  float width() const {
    return width_;
  }
  size_t locate(float d) const {
    size_t b = 0;
    if (d >= 0) {
      const float x = d / width_;
      b = x >= float(N - 2) ? N - 1 : size_t(x) + 1;
    }
    return b;
  }
  float lower_bound(float d) const {
    size_t b = locate(d);
    return b == 0 ? -std::numeric_limits<float>::infinity()
                  : float(b - 1) * width_;
  }
  void insert(uint32_t id, float d, uint32_t version = 0) {
    size_t b = locate(d);
    if (bins_[b].empty()) touched_.push_back(b);
    bins_[b].push_back((uint64_t(version) << 32) | id);
#if defined(QG_QUEUE_STATS)
    ++count;
    peak = std::max(peak, count);
#endif
    bits_[b / 64] |= uint64_t{1} << (b % 64);
    summary_ |= uint64_t{1} << (b / 64);
#if defined(QG_BUCKET_CACHEMIN)
    first_ = std::min(first_, b);
#endif
  }
  bool has_next() const {
    return summary_ != 0;
  }
  size_t first() const {
#if defined(QG_BUCKET_CACHEMIN)
    return first_;
#else
    size_t w = __builtin_ctzll(summary_);
    return 64 * w + __builtin_ctzll(bits_[w]);
#endif
  }
  uint32_t next_id_at(size_t) const {
    return bins_[first()].back();
  }
  uint32_t next_version() const {
    return uint32_t(bins_[first()].back() >> 32);
  }
  float next_distance() const {
    size_t b = first();
    return b == 0 ? -std::numeric_limits<float>::infinity()
                  : float(b - 1) * width_;
  }
  uint32_t pop() {
#if defined(QG_QUEUE_STATS)
    --count;
#endif
    size_t b = first();
    uint32_t id = bins_[b].back();
    bins_[b].pop_back();
    if (bins_[b].empty()) {
      bits_[b / 64] &= ~(uint64_t{1} << (b % 64));
      if (bits_[b / 64] == 0) summary_ &= ~(uint64_t{1} << (b / 64));
#if defined(QG_BUCKET_CACHEMIN)
      if (summary_) {
        size_t w = __builtin_ctzll(summary_);
        first_ = 64 * w + __builtin_ctzll(bits_[w]);
      } else
        first_ = N;
#endif
    }
    return id;
  }
  bool is_full(float) const {
    return false;
  }

 private:
  std::array<std::vector<uint64_t>, N> bins_;
  std::array<uint64_t, N / 64> bits_{};
  uint64_t summary_ = 0;
  std::vector<size_t> touched_;
  float width_ = 1;
#if defined(QG_BUCKET_CACHEMIN)
  size_t first_ = N;
#endif
};

class QuantizedGraphFusion {
 public:
  static constexpr size_t kSlots = 2048;
  static constexpr size_t kProbes = 8;
  static constexpr float kLambda = 0.5903141882326695f;
  struct Slot {
    uint32_t id = 0, epoch = 0;
    float estimate = 0, variance = 0;
    // 1: first arrival, 2: second processed without fusion, 3: fused.
    uint32_t phase = 0;
  };
#if defined(QG_FUSION_AUDIT)
  uint64_t first = 0, second = 0, fused = 0, later = 0, overflow = 0;
  uint64_t fallback = 0, stale = 0;
#endif
  void reset(bool enabled) {
    enabled_ = enabled;
    if (++epoch_ == 0) {
      slots_.fill({});
      epoch_ = 1;
    }
#if defined(QG_FUSION_AUDIT)
    first = second = fused = later = overflow = fallback = stale = 0;
#endif
  }
  static size_t hash(uint32_t id) {
    return (id * uint32_t{2654435761}) & (kSlots - 1);
  }
  float observe(uint32_t id, float estimate, float variance) {
    if (!std::isfinite(estimate) || !std::isfinite(variance) || variance < 0)
      return estimate;
    size_t h = hash(id);
    for (size_t p = 0; p < kProbes; ++p, h = (h + 1) & (kSlots - 1)) {
      Slot &s = slots_[h];
      if (s.epoch != epoch_) {
        s = {id, epoch_, estimate, variance, 1};
#if defined(QG_FUSION_AUDIT)
        ++first;
#endif
        return estimate;
      }
      if (s.id != id) continue;
      if (s.phase != 1) {
#if defined(QG_FUSION_AUDIT)
        ++later;
#endif
        return estimate;
      }
      s.phase = 2;
#if defined(QG_FUSION_AUDIT)
      ++second;
#endif
      const float sum = s.variance + variance;
      if (sum <= 0 || estimate == s.estimate || !std::isfinite(sum)) {
#if defined(QG_FUSION_AUDIT)
        ++fallback;
#endif
        return estimate;
      }
      const float weight = variance / sum;
      const float mean = weight * s.estimate + (1.f - weight) * estimate;
      const float fused_variance = s.variance * weight;
      const float score =
          mean - kLambda * std::sqrt((2.f / 768.f) * std::max(mean, 0.f) *
                                     fused_variance);
      if (!std::isfinite(score)) return estimate;
      // Disabled mode retains metadata/cache work, but exactly the old queue
      // scores and versions. This is the cost-control ablation.
      if (!enabled_) return estimate;
      s.phase = 3;
#if defined(QG_FUSION_AUDIT)
      ++fused;
#endif
      return score;
    }
#if defined(QG_FUSION_AUDIT)
    ++overflow;
#endif
    // No eviction: old versions remain checkable. Untracked IDs retain all
    // original estimates; a full probe chain remains full until query reset.
    return estimate;
  }
  uint32_t version(uint32_t id) const {
    if (!enabled_) return 0;
    size_t h = hash(id);
    for (size_t p = 0; p < kProbes; ++p, h = (h + 1) & (kSlots - 1)) {
      const Slot &s = slots_[h];
      if (s.epoch != epoch_) return 0;
      if (s.id == id) return uint32_t(s.phase == 3);
    }
    return 0;
  }
  bool valid(uint32_t id, uint32_t queued_version) const {
    return version(id) == queued_version;
  }
  // FP16 encoding runs during prebuild only; callers check F16C support.
  __attribute__((target("f16c"))) static uint16_t encode(float x) {
    return _cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
  // Normal positive halves need just an exponent adjustment; subnormals are
  // uncommon but explicitly supported. No per-edge ISA dispatch/call.
  static float decode(uint16_t x) {
    const uint32_t exponent = (x >> 10) & 31;
    if (exponent == 0)
      return std::ldexp((x & 0x8000) ? -float(x & 1023) : float(x & 1023), -24);
    uint32_t bits = (uint32_t(x & 0x7fff) << 13) + (uint32_t{112} << 23);
    if (exponent == 31) bits = 0x7f800000u | (uint32_t(x & 1023) << 13);
    bits |= uint32_t(x & 0x8000) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  static bool error_scale(const float *center, const float *vector, size_t dim,
                          uint16_t &out) {
    double r2 = 0, l1 = 0;
    for (size_t j = 0; j < dim; ++j) {
      const double r = double(vector[j]) - center[j];
      r2 += r * r;
      l1 += std::abs(r);
    }
    if (!std::isfinite(r2) || !std::isfinite(l1)) return false;
    const double scale = l1 > 0 ? r2 / l1 : 0;
    const double e2 = std::max(0., double(dim) * scale * scale - r2);
    if (!std::isfinite(e2) || e2 > 65504) return false;
    out = encode(float(e2));
    return true;
  }
  template <class Codec>
  static auto supported(const Codec &codec,
                        int) -> decltype(codec.fusion_supported()) {
    return codec.fusion_supported();
  }
  template <class Codec>
  static bool supported(const Codec &, long) {
    return false;
  }

 private:
  std::array<Slot, kSlots> slots_{};
  uint32_t epoch_ = 0;
  bool enabled_ = false;
};
}  // namespace zvec::core
