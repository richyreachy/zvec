#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
namespace zvec::core {
// Research queue: approximate ordering, conservative bucket score for stopping.
// The range is scaled once by the exact entry distance. Extreme bins retain
// every candidate; no item is dropped. Normal-bin ordering error <= width.
class QuantizedGraphBucket {
 public:
  static constexpr size_t N = 2048;
#if defined(QG_QUEUE_STATS)
  size_t count = 0, peak = 0;
  size_t memory_bytes() const {
    size_t n = sizeof(*this) + touched_.capacity() * sizeof(size_t);
    for (const auto &b : bins_) n += b.capacity() * sizeof(uint32_t);
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
  }
  void set_scale(float distance) {
    width_ = std::max(distance, 1e-6f) * QG_BUCKET_WIDTH;
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
  void insert(uint32_t id, float d) {
    size_t b = locate(d);
    if (bins_[b].empty()) touched_.push_back(b);
    bins_[b].push_back(id);
#if defined(QG_QUEUE_STATS)
    ++count;
    peak = std::max(peak, count);
#endif
    bits_[b / 64] |= uint64_t{1} << (b % 64);
    summary_ |= uint64_t{1} << (b / 64);
  }
  bool has_next() const {
    return summary_ != 0;
  }
  size_t first() const {
    size_t w = __builtin_ctzll(summary_);
    return 64 * w + __builtin_ctzll(bits_[w]);
  }
  uint32_t next_id_at(size_t) const {
    return bins_[first()].back();
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
    }
    return id;
  }
  bool is_full(float) const {
    return false;
  }

 private:
  std::array<std::vector<uint32_t>, N> bins_;
  std::array<uint64_t, N / 64> bits_{};
  uint64_t summary_ = 0;
  std::vector<size_t> touched_;
  float width_ = 1;
};
}  // namespace zvec::core
