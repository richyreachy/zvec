#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace zvec::core {
#ifndef QG_BUCKET_COUNT
#define QG_BUCKET_COUNT 2048
#endif
#if !defined(QG_FLAT_BUCKET)
// Research queue: approximate ordering, conservative bucket score for stopping.
// The range is scaled once by the exact entry distance. Extreme bins retain
// every candidate; no item is dropped. Normal-bin ordering error <= width.
class QuantizedGraphBucket {
 public:
  static constexpr size_t N = QG_BUCKET_COUNT;
  static_assert(N >= 64 && N <= 4096 && N % 64 == 0);
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
  std::array<std::vector<uint32_t>, N> bins_;
  std::array<uint64_t, N / 64> bits_{};
  uint64_t summary_ = 0;
  std::vector<size_t> touched_;
  float width_ = 1;
#if defined(QG_BUCKET_CACHEMIN)
  size_t first_ = N;
#endif
};
#else
// Same LIFO buckets, but all entries live in one append-only per-query arena.
class QuantizedGraphBucket {
 public:
  static constexpr size_t N = QG_BUCKET_COUNT;
  static_assert(N >= 64 && N <= 4096 && N % 64 == 0);
  static constexpr uint32_t empty = UINT32_MAX;
  QuantizedGraphBucket() {
    heads_.fill(empty);
  }
#if defined(QG_QUEUE_STATS)
  size_t count = 0, peak = 0;
  size_t memory_bytes() const {
    return sizeof(*this) + entries_.capacity() * sizeof(Entry) +
           touched_.capacity() * sizeof(size_t);
  }
#endif
  void reset(size_t) {
    for (size_t b : touched_) heads_[b] = empty;
    touched_.clear();
    entries_.clear();
    bits_.fill(0);
    summary_ = 0;
    width_ = 1.f;
#if defined(QG_QUEUE_STATS)
    count = peak = 0;
#endif
  }
  void set_scale(float distance) {
    width_ = std::max(distance, 1e-6f) * QG_BUCKET_WIDTH;
  }
  float width() const {
    return width_;
  }
  size_t locate(float d) const {
    if (d < 0) return 0;
    float x = d / width_;
    return x >= float(N - 2) ? N - 1 : size_t(x) + 1;
  }
  float lower_bound(float d) const {
    size_t b = locate(d);
    return b ? float(b - 1) * width_ : -std::numeric_limits<float>::infinity();
  }
  void insert(uint32_t id, float d) {
    size_t b = locate(d);
    if (heads_[b] == empty) touched_.push_back(b);
    // At most O(number of graph edges) records; our 1M static graph is well
    // below the uint32 offset limit. Fail explicitly rather than wrap offsets.
    if (entries_.size() >= empty)
      throw std::length_error("bucket arena offset overflow");
    const uint32_t index = uint32_t(entries_.size());
    entries_.push_back({id, heads_[b]});
    heads_[b] = index;
    bits_[b / 64] |= uint64_t{1} << (b % 64);
    summary_ |= uint64_t{1} << (b / 64);
#if defined(QG_QUEUE_STATS)
    ++count;
    peak = std::max(count, peak);
#endif
  }
  bool has_next() const {
    return summary_ != 0;
  }
  size_t first() const {
    size_t w = __builtin_ctzll(summary_);
    return 64 * w + __builtin_ctzll(bits_[w]);
  }
  uint32_t next_id_at(size_t) const {
    return entries_[heads_[first()]].id;
  }
  float next_distance() const {
    size_t b = first();
    return b ? float(b - 1) * width_ : -std::numeric_limits<float>::infinity();
  }
  uint32_t pop() {
    size_t b = first();
    Entry e = entries_[heads_[b]];
    heads_[b] = e.next;
    if (e.next == empty) {
      bits_[b / 64] &= ~(uint64_t{1} << (b % 64));
      if (!bits_[b / 64]) summary_ &= ~(uint64_t{1} << (b / 64));
    }
#if defined(QG_QUEUE_STATS)
    --count;
#endif
    return e.id;
  }
  bool is_full(float) const {
    return false;
  }

 private:
  struct Entry {
    uint32_t id, next;
  };
  std::array<uint32_t, N> heads_;
  std::array<uint64_t, N / 64> bits_{};
  uint64_t summary_ = 0;
  std::vector<Entry> entries_;
  std::vector<size_t> touched_;
  float width_ = 1;
};
#endif
// Discard only strictly worse bucket proposals for an unexpanded node.
// Equal-bin proposals remain: the newest LIFO position can affect traversal.
class QuantizedGraphBucketDominance {
 public:
  void reset() {
    if (table_.empty()) table_.resize(1024);
    for (auto &e : table_) e.id = UINT32_MAX;
    size_ = 0;
  }
  bool admit(uint32_t id, uint32_t bin) {
    size_t i = hash(id) & (table_.size() - 1);
    while (table_[i].id != UINT32_MAX && table_[i].id != id)
      i = (i + 1) & (table_.size() - 1);
    if (table_[i].id == id) {
      if (bin > table_[i].bin) return false;
      table_[i].bin = bin;
      return true;
    }
    if (2 * (size_ + 1) >= table_.size()) {
      grow();
      return admit(id, bin);
    }
    table_[i] = {id, bin};
    ++size_;
    return true;
  }

 private:
  struct Entry {
    uint32_t id = UINT32_MAX, bin = 0;
  };
  static uint32_t hash(uint32_t id) {
    uint32_t h = id * 2654435761U;
    return h ^ (h >> 16);
  }
  void grow() {
    auto old = std::move(table_);
    table_.resize(old.size() * 2);
    size_ = 0;
    for (const auto &e : old)
      if (e.id != UINT32_MAX) admit(e.id, e.bin);
  }
  std::vector<Entry> table_;
  size_t size_ = 0;
};
// Direct-mapped memoization may forget a prior proposal on collision. Such a
// miss only permits extra queue work; a matching full ID still gives a valid
// dominance witness. Never reject based on a fingerprint alone.
class QuantizedGraphDominanceCache {
 public:
  void reset() {
    for (auto &e : table_) e.id = UINT32_MAX;
  }
  bool admit(uint32_t id, uint32_t bin) {
    auto &e = table_[id & 255U];
    if (e.id == id && bin > e.bin) return false;
    e = {id, bin};
    return true;
  }

 private:
  struct Entry {
    uint32_t id = UINT32_MAX, bin = 0;
  };
  std::array<Entry, 256> table_;
};
// A single dictionary serves expansion visits and best-bucket proposals.
// The last lookup is cached across visited(id) -> admit(id) / set_visited(id).
class QuantizedGraphSharedVisit {
 public:
  void reset(size_t) {
    if (table_.empty()) table_.resize(1024);
    for (auto &e : table_) e.id = UINT32_MAX;
    size_ = 0;
    cached_id_ = UINT32_MAX;
  }
  bool visited(uint32_t id) const {
    size_t i = find(id);
    return table_[i].id == id && (table_[i].marker & expanded_flag);
  }
  void set_visited(uint32_t id) {
    table_[ensure(id)].marker |= expanded_flag;
  }
  bool admit(uint32_t id, uint32_t bin) {
    Entry &e = table_[ensure(id)];
    if ((e.marker & expanded_flag) || bin > e.marker) return false;
    e.marker = bin;
    return true;
  }

 private:
  static constexpr uint32_t expanded_flag = uint32_t{1} << 31;
  struct Entry {
    uint32_t id = UINT32_MAX, marker = expanded_flag - 1;
  };
  static uint32_t hash(uint32_t id) {
    uint32_t h = id * 2654435761U;
    return h ^ (h >> 16);
  }
  size_t find(uint32_t id) const {
    if (cached_id_ == id) return cached_index_;
    size_t i = hash(id) & (table_.size() - 1);
    while (table_[i].id != UINT32_MAX && table_[i].id != id)
      i = (i + 1) & (table_.size() - 1);
    cached_id_ = id;
    cached_index_ = i;
    return i;
  }
  size_t ensure(uint32_t id) {
    size_t i = find(id);
    if (table_[i].id == id) return i;
    if (2 * (size_ + 1) >= table_.size()) {
      grow();
      i = find(id);
    }
    table_[i] = {id, expanded_flag - 1};
    ++size_;
    return i;
  }
  void grow() {
    auto old = std::move(table_);
    table_.resize(old.size() * 2);
    size_ = 0;
    cached_id_ = UINT32_MAX;
    for (const auto &e : old)
      if (e.id != UINT32_MAX) {
        size_t i = ensure(e.id);
        table_[i] = e;
      }
    cached_id_ = UINT32_MAX;
  }
  std::vector<Entry> table_;
  size_t size_ = 0;
  mutable uint32_t cached_id_ = UINT32_MAX;
  mutable size_t cached_index_ = 0;
};
}  // namespace zvec::core
