// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace zvec::core {

// Sorted linear beam mirroring SymphonyQG's SearchBuffer: branchless binary
// search and raw memmove inserts into capacity+1 slots. Expansion state is
// separate from the id so every bit of an HNSW node id is preserved.
class SymphonyQGBeam {
 public:
  explicit SymphonyQGBeam(size_t capacity)
      : capacity_(std::max(size_t{1}, capacity)) {
    entries_.resize(capacity_ + 1);
  }

  void reset(size_t capacity) {
    capacity_ = std::max(size_t{1}, capacity);
    if (entries_.size() < capacity_ + 1) entries_.resize(capacity_ + 1);
    size_ = 0;
    cursor_ = 0;
  }

  // Cheapest rejection first: the caller tests this before touching the visit
  // set or loading the neighbor id.
  bool is_full(float distance) const {
    return size_ == capacity_ && distance > entries_[size_ - 1].distance;
  }

  void insert(uint32_t id, float distance) {
    if (is_full(distance)) return;
    const size_t lo = binary_search(distance);
    std::memmove(&entries_[lo + 1], &entries_[lo],
                 (size_ - lo) * sizeof(Entry));
    entries_[lo] = Entry{id, distance, false};
    size_ += static_cast<size_t>(size_ < capacity_);
    cursor_ = lo < cursor_ ? lo : cursor_;
  }

  bool has_next() const {
    return cursor_ < size_;
  }

  // Whether an index beyond the cursor is still an unexpanded candidate.
  bool has_next_at(size_t ahead) const {
    size_t i = cursor_;
    while (i < size_ && ahead > 0) {
      if (!entries_[i].expanded) --ahead;
      ++i;
    }
    while (i < size_ && entries_[i].expanded) ++i;
    return i < size_;
  }

  uint32_t next_id_at(size_t ahead) const {
    size_t i = cursor_;
    while (i < size_ && ahead > 0) {
      if (!entries_[i].expanded) --ahead;
      ++i;
    }
    while (i < size_ && entries_[i].expanded) ++i;
    return entries_[i < size_ ? i : size_ - 1].id;
  }

  uint32_t pop() {
    Entry &entry = entries_[cursor_++];
    entry.expanded = true;
    const uint32_t id = entry.id;
    while (cursor_ < size_ && entries_[cursor_].expanded) ++cursor_;
    return id;
  }

 private:
  size_t binary_search(float distance) const {
    size_t lo = 0;
    size_t len = size_;
    while (len > 1) {
      const size_t half = len >> 1;
      len -= half;
      lo += static_cast<size_t>(entries_[lo + half - 1].distance < distance) *
            half;
    }
    return (lo < size_ && entries_[lo].distance < distance) ? lo + 1 : lo;
  }

  struct Entry {
    uint32_t id;
    float distance;
    bool expanded;
  };
  size_t capacity_;
  size_t size_{0};
  size_t cursor_{0};
  std::vector<Entry> entries_;
};

}  // namespace zvec::core
