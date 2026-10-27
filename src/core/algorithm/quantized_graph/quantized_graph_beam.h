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
// search and raw memmove inserts into capacity+1 slots. Expansion flags share
// the id word until a full-width id is inserted; that query then moves the
// flags to a separate array without sacrificing any bits of a node id.
class QuantizedGraphBeam {
 public:
  explicit QuantizedGraphBeam(size_t capacity)
      : capacity_(std::max(size_t{1}, capacity)) {
    entries_.resize(capacity_ + 1);
  }

  void reset(size_t capacity) {
    capacity_ = std::max(size_t{1}, capacity);
    if (entries_.size() < capacity_ + 1) entries_.resize(capacity_ + 1);
    full_width_ = false;
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
    if (!full_width_ && (id & kExpanded)) {
      expanded_.resize(capacity_ + 1);
      for (size_t i = 0; i < size_; ++i) {
        expanded_[i] = (entries_[i].id & kExpanded) != 0;
        entries_[i].id &= ~kExpanded;
      }
      full_width_ = true;
    }
    const size_t lo = binary_search(distance);
    std::memmove(&entries_[lo + 1], &entries_[lo],
                 (size_ - lo) * sizeof(Entry));
    if (full_width_) {
      std::memmove(&expanded_[lo + 1], &expanded_[lo], size_ - lo);
      expanded_[lo] = false;
    }
    entries_[lo] = Entry{id, distance};
    size_ += static_cast<size_t>(size_ < capacity_);
    cursor_ = lo < cursor_ ? lo : cursor_;
  }

  // Experimental beam over unique vertices; repeated edge estimates can
  // improve a candidate's key, but cannot consume multiple beam slots.
  void insert_unique(uint32_t id, float distance) {
    for (size_t i = 0; i < size_; ++i) {
      if (node_id(i) != id) continue;
      if (entries_[i].distance <= distance) return;
      std::memmove(&entries_[i], &entries_[i + 1],
                   (size_ - i - 1) * sizeof(Entry));
      if (full_width_)
        std::memmove(&expanded_[i], &expanded_[i + 1], size_ - i - 1);
      --size_;
      if (i < cursor_) --cursor_;
      while (cursor_ < size_ && expanded(cursor_)) ++cursor_;
      break;
    }
    insert(id, distance);
  }

  bool has_next() const {
    return cursor_ < size_;
  }

  // Whether an index beyond the cursor is still an unexpanded candidate.
  bool has_next_at(size_t ahead) const {
    size_t i = cursor_;
    while (i < size_ && ahead > 0) {
      if (!expanded(i)) --ahead;
      ++i;
    }
    while (i < size_ && expanded(i)) ++i;
    return i < size_;
  }

  uint32_t next_id_at(size_t ahead) const {
    size_t i = cursor_;
    while (i < size_ && ahead > 0) {
      if (!expanded(i)) --ahead;
      ++i;
    }
    while (i < size_ && expanded(i)) ++i;
    return node_id(i < size_ ? i : size_ - 1);
  }

  float next_distance() const {
    return entries_[cursor_].distance;
  }

  uint32_t pop() {
    const uint32_t id = node_id(cursor_);
    if (full_width_)
      expanded_[cursor_] = true;
    else
      entries_[cursor_].id |= kExpanded;
    ++cursor_;
    while (cursor_ < size_ && expanded(cursor_)) ++cursor_;
    return id;
  }

 private:
  static constexpr uint32_t kExpanded = uint32_t{1} << 31;
  bool expanded(size_t i) const {
    return full_width_ ? expanded_[i] != 0 : (entries_[i].id & kExpanded) != 0;
  }
  uint32_t node_id(size_t i) const {
    return full_width_ ? entries_[i].id : entries_[i].id & ~kExpanded;
  }
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
  };
  static_assert(sizeof(Entry) == 8, "Beam entries must remain compact");
  size_t capacity_;
  size_t size_{0};
  size_t cursor_{0};
  bool full_width_{false};
  std::vector<Entry> entries_;
  std::vector<uint8_t> expanded_;
};

}  // namespace zvec::core
