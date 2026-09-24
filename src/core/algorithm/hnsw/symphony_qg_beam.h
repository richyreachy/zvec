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
#include <vector>

namespace zvec::core {

// Sorted bounded beam. Keep the expanded flag separate from the ID so the
// whole HNSW uint32_t ID range remains usable. Estimates are local to an edge,
// so duplicate IDs with different estimates are intentional (visited on pop).
class SymphonyQGBeam {
 public:
  explicit SymphonyQGBeam(size_t capacity)
      : capacity_(std::max(size_t{1}, capacity)) {
    entries_.reserve(capacity_ + 1);
  }

  void insert(uint32_t id, float distance) {
    if (entries_.size() == capacity_ && distance > entries_.back().distance)
      return;
    auto it = std::lower_bound(
        entries_.begin(), entries_.end(), distance,
        [](const Entry &entry, float value) { return entry.distance < value; });
    const size_t pos = it - entries_.begin();
    entries_.insert(it, Entry{id, distance, false});
    if (entries_.size() > capacity_) entries_.pop_back();
    cursor_ = std::min(cursor_, pos);
  }

  bool has_next() const {
    return cursor_ < entries_.size();
  }

  uint32_t pop() {
    auto &entry = entries_[cursor_++];
    entry.expanded = true;
    const uint32_t id = entry.id;
    while (has_next() && entries_[cursor_].expanded) ++cursor_;
    return id;
  }

 private:
  struct Entry {
    uint32_t id;
    float distance;
    bool expanded;
  };
  size_t capacity_;
  size_t cursor_{0};
  std::vector<Entry> entries_;
};

}  // namespace zvec::core
