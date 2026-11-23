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
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <zvec/ailego/container/heap.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_metric.h>
#include "quantized_graph_options.h"

namespace zvec::core {

// Each concurrent query needs its own context and normalized cosine input.
class QuantizedGraphQuery {
 public:
  using Heap = ailego::KeyValueHeap<uint32_t, float>;
  using Filter =
      std::function<bool(uint32_t)>;  // true excludes an internal ID.

  QuantizedGraphQuery(size_t dimension, IndexMetric::MatrixDistance distance,
                      size_t distance_dimension = 0)
      : dimension_(dimension),
        distance_dimension_(distance_dimension ? distance_dimension
                                               : dimension),
        distance_(std::move(distance)) {}

  int reset(const float *query, uint32_t topk, size_t capacity,
            const QuantizedGraphSearchOptions &options = {},
            uint64_t scan_limit = 0, Filter filter = {}) {
    query_ = nullptr;
    topk_ = 0;
    heap_.clear();
    comparisons_ = expanded_ = 0;
    if (!query || !distance_ || !dimension_ || dimension_ > 4096 ||
        !std::isfinite(options.margin) ||
        (options.margin < 0 && options.margin != -1) ||
        !std::isfinite(options.bucket_width) || options.bucket_width < 0 ||
        options.bucket_width > 1 || options.route_beam > 4 ||
        !std::isfinite(options.uncertainty_bonus) ||
        options.uncertainty_bonus < 0 || options.uncertainty_bonus > 1)
      return IndexError_InvalidArgument;

    query_ = query;
    topk_ = topk;
    capacity_ = std::max({size_t{1}, size_t(topk), capacity});
    options_ = options;
    scan_limit_ = scan_limit;
    filter_ = std::move(filter);
    return 0;
  }

  Heap &reset_results() {
    heap_.clear();
    heap_.limit(capacity_);
    comparisons_ = expanded_ = 0;
    return heap_;
  }

  // Sorted internal IDs and exact scores; callers may map IDs to external keys.
  const Heap &results() {
    heap_.sort();
    return heap_;
  }

  uint32_t topk() const {
    return topk_;
  }
  const void *query() const {
    return query_;
  }

  const QuantizedGraphSearchOptions &quantized_graph_options() const {
    return options_;
  }

  float distance(const void *vector) {
    ++comparisons_;
    float score = 0;
    distance_(vector, query_, distance_dimension_, &score);
    return score;
  }

  bool excluded(uint32_t id) const {
    return filter_ && filter_(id);
  }

  bool reach_scan_limit() const {
    return scan_limit_ && comparisons_ >= scan_limit_;
  }

  void on_expand() {
    ++expanded_;
  }
  uint64_t comparisons() const {
    return comparisons_;
  }
  uint64_t expanded() const {
    return expanded_;
  }

 private:
  size_t dimension_, distance_dimension_, capacity_ = 1;
  IndexMetric::MatrixDistance distance_;
  const float *query_ = nullptr;
  uint32_t topk_ = 0;
  QuantizedGraphSearchOptions options_;
  uint64_t scan_limit_ = 0, comparisons_ = 0, expanded_ = 0;
  Filter filter_;
  Heap heap_;
};

}  // namespace zvec::core
