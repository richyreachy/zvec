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

#include <cstdint>
#include <cstdio>
#include "turbo/preprocessor/quantized_graph_rotator/quantized_graph_rotator.h"
#include "quantized_graph_router_io.h"

namespace zvec::core {

// V2: fixed 768D cosine. V3 additionally records original dimension, metric,
// rotation dimension and SIMD code stride. The projection is never the graph
// dimension. V4 also records independent rotation views. Metric 0 is cosine,
// 1 is squared L2. V2/V3 always use the historical single-view transform.
struct QuantizedGraphRouterFormat {
  uint32_t version = 3, projected = 384, groups = 0, docs = 0;
  uint32_t dimension = 768, metric = 0, rotated = 768, stride = 384;
  uint32_t rotation_views = 1;

  static constexpr uint32_t kMagic = 0x51475259, kTransform = 0x53514701;
  static constexpr uint32_t kMaxDimension = 4096;

  bool valid() const {
    if ((version != 2 && version != 3 && version != 4) || dimension == 0 ||
        dimension > kMaxDimension || metric > 1 || projected == 0 ||
        projected > rotated || stride != ((projected + 63u) & ~63u) ||
        groups == 0 || groups > 100000 || docs == 0 || docs > INT32_MAX)
      return false;
    if (!turbo::QuantizedGraphRotator::valid_views(rotation_views) ||
        (version != 4 && rotation_views != 1))
      return false;

    turbo::QuantizedGraphRotator rotation(dimension, rotation_views);
    return rotated == rotation.padded_dim() &&
           (version != 2 ||
            (dimension == 768 && metric == 0 && projected == 384));
  }

  bool read(FILE *f) {
    auto reader = [f](void *p, size_t n) {
      return f && std::fread(p, 1, n, f) == n;
    };

    return read_bytes(reader);
  }

  template <class Reader>
  bool read_bytes(Reader &reader) {
    uint32_t h[5];
    if (!reader(h, sizeof(h)) || h[0] != kMagic) return false;

    version = h[1];
    projected = h[2];
    groups = h[3];
    docs = h[4];
    rotation_views = 1;

    if (version == 3 || version == 4) {
      uint32_t extra[4];
      if (!reader(extra, sizeof(extra))) return false;
      dimension = extra[0];
      metric = extra[1];
      rotated = extra[2];
      stride = extra[3];
      if (version == 4 && !reader(&rotation_views, sizeof(rotation_views)))
        return false;
    } else if (version == 2) {
      dimension = rotated = 768;
      metric = 0;
      stride = 384;
    } else
      return false;

    return valid();
  }
};

}  // namespace zvec::core
