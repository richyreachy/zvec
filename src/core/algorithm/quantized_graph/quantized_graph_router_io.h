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
#include <cstring>
#include <string_view>

namespace zvec::core {

// Bounded byte reader for embedded index segments.
struct QuantizedGraphRouterReader {
  std::string_view data;
  size_t offset = 0;

  bool operator()(void *out, size_t bytes) {
    if (bytes > data.size() - offset) return false;

    std::memcpy(out, data.data() + offset, bytes);
    offset += bytes;
    return true;
  }
};

inline constexpr const char *kQuantizedGraphRouterSegment =
    "hnsw.quantized_graph.router";
inline constexpr size_t kMaxQuantizedGraphRouterBytes = size_t{1} << 30;

}  // namespace zvec::core
