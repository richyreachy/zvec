// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
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
