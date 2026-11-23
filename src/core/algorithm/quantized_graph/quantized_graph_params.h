// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <string>
#include <zvec/ailego/container/params.h>

namespace zvec::core::quantized_graph_pages {

inline const std::string kPolicy = "proxima.quantized_graph.thp_policy";
inline const std::string kMinimum = "proxima.quantized_graph.thp_min_coverage";
inline const std::string kFloor =
    "proxima.quantized_graph.thp_min_coverage_floor";

inline std::string parameter_key(const ailego::Params &params,
                                 const std::string &key) {
  return params.has(key) ? key : "proxima.hnsw." + key.substr(8);
}

}  // namespace zvec::core::quantized_graph_pages
