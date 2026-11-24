// Copyright 2026-present the zvec project
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

#include <stdexcept>
#include "algorithm/quantized_graph/quantized_graph_builder.h"
#include "algorithm/quantized_graph/quantized_graph_params.h"
#include <yaml-cpp/yaml.h>

namespace zvec::core::qg_bench_options {

inline ailego::Params page_params(const YAML::Node &params) {
  ailego::Params out;
  if (params && params[quantized_graph_pages::kPolicy]) {
    out.set(quantized_graph_pages::kPolicy,
            params[quantized_graph_pages::kPolicy].as<std::string>());
  }
  for (const auto &key :
       {quantized_graph_pages::kMinimum, quantized_graph_pages::kFloor}) {
    if (params && params[key]) out.set(key, params[key].as<double>());
  }
  return out;
}

inline QuantizedGraphBuilder::Options build_options(const YAML::Node &profile) {
  const auto b = profile["build"];
  QuantizedGraphBuilder::Options o;
  o.threads = b["threads"].as<size_t>();
  o.passes = b["passes"].as<size_t>();
  o.search_list_size = b["search_list_size"].as<size_t>();
  o.seed = b["seed"].as<uint32_t>();
  o.router_anchors = b["router_anchors"].as<size_t>();
  o.router_dimension = b["router_dimension"].as<size_t>();
  o.router_seed = b["router_seed"].as<uint64_t>();
  o.page_params = page_params(profile["page_params"]);

  const auto e = b["encoding"];
  using Encoding = QuantizedGraphEncodingOptions;
  auto &enc = o.encoding;
  const auto codec = e["codec"].as<std::string>();
  if (!Encoding::parse_name(codec, enc.codec))
    throw std::invalid_argument("unknown QG codec: " + codec);

  const auto book = e["codebook"].as<std::string>();
  if (!Encoding::parse_name(book, enc.codebook))
    throw std::invalid_argument("unknown QG codebook: " + book);
  enc.max_neighbors = e["max_neighbors"].as<size_t>();
  enc.rotation_views = e["rotation_views"].as<size_t>();
  enc.block_dimension = e["block_dimension"].as<size_t>();
  enc.rank = e["rank"].as<size_t>();
  enc.training_samples = e["training_samples"].as<size_t>();
  enc.center_weight = e["center_weight"].as<double>();
  enc.seed = e["seed"].as<uint32_t>();
  return o;
}

}  // namespace zvec::core::qg_bench_options
