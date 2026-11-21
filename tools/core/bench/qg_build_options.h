// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <stdexcept>
#include <string>
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

inline QuantizedGraphBuildOptions encoding_options(const YAML::Node &e,
                                                  bool legacy = false) {
  QuantizedGraphBuildOptions enc;
  const auto codec = e["codec"].as<std::string>();
  using Codec = QuantizedGraphBuildOptions::Codec;
  if (codec == "rabitq" || (legacy && codec == "native"))
    enc.codec = Codec::RaBitQ;
  else if (codec == "global")
    enc.codec = Codec::Global;
  else if (codec == "low_rank")
    enc.codec = Codec::LowRank;
  else
    throw std::invalid_argument("unknown QG codec: " + codec);
  const auto book = e["codebook"].as<std::string>();
  using Codebook = QuantizedGraphBuildOptions::Codebook;
  if (book == "hadamard")
    enc.codebook = Codebook::Hadamard;
  else if (book == "gaussian")
    enc.codebook = Codebook::Gaussian;
  else
    throw std::invalid_argument("unknown QG codebook: " + book);
  enc.max_neighbors = e["max_neighbors"].as<size_t>();
  enc.rotation_views = e["rotation_views"].as<size_t>();
  enc.block_dimension = e["block_dimension"].as<size_t>();
  enc.rank = e["rank"].as<size_t>();
  enc.training_samples = e["training_samples"].as<size_t>();
  enc.center_weight = e["center_weight"].as<double>();
  enc.seed = e["seed"].as<uint32_t>();
  return enc;
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

  o.encoding = encoding_options(b["encoding"]);
  return o;
}

}  // namespace zvec::core::qg_bench_options
