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

#include <string>
#include <zvec/ailego/container/params.h>

namespace zvec::core::qg_build_params {

inline const std::string kPasses =
    "zvec.quantized_graph.builder.passes";
inline const std::string kThreads =
    "zvec.quantized_graph.builder.threads";
inline const std::string kSearchListSize =
    "zvec.quantized_graph.builder.search_list_size";
inline const std::string kSeed =
    "zvec.quantized_graph.builder.seed";
inline const std::string kRouterSeed =
    "zvec.quantized_graph.builder.router_seed";
inline const std::string kRouterDimension =
    "zvec.quantized_graph.builder.router_dimension";
inline const std::string kRouterAnchors =
    "zvec.quantized_graph.builder.router_anchors";
inline const std::string kEncodingCodec =
    "zvec.quantized_graph.builder.encoding.codec";
inline const std::string kEncodingCodebook =
    "zvec.quantized_graph.builder.encoding.codebook";
inline const std::string kEncodingMaxNeighbors =
    "zvec.quantized_graph.builder.encoding.max_neighbors";
inline const std::string kEncodingRank =
    "zvec.quantized_graph.builder.encoding.rank";
inline const std::string kEncodingRotationViews =
    "zvec.quantized_graph.builder.encoding.rotation_views";
inline const std::string kEncodingBlockDimension =
    "zvec.quantized_graph.builder.encoding.block_dimension";
inline const std::string kEncodingTrainingSamples =
    "zvec.quantized_graph.builder.encoding.training_samples";
inline const std::string kEncodingCenterWeight =
    "zvec.quantized_graph.builder.encoding.center_weight";
inline const std::string kEncodingSeed =
    "zvec.quantized_graph.builder.encoding.seed";

template <class Options, class Field>
void build_fields(Options &options, Field field) {
  field(kPasses, options.passes);
  field(kThreads, options.threads);
  field(kSearchListSize, options.search_list_size);
  field(kSeed, options.seed);
  field(kRouterSeed, options.router_seed);
  field(kRouterDimension, options.router_dimension);
  field(kRouterAnchors, options.router_anchors);
}

template <class Options, class Field>
void encoding_fields(Options &options, Field field) {
  field(kEncodingCodec, options.codec);
  field(kEncodingCodebook, options.codebook);
  field(kEncodingMaxNeighbors, options.max_neighbors);
  field(kEncodingRank, options.rank);
  field(kEncodingRotationViews, options.rotation_views);
  field(kEncodingBlockDimension, options.block_dimension);
  field(kEncodingTrainingSamples, options.training_samples);
  field(kEncodingCenterWeight, options.center_weight);
  field(kEncodingSeed, options.seed);
}

}  // namespace zvec::core::qg_build_params

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
