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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_storage.h>
#include "algorithm/quantized_graph/quantized_graph_build_options.h"

namespace zvec::core {
class HnswEntity;
class HnswContext;

// Immutable projected-entry SymphonyQG snapshot over HNSW level zero.
class HnswQuantizedGraph {
 public:
  static constexpr const char *kSnapshotSegment = "hnsw.quantized_graph.blocks";
  HnswQuantizedGraph(size_t dimension, size_t max_neighbors, bool cosine,
                     size_t stored_vector_bytes, const std::string &model_data,
                     bool fp16 = false,
                     const std::string *codec_data = nullptr);
  static int train_codec(const HnswEntity &entity, size_t dimension,
                         bool cosine, bool fp16,
                         const QuantizedGraphBuildOptions &options,
                         std::string &data);
  static int load(const HnswEntity &entity, size_t dimension, bool cosine,
                  size_t stored_vector_bytes, const std::string &router_data,
                  bool fp16, const IndexStorage::Segment::Pointer &segment,
                  std::shared_ptr<HnswQuantizedGraph> &out);
  int dump(const IndexDumper::Pointer &dumper) const;
  static bool supported();
  static int train_router(const HnswEntity &entity, size_t dimension,
                          bool cosine, size_t anchors, size_t projected,
                          uint64_t seed, std::string &data, bool fp16 = false);
  ~HnswQuantizedGraph();
  int search(uint32_t entry, HnswContext &ctx) const;
  int prebuild(const HnswEntity &entity, size_t doc_cnt, size_t threads);
  uint32_t entry() const;

 private:
  struct Impl;
  template <bool Fp16>
  struct Model;
  std::unique_ptr<Impl> impl_;
};
}  // namespace zvec::core
