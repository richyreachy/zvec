// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <zvec/core/framework/index_builder.h>
#include "quantized_graph_index.h"

namespace zvec::core {

class QuantizedGraphIndex;

// Owned FP32 rows and single-layer adjacency used by QuantizedGraphIndex.
struct QuantizedGraphData {
  struct Vector {
    const float *values = nullptr;
    const void *data() const {
      return values;
    }
  };
  struct Neighbors {
    const uint32_t *values;
    size_t count;
    size_t size() const {
      return count;
    }
    uint32_t operator[](size_t i) const {
      return values[i];
    }
  };

  size_t dimension = 0;
  bool cosine = false;
  std::vector<float> vectors;
  std::vector<uint64_t> offsets;
  std::vector<uint32_t> edges;

  size_t size() const {
    return dimension ? vectors.size() / dimension : 0;
  }
  bool valid(uint32_t id) const {
    return id < size();
  }
  int get_vector(uint32_t id, Vector &out) const {
    if (!valid(id)) return IndexError_NoExist;
    out.values = vectors.data() + size_t(id) * dimension;
    return 0;
  }
  Neighbors neighbors(uint32_t id) const {
    const auto count = size_t(offsets[id + 1] - offsets[id]);
    return {count ? edges.data() + offsets[id] : nullptr, count};
  }
};

// Builds the topology, codec, router and quantized graph directly from FP32
// rows.
class QuantizedGraphBuilder final : public IndexBuilder {
 public:
  struct Options {
    QuantizedGraphBuildOptions encoding;
    size_t search_list_size = 400;
    size_t threads = 2;
    size_t passes = 2;
    uint32_t seed = 20261120;
    size_t router_anchors = 65536;
    size_t router_dimension = 64;
    uint64_t router_seed = 202610261259;
    ailego::Params page_params;

    bool valid() const {
      return encoding.valid() && passes > 0 && passes <= 8 && threads > 0 &&
             threads <= 256 && search_list_size >= encoding.max_neighbors &&
             search_list_size <= (1u << 20) && (!router_anchors || router_dimension);
    }
  };

  struct Result {
    std::shared_ptr<const QuantizedGraphData> data;
    std::shared_ptr<QuantizedGraphIndex> index;
    std::string codec;
    std::string router;
  };

  // Takes ownership of the input allocation when called with std::move.
  // Cosine input is normalized once here; query normalization is caller-owned.
  // IDs are input row ordinals. No queries or ground truth enter construction.
  // On error, result retains its previous value.
  static int build(std::vector<float> vectors, size_t dimension, bool cosine,
                   const Options &options, Result &result);

  using IndexBuilder::build;
  using IndexBuilder::train;
  int init(const IndexMeta &meta, const ailego::Params &params) override;
  int train(IndexThreads::Pointer, IndexHolder::Pointer holder) override;
  int build(IndexThreads::Pointer, IndexHolder::Pointer holder) override;
  int dump(const IndexDumper::Pointer &dumper) override;
  int dump(IndexStorage &storage);
  int cleanup() override;
  const Stats &stats() const override {
    return stats_;
  }

 private:
  bool compatible(const IndexHolder::Pointer &holder) const;
  template <class Part>
  int serialize(Part part);
  Stats stats_;
  IndexMeta meta_;
  Options options_;
  Result built_;
  std::vector<uint64_t> keys_;
  bool initialized_ = false;
};

}  // namespace zvec::core
