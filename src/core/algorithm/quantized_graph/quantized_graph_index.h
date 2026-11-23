// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_storage.h>
#include "quantized_graph_build_options.h"

#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#define ZVEC_QUANTIZED_GRAPH_SUPPORTED 1
#include <turbo/norm/normalizer.h>
#include <turbo/quantizer/quantized_graph_quantizer/quantized_graph_quantizer.h>
#include "quantized_graph.h"
#include "quantized_graph_codec_builder.h"
#include "quantized_graph_router_builder.h"
#else
#define ZVEC_QUANTIZED_GRAPH_SUPPORTED 0
#endif

namespace zvec::core {

inline bool QuantizedGraphSupportsMeta(const IndexMeta &m) {
  return m.meta_type() == IndexMeta::MT_DENSE &&
         m.data_type() == IndexMeta::DT_FP32 && m.dimension() &&
         m.dimension() <= 4096 &&
         m.element_size() == size_t(m.dimension()) * sizeof(float) &&
         (m.metric_name() == "Cosine" || m.metric_name() == "SquaredEuclidean");
}

inline bool QuantizedGraphSupported() {
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
  return __builtin_cpu_supports("avx512f") &&
         __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512vl") &&
         __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
namespace quantized_graph_detail {
// Version 1 binds the Block layout and codec packing. Empty/RaBitQ and V1
// codec payloads use one rotation; V2 payloads carry explicit rotation views.
// All fields are little-endian uint64_t on the supported x86-64 platform.
// Followed by codec bytes, graph {stride, count, entry}, then the raw arena.
// Router bytes live in their existing segment; bind them by size and CRC.
using SnapshotHeader = std::array<uint64_t, 10>;
constexpr uint64_t kSnapshotMagic = 0x3150414e53475153ULL;
struct SnapshotReader {
  IndexStorage::Segment::Pointer segment;
  size_t offset = 0;
  uint32_t crc = 0;
  size_t remaining() const {
    return segment->data_size() - offset;
  }
  bool operator()(void *data, size_t bytes) {
    if (bytes > remaining()) return false;
    auto *out = static_cast<char *>(data);
    while (bytes) {
      const size_t chunk = std::min(bytes, size_t{1} << 20);
      IndexStorage::MemoryBlock block;
      if (segment->read(offset, block, chunk) != chunk || !block.data())
        return false;
      std::memcpy(out, block.data(), chunk);
      crc = ailego::Crc32c::Hash(out, chunk, crc);
      segment->release_read_cache(offset, chunk);
      offset += chunk;
      out += chunk;
      bytes -= chunk;
    }
    return true;
  }
};
struct SnapshotWriter {
  IndexDumper::Pointer dumper;
  size_t bytes = 0;
  uint32_t crc = 0;
  bool operator()(const void *data, size_t length) {
    const auto *input = static_cast<const char *>(data);
    while (length) {
      const size_t chunk = std::min(length, size_t{1} << 20);
      if (dumper->write(input, chunk) != chunk) return false;
      crc = ailego::Crc32c::Hash(input, chunk, crc);
      bytes += chunk;
      input += chunk;
      length -= chunk;
    }
    return true;
  }
};

// Training sees FP32 coordinates even when the provider stores FP16. Search
// uses the original provider and its exact metric; no full corpus copy is made.
template <class Graph>
struct DecodedGraph {
  const Graph &source;
  size_t dimension;

  struct Vector {
    typename Graph::Vector storage;
    std::array<float, 4096> values;

    const void *data() const {
      return values.data();
    }
  };

  bool valid(uint32_t id) const {
    return source.valid(id);
  }
  auto neighbors(uint32_t id) const {
    return source.neighbors(id);
  }

  int get_vector(uint32_t id, Vector &vector) const {
    const int ret = source.get_vector(id, vector.storage);
    if (ret != 0) return ret;
    if (!vector.storage.data()) return IndexError_Mismatch;

    ailego::FloatHelper::ToFP32(
        static_cast<const uint16_t *>(vector.storage.data()), dimension,
        vector.values.data());
    return 0;
  }
};

}  // namespace quantized_graph_detail

// Immutable index over any provider implementing valid(id), neighbors(id), and
// get_vector(id, Vector&). It does not build the input graph: HNSW, Vamana, or
// a caller-supplied adjacency list may provide it. The provider owns the
// vectors, IDs and fallback adjacency; keep it alive and immutable through all
// searches. The segment stores only quantized blocks/codec, bound to the
// supplied router. Callers persist and bind the original corpus separately.
class QuantizedGraphIndex {
 public:
  static constexpr const char *kSnapshotSegment = "quantized_graph.blocks";

  QuantizedGraphIndex(size_t dimension, size_t max_neighbors, bool cosine,
                      size_t stored_vector_bytes, const std::string &router,
                      bool fp16 = false, const std::string *codec = nullptr)
      : graph_(dimension,
               make_codec(dimension, max_neighbors, cosine, stored_vector_bytes,
                          fp16, codec),
               max_neighbors, stored_vector_bytes, cosine, router),
        dimension_(dimension),
        cosine_(cosine),
        fp16_(fp16),
        stored_vector_bytes_(stored_vector_bytes),
        router_bytes_(router.size()),
        router_crc_(ailego::Crc32c::Hash(router.data(), router.size())) {}

  template <class Graph>
  int prebuild(const Graph &source, size_t count, size_t threads,
               const ailego::Params &page_params = {}) {
    if (fp16_)
      return graph_.prebuild(
          quantized_graph_detail::DecodedGraph<Graph>{source, dimension_},
          count, threads, page_params);
    return graph_.prebuild(source, count, threads, page_params);
  }

  // Context supplies the FP32 geometry query and the exact stored-vector
  // metric. QuantizedGraphQuery is a ready-to-use independent implementation;
  // templated contexts also let legacy callers keep their original scoring.
  template <class Graph, class Context>
  int search(const Graph &source, Context &context) const {
    return search(entry(), source, context);
  }

  template <class Graph, class Context>
  int search(uint32_t start, const Graph &source, Context &context) const {
    return graph_.search(start, source, context);
  }

  uint32_t entry() const {
    return graph_.entry();
  }

  template <class Graph>
  static int load(const Graph &source, size_t count, size_t dimension,
                  bool cosine, size_t stored_vector_bytes,
                  const std::string &router, bool fp16,
                  const IndexStorage::Segment::Pointer &segment,
                  std::shared_ptr<QuantizedGraphIndex> &out,
                  const ailego::Params &page_params = {}) {
    using namespace quantized_graph_detail;
    if (!QuantizedGraphSupported()) return IndexError_Unsupported;
    if (!segment) return IndexError_NoExist;
    SnapshotReader read{segment};
    SnapshotHeader header{};
    if (!read(header.data(), sizeof(header)) || header[0] != kSnapshotMagic ||
        header[1] != 1 || header[2] != dimension ||
        header[3] != stored_vector_bytes || header[4] != uint64_t(cosine) ||
        header[5] != uint64_t(fp16) || header[6] < 32 || header[6] > 256 ||
        header[6] % 32 || header[7] > (4u << 20) ||
        header[7] > read.remaining() || header[8] != router.size() ||
        header[9] != ailego::Crc32c::Hash(router.data(), router.size()))
      return IndexError_InvalidFormat;

    try {
      std::string codec(size_t(header[7]), '\0');
      if (!read(codec.data(), codec.size())) return IndexError_ReadData;
      auto index = std::make_shared<QuantizedGraphIndex>(
          dimension, size_t(header[6]), cosine, stored_vector_bytes, router,
          fp16, &codec);
      const int ret =
          fp16 ? index->graph_.load(DecodedGraph<Graph>{source, dimension},
                                    count, read, read.remaining(), page_params)
               : index->graph_.load(source, count, read, read.remaining(),
                                    page_params);
      if (ret != 0) return ret;
      if (read.remaining() != 0 || read.crc != segment->data_crc())
        return IndexError_InvalidFormat;
      out = std::move(index);
      return 0;
    } catch (const std::bad_alloc &e) {
      LOG_ERROR("Quantized graph allocation failed: %s", e.what());
      return IndexError_NoMemory;
    } catch (const std::exception &e) {
      LOG_ERROR("Invalid quantized graph snapshot: %s", e.what());
      return IndexError_InvalidFormat;
    }
  }

  int dump(const IndexDumper::Pointer &dumper,
           const std::string &segment = kSnapshotSegment) const {
    using namespace quantized_graph_detail;
    if (!dumper) return IndexError_InvalidArgument;
    SnapshotWriter write{dumper};
    if (!write_snapshot(write)) return IndexError_WriteData;

    const size_t padding = (32 - write.bytes % 32) % 32;
    const char zero[32]{};
    if (padding && dumper->write(zero, padding) != padding)
      return IndexError_WriteData;
    return dumper->append(segment, write.bytes, padding, write.crc);
  }

  template <class Writer>
  bool write_snapshot(Writer &write) const {
    using namespace quantized_graph_detail;
    const auto &codec = graph_.codec().model_data();
    const SnapshotHeader header{kSnapshotMagic,
                                1,
                                dimension_,
                                stored_vector_bytes_,
                                cosine_,
                                fp16_,
                                graph_.max_neighbors(),
                                codec.size(),
                                router_bytes_,
                                router_crc_};
    return write(header.data(), sizeof(header)) &&
           write(codec.data(), codec.size()) && graph_.dump(write);
  }

  template <class Graph>
  static int train_codec(const Graph &source, size_t count, size_t dimension,
                         bool cosine, bool fp16,
                         const QuantizedGraphBuildOptions &options,
                         std::string &data) {
    if (!options.valid()) return IndexError_InvalidArgument;
    if (!QuantizedGraphSupported()) return IndexError_Unsupported;
    try {
      const auto result = fp16
                              ? QuantizedGraphCodecBuilder::train(
                                    quantized_graph_detail::DecodedGraph<Graph>{
                                        source, dimension},
                                    count, dimension, cosine, options)
                              : QuantizedGraphCodecBuilder::train(
                                    source, count, dimension, cosine, options);
      data = result.data;
      LOG_INFO(
          "QuantizedGraph codec trained: samples=%zu lambda=%.9g "
          "rank_energy=%.6f "
          "bytes=%zu",
          result.samples, result.lambda, result.captured_energy, data.size());
      return 0;
    } catch (const std::invalid_argument &e) {
      LOG_ERROR("Invalid codec training options: %s", e.what());
      return IndexError_InvalidArgument;
    } catch (const std::domain_error &e) {
      LOG_ERROR("Invalid codec training data: %s", e.what());
      return IndexError_Mismatch;
    } catch (const std::exception &e) {
      LOG_ERROR("Codec training failed: %s", e.what());
      return IndexError_Runtime;
    }
  }

  template <class Graph>
  static int train_router(const Graph &source, size_t count, size_t dimension,
                          bool cosine, size_t anchors, size_t projected,
                          uint64_t seed, std::string &data, bool fp16 = false,
                          size_t rotation_views = 1) {
    using Builder = QuantizedGraphRouterBuilder;
    if (!QuantizedGraphSupported()) return IndexError_Unsupported;
    if (!anchors || anchors > Builder::kMaxAnchors || !dimension ||
        dimension > 4096 || !count || count > uint32_t(INT32_MAX) ||
        !turbo::QuantizedGraphRotator::valid_views(rotation_views))
      return IndexError_InvalidArgument;
    try {
      typename Graph::Vector block;
      std::array<float, 4096> decoded;
      const auto training =
          Builder::collect(count, dimension, cosine, anchors, [&](uint32_t id) {
            if (!source.valid(id) || source.get_vector(id, block) != 0 ||
                !block.data())
              return static_cast<const float *>(nullptr);
            if (fp16) {
              ailego::FloatHelper::ToFP32(
                  static_cast<const uint16_t *>(block.data()), dimension,
                  decoded.data());
              // FP16 rounding may move the norm outside the FP32 training
              // tolerance. Spherical clustering trains on unit directions.
              if (cosine) {
                float norm;
                turbo::Normalizer<float>::L2(decoded.data(), dimension, &norm);
              }
              return static_cast<const float *>(decoded.data());
            }
            return static_cast<const float *>(block.data());
          });
      Builder::Options options;
      options.projected = projected;
      options.seed = seed;
      options.rotation_views = rotation_views;
      const auto model = Builder::train(training, options);
      std::ostringstream out(std::ios::out | std::ios::binary);
      model.write(out);
      data = out.str();
      return 0;
    } catch (const std::domain_error &e) {
      LOG_ERROR("Invalid router training data: %s", e.what());
      return IndexError_Mismatch;
    } catch (const std::exception &e) {
      LOG_ERROR("Router training failed: %s", e.what());
      return IndexError_Runtime;
    }
  }

 private:
  using Codec = turbo::QuantizedGraphBlockQuantizer;

  static Codec make_codec(size_t dimension, size_t neighbors, bool cosine,
                          size_t vector_bytes, bool fp16,
                          const std::string *codec) {
    if (!QuantizedGraphSupported() || !dimension || dimension > 4096 ||
        neighbors < 32 || neighbors > 256 || neighbors % 32 ||
        vector_bytes < dimension * (fp16 ? 2 : 4))
      throw std::invalid_argument("unsupported quantized graph shape or CPU");
    return Codec(dimension, cosine, codec);
  }

  QuantizedGraph<Codec> graph_;
  size_t dimension_;
  bool cosine_, fp16_;
  size_t stored_vector_bytes_, router_bytes_;
  uint32_t router_crc_;
};
#endif  // ZVEC_QUANTIZED_GRAPH_SUPPORTED

}  // namespace zvec::core
