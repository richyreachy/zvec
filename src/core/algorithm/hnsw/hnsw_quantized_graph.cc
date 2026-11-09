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
#include "hnsw_quantized_graph.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <turbo/norm/normalizer.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_error.h>
#include "hnsw_context.h"
#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#define ZVEC_PROJECTED_QG_SUPPORTED 1
#include <turbo/distance/avx512/fp16/fp32_query.h>
#include <turbo/quantizer/symqg_quantizer/symqg_quantizer.h>
#include "algorithm/quantized_graph/quantized_graph.h"
#include "algorithm/quantized_graph/quantized_graph_codec_builder.h"
#include "algorithm/quantized_graph/quantized_graph_router_builder.h"
#endif
namespace zvec::core {
namespace {
// Version 1 binds the Block layout and codec packing. Empty/native and V1
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
#if ZVEC_PROJECTED_QG_SUPPORTED
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
#endif
}  // namespace
#if ZVEC_PROJECTED_QG_SUPPORTED
namespace {
struct HnswGraphAdapter {
  using Vector = IndexStorage::MemoryBlock;
  const HnswEntity &entity;
  bool valid(uint32_t id) const {
    return entity.get_key(id) != kInvalidKey;
  }
  int get_vector(uint32_t id, Vector &vector) const {
    return entity.get_vector(id, vector);
  }
  Neighbors neighbors(uint32_t id) const {
    return entity.get_neighbors(0, id);
  }
};

// Decode only the requested vector. No persistent FP32 copy of the database.
// The graph's bounds and terminal scores refer to these widened stored values.
struct HnswFp16GraphAdapter : HnswGraphAdapter {
  struct Vector {
    IndexStorage::MemoryBlock storage;
    std::array<float, 4096> values;
    const void *data() const {
      return values.data();
    }
  };
  size_t dimension;
  int get_vector(uint32_t id, Vector &vector) const {
    const int ret = entity.get_vector(id, vector.storage);
    if (ret != 0) return ret;
    if (!vector.storage.data()) return IndexError_Mismatch;
    ailego::FloatHelper::ToFP32(
        static_cast<const uint16_t *>(vector.storage.data()), dimension,
        vector.values.data());
    return 0;
  }
};

struct HnswQueryAdapter {
  HnswContext &ctx;
  TopkHeap &reset_results() {
    return ctx.search_heap().reset<TopkHeap>(std::max(ctx.topk(), ctx.ef()));
  }
  const QuantizedGraphSearchOptions &quantized_graph_options() const {
    return ctx.quantized_graph_options();
  }
  uint32_t topk() const {
    return ctx.topk();
  }
  const void *query() const {
    return ctx.dist_calculator().query();
  }
  float distance(const void *vector) {
    return ctx.dist_calculator().dist(vector);
  }
  bool excluded(uint32_t id) const {
    return ctx.filter().is_valid() &&
           ctx.filter()(ctx.get_entity().get_key(id));
  }
  bool reach_scan_limit() const {
    return ctx.reach_scan_limit();
  }
  void on_expand() {
    if (ctx.debugging()) ++(*ctx.mutable_stats_get_neighbors());
  }
};
struct HnswFp16QueryAdapter : HnswQueryAdapter {
  const float *values;
  size_t dimension;
  bool cosine;
  const void *query() const {
    return values;
  }
  float distance(const void *vector) {
    ctx.dist_calculator().record_comparison();
    // Widen in registers after the graph has prefetched the storage batch.
    float score;
    if (cosine)
      turbo::avx512::cosine_fp16_fp32_distance_avx512(vector, values, dimension,
                                                      &score);
    else
      turbo::avx512::squared_euclidean_fp16_fp32_distance_avx512(
          vector, values, dimension, &score);
    return score;
  }
};
}  // namespace

struct HnswQuantizedGraph::Impl {
  using Codec = turbo::SymQGBlockQuantizer;
  Impl(size_t dimension, size_t max_neighbors, bool cosine, bool fp16,
       size_t stored_vector_bytes, const std::string &model_data,
       const std::string &codec_data)
      : graph(dimension, Codec(dimension, cosine, codec_data), max_neighbors,
              stored_vector_bytes, cosine, model_data),
        dimension_(dimension),
        cosine_(cosine),
        fp16_(fp16),
        stored_vector_bytes_(stored_vector_bytes),
        router_bytes_(model_data.size()),
        router_crc_(
            ailego::Crc32c::Hash(model_data.data(), model_data.size())) {}
  int prebuild(const HnswEntity &entity, size_t count, size_t threads) {
    if (fp16_)
      return graph.prebuild(HnswFp16GraphAdapter{{entity}, dimension_}, count,
                            threads);
    else
      return graph.prebuild(HnswGraphAdapter{entity}, count, threads);
  }
  int search(uint32_t entry_id, HnswContext &ctx) const {
    if (fp16_) {
      std::array<float, 4096> values;
      const float *query_values = ctx.quantized_graph_fp32_query();
      if (!query_values) {
        // Direct streamer callers supply a stored-format FP16 query.
        ailego::FloatHelper::ToFP32(
            static_cast<const uint16_t *>(ctx.dist_calculator().query()),
            dimension_, values.data());
        query_values = values.data();
      } else if (cosine_) {
        std::copy_n(query_values, dimension_, values.data());
        float norm;
        turbo::Normalizer<float>::L2(values.data(), dimension_, &norm);
        query_values = values.data();
      }
      HnswFp16QueryAdapter query{{ctx}, query_values, dimension_, cosine_};
      return graph.search(entry_id, HnswGraphAdapter{ctx.get_entity()}, query);
    } else {
      HnswQueryAdapter query{ctx};
      return graph.search(entry_id, HnswGraphAdapter{ctx.get_entity()}, query);
    }
  }
  uint32_t entry() const {
    return graph.entry();
  }
  int load(const HnswEntity &entity, SnapshotReader &read) {
    return graph.load(HnswGraphAdapter{entity}, entity.doc_cnt(), read,
                      read.remaining());
  }
  int dump(const IndexDumper::Pointer &dumper) const {
    const auto &codec = graph.codec().model_data();
    const SnapshotHeader header{kSnapshotMagic,
                                1,
                                dimension_,
                                stored_vector_bytes_,
                                cosine_,
                                fp16_,
                                graph.max_neighbors(),
                                codec.size(),
                                router_bytes_,
                                router_crc_};
    SnapshotWriter write{dumper};
    if (!write(header.data(), sizeof(header)) ||
        !write(codec.data(), codec.size()) || !graph.dump(write))
      return IndexError_WriteData;
    size_t padding = 0;
    const int ret =
        HnswEntity::CalcAndAddPadding(dumper, write.bytes, &padding);
    if (ret != 0) return ret;
    return dumper->append(kSnapshotSegment, write.bytes, padding, write.crc);
  }
  QuantizedGraph<Codec> graph;
  size_t dimension_;
  bool cosine_, fp16_;
  size_t stored_vector_bytes_, router_bytes_;
  uint32_t router_crc_;
};
#else
struct HnswQuantizedGraph::Impl {
  int prebuild(const HnswEntity &, size_t, size_t) {
    return IndexError_Unsupported;
  }
  int search(uint32_t, HnswContext &) const {
    return IndexError_Unsupported;
  }
  uint32_t entry() const {
    return UINT32_MAX;
  }
  int load(const HnswEntity &, SnapshotReader &) {
    return IndexError_Unsupported;
  }
  int dump(const IndexDumper::Pointer &) const {
    return IndexError_Unsupported;
  }
};
#endif
bool HnswQuantizedGraph::supported() {
#if ZVEC_PROJECTED_QG_SUPPORTED
  return __builtin_cpu_supports("avx512f") &&
         __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512vl") &&
         __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}
HnswQuantizedGraph::HnswQuantizedGraph(size_t dimension, size_t max_neighbors,
                                       bool cosine, size_t stored_vector_bytes,
                                       const std::string &model_data, bool fp16,
                                       const std::string &codec_data) {
#if ZVEC_PROJECTED_QG_SUPPORTED
  if (supported() && dimension > 0 && dimension <= 4096 &&
      max_neighbors >= 32 && max_neighbors <= 256 && max_neighbors % 32 == 0) {
    impl_ = std::make_unique<Impl>(dimension, max_neighbors, cosine, fp16,
                                   stored_vector_bytes, model_data, codec_data);
  }
#else
  (void)dimension;
  (void)max_neighbors;
  (void)cosine;
  (void)stored_vector_bytes;
  (void)model_data;
  (void)fp16;
  (void)codec_data;
#endif
}
int HnswQuantizedGraph::train_codec(const HnswEntity &entity, size_t dimension,
                                    bool cosine, bool fp16,
                                    const QuantizedGraphBuildOptions &options,
                                    std::string &data) {
  if (!options.valid()) return IndexError_InvalidArgument;
#if ZVEC_PROJECTED_QG_SUPPORTED
  if (!supported()) return IndexError_Unsupported;
  try {
    const auto result = fp16 ? SymQGCodecBuilder::train(
                                   HnswFp16GraphAdapter{{entity}, dimension},
                                   entity.doc_cnt(), dimension, cosine, options)
                             : SymQGCodecBuilder::train(
                                   HnswGraphAdapter{entity}, entity.doc_cnt(),
                                   dimension, cosine, options);
    data = result.data;
    LOG_INFO(
        "SymQG codec trained: samples=%zu lambda=%.9g rank_energy=%.6f "
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
#else
  (void)entity;
  (void)dimension;
  (void)cosine;
  (void)fp16;
  (void)data;
  return IndexError_Unsupported;
#endif
}
int HnswQuantizedGraph::load(const HnswEntity &entity, size_t dimension,
                             bool cosine, size_t stored_vector_bytes,
                             const std::string &router_data, bool fp16,
                             const IndexStorage::Segment::Pointer &segment,
                             std::shared_ptr<HnswQuantizedGraph> &out) {
  if (!supported()) return IndexError_Unsupported;
  if (!segment) return IndexError_NoExist;
  SnapshotReader read{segment};
  SnapshotHeader header{};
  if (!read(header.data(), sizeof(header)) || header[0] != kSnapshotMagic ||
      header[1] != 1 || header[2] != dimension ||
      header[3] != stored_vector_bytes || header[4] != uint64_t(cosine) ||
      header[5] != uint64_t(fp16) || header[6] < 32 || header[6] > 256 ||
      header[6] % 32 || header[7] > (4u << 20) ||
      header[7] > read.remaining() || header[8] != router_data.size() ||
      header[9] != ailego::Crc32c::Hash(router_data.data(), router_data.size()))
    return IndexError_InvalidFormat;
  std::string codec(size_t(header[7]), '\0');
  if (!read(codec.data(), codec.size())) return IndexError_ReadData;
  try {
    auto graph = std::make_shared<HnswQuantizedGraph>(
        dimension, size_t(header[6]), cosine, stored_vector_bytes, router_data,
        fp16, codec);
    if (!graph->impl_) return IndexError_Unsupported;
    const int ret = graph->impl_->load(entity, read);
    if (ret != 0) return ret;
    if (read.remaining() != 0 || read.crc != segment->data_crc())
      return IndexError_InvalidFormat;
    out = std::move(graph);
    return 0;
  } catch (const std::exception &e) {
    LOG_ERROR("Invalid quantized graph snapshot: %s", e.what());
    return IndexError_InvalidFormat;
  }
}
int HnswQuantizedGraph::dump(const IndexDumper::Pointer &dumper) const {
  return impl_ ? impl_->dump(dumper) : IndexError_Unsupported;
}
int HnswQuantizedGraph::train_router(const HnswEntity &entity, size_t dimension,
                                     bool cosine, size_t anchors,
                                     size_t projected, uint64_t seed,
                                     std::string &data, bool fp16,
                                     size_t rotation_views) {
#if ZVEC_PROJECTED_QG_SUPPORTED
  using Builder = QuantizedGraphRouterBuilder;
  if (!supported()) return IndexError_Unsupported;
  if (!anchors || anchors > Builder::kMaxAnchors || !dimension ||
      dimension > 4096 || !entity.doc_cnt() ||
      entity.doc_cnt() > uint32_t(INT32_MAX) ||
      !turbo::SymQGRotation::valid_views(rotation_views))
    return IndexError_InvalidArgument;
  try {
    IndexStorage::MemoryBlock block;
    std::array<float, 4096> decoded;
    const auto training = Builder::collect(
        entity.doc_cnt(), dimension, cosine, anchors, [&](uint32_t id) {
          if (entity.get_key(id) == kInvalidKey ||
              entity.get_vector(id, block) != 0 || !block.data())
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
#else
  (void)entity;
  (void)dimension;
  (void)cosine;
  (void)anchors;
  (void)projected;
  (void)seed;
  (void)data;
  (void)fp16;
  (void)rotation_views;
  return IndexError_Unsupported;
#endif
}
int HnswQuantizedGraph::prebuild(const HnswEntity &entity, size_t count,
                                 size_t threads) {
  return impl_ ? impl_->prebuild(entity, count, threads)
               : IndexError_Unsupported;
}
int HnswQuantizedGraph::search(uint32_t entry_id, HnswContext &ctx) const {
  return impl_ ? impl_->search(entry_id, ctx) : IndexError_Unsupported;
}
uint32_t HnswQuantizedGraph::entry() const {
  return impl_ ? impl_->entry() : UINT32_MAX;
}
HnswQuantizedGraph::~HnswQuantizedGraph() = default;
}  // namespace zvec::core
