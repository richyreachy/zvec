// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include "quantized_graph_framework.h"
#include <cmath>
#include <numeric>
#include <zvec/ailego/encoding/json.h>
#include <zvec/core/framework/index_helper.h>
#include "quantized_graph_builder.h"
#include "quantized_graph_query.h"

namespace zvec::core {
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
namespace {
using Storage = IndexStorage;
using Builder = QuantizedGraphBuilder;
using Graph = QuantizedGraphIndex;
using Encoding = QuantizedGraphBuildOptions;
class OptionsReader {
 public:
  explicit OptionsReader(const ailego::JsonObject &o) : obj_(o) {}
  template <class T>
  void take(const char *key, T &out) {
    ailego::JsonValue v;
    if (!obj_.get(key, &v)) return;
    ++consumed_;
    if constexpr (std::is_integral_v<T>) {
      if (!v.is_integer() || v.as_integer() < 0 ||
          uint64_t(v.as_integer()) > std::numeric_limits<T>::max()) {
        ok_ = false;
        return;
      }
      out = static_cast<T>(v.as_integer());
    } else if constexpr (std::is_floating_point_v<T>) {
      if (!v.is_integer() && !v.is_float()) {
        ok_ = false;
        return;
      }
      out = v.is_float() ? v.as_float() : double(v.as_integer());
    } else {
      if (!obj_.get(key, &out)) ok_ = false;
    }
  }
  bool valid() const {
    return ok_ && consumed_ == obj_.size();
  }

 private:
  ailego::JsonObject obj_;
  size_t consumed_ = 0;
  bool ok_ = true;
};

bool parse_options(const std::string &json, Builder::Options &out) {
  ailego::JsonValue value;
  if (!value.parse(json) || !value.is_object()) return false;
  OptionsReader r(value.as_object());
  r.take("passes", out.passes);
  r.take("threads", out.threads);
  r.take("search_list_size", out.search_list_size);
  r.take("seed", out.seed);
  r.take("router_seed", out.router_seed);
  r.take("router_dimension", out.router_dimension);
  r.take("router_anchors", out.router_anchors);
  ailego::JsonObject encoding;
  r.take("encoding", encoding);
  auto &enc = out.encoding;
  OptionsReader e(encoding);
  std::string codec = "rabitq", codebook = "hadamard";
  e.take("codec", codec);
  e.take("codebook", codebook);
  if (codec == "rabitq")
    enc.codec = Encoding::Codec::RaBitQ;
  else if (codec == "global")
    enc.codec = Encoding::Codec::Global;
  else if (codec == "low_rank")
    enc.codec = Encoding::Codec::LowRank;
  else
    return false;
  if (codebook == "gaussian")
    enc.codebook = Encoding::Codebook::Gaussian;
  else if (codebook != "hadamard")
    return false;
  e.take("max_neighbors", enc.max_neighbors);
  e.take("rank", enc.rank);
  e.take("rotation_views", enc.rotation_views);
  e.take("block_dimension", enc.block_dimension);
  e.take("training_samples", enc.training_samples);
  e.take("center_weight", enc.center_weight);
  e.take("seed", enc.seed);
  return e.valid() && r.valid() && enc.valid() && out.passes &&
         out.passes <= 8 && out.threads && out.threads <= 256 &&
         out.search_list_size >= enc.max_neighbors &&
         out.search_list_size <= (1u << 20) &&
         (!out.router_anchors || out.router_dimension);
}

template <class Emit>
bool write_part(Storage &store, const char *name, Emit emit) {
  size_t bytes = 0;
  auto count = [&](const void *, size_t n) {
    bytes += n;
    return true;
  };
  if (!emit(count) || store.append(name, std::max<size_t>(bytes, 1)) != 0)
    return false;
  auto seg = store.get(name);
  if (!seg) return false;
  uint32_t crc = 0;
  size_t pos = 0;
  auto write = [&](const void *p, size_t n) {
    if (!n) return true;
    if (seg->write(pos, p, n) != n) return false;
    pos += n;
    crc = ailego::Crc32c::Hash(p, n, crc);
    return true;
  };
  if (!emit(write) || pos != bytes || seg->resize(bytes) != bytes) return false;
  seg->update_data_crc(crc);
  return true;
}
template <class Buffer>
bool read_part(Storage &s, const char *name, size_t max_count, Buffer &out) {
  auto seg = s.get(name);
  constexpr auto width = sizeof(typename Buffer::value_type);
  if (!seg || seg->data_size() % width || seg->data_size() / width > max_count)
    return false;
  out.resize(seg->data_size() / width);
  if (out.empty()) return seg->data_crc() == 0;
  core::quantized_graph_detail::SnapshotReader reader{seg};
  return reader(out.data(), seg->data_size()) && reader.crc == seg->data_crc();
}
bool valid_meta(const IndexMeta &m) {
  return m.meta_type() == IndexMeta::MT_DENSE &&
         m.data_type() == IndexMeta::DT_FP32 && m.dimension() &&
         m.dimension() <= 4096 &&
         m.element_size() == size_t(m.dimension()) * sizeof(float) &&
         (m.metric_name() == "Cosine" || m.metric_name() == "SquaredEuclidean");
}

class QuantizedGraphIndexBuilder final : public IndexBuilder {
 public:
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    Builder::Options options;
    std::string json = "{}";
    if (!QuantizedGraphSupported()) return IndexError_Unsupported;
    if (!valid_meta(meta) ||
        (params.has("options") && !params.get("options", &json)) ||
        !parse_options(json, options))
      return IndexError_InvalidArgument;
    cleanup();
    meta_ = meta;
    meta_.set_builder("QuantizedGraphIndexBuilder", 0, params);
    meta_.set_searcher("QuantizedGraphSearcher", 0, ailego::Params{});
    meta_.set_streamer("QuantizedGraphStreamer", 0, ailego::Params{});
    options_ = std::move(options);
    initialized_ = true;
    return 0;
  }
  int cleanup() override {
    built_ = {};
    keys_.clear();
    stats_.clear();
    initialized_ = false;
    return 0;
  }
  const Stats &stats() const override {
    return stats_;
  }
  int train(IndexThreads::Pointer, IndexHolder::Pointer h) override {
    // Topology, codec and router are trained together during build().
    return compatible(h) ? 0 : IndexError_InvalidArgument;
  }
  int build(IndexThreads::Pointer, IndexHolder::Pointer h) override {
    if (!compatible(h)) return IndexError_InvalidArgument;
    const auto n = h->count(), dim = h->dimension();
    std::vector<uint64_t> keys;
    std::vector<float> rows(n * dim);
    keys.reserve(n);
    auto it = h->create_iterator();
    if (!it) return IndexError_ReadData;
    for (; it->is_valid(); it->next()) {
      if (keys.size() >= n || !it->data()) return IndexError_ReadData;
      std::memcpy(rows.data() + keys.size() * dim, it->data(),
                  dim * sizeof(float));
      keys.push_back(it->key());
    }
    if (it->status() || keys.size() != n) return IndexError_ReadData;
    if (!std::is_sorted(keys.begin(), keys.end())) {
      std::vector<size_t> order(n);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](size_t a, size_t b) { return keys[a] < keys[b]; });
      std::vector<float> sorted(rows.size());
      for (size_t i = 0; i < n; ++i)
        std::copy_n(rows.data() + order[i] * dim, dim, sorted.data() + i * dim);
      rows.swap(sorted);
      std::sort(keys.begin(), keys.end());
    }
    if (std::adjacent_find(keys.begin(), keys.end()) != keys.end())
      return IndexError_InvalidArgument;
    Builder::Result result;
    int rc = Builder::build(std::move(rows), dim,
                            meta_.metric_name() == "Cosine", options_, result);
    if (rc) return rc;
    keys_ = std::move(keys);
    built_ = std::move(result);
    stats_.set_built_count(n);
    return 0;
  }

  template <class Part>
  int serialize(Part part) {
    if (!built_.index) return IndexError_InvalidArgument;
    auto buffer = [&](const char *name, const auto &values) {
      return part(name, [&](auto &write) {
        return write(values.data(), values.size() * sizeof(*values.data()));
      });
    };
    if (keys_.back() <= UINT32_MAX) {
      std::vector<uint32_t> compact(keys_.begin(), keys_.end());
      if (!buffer("qg.keys", compact)) return IndexError_WriteData;
    } else if (!buffer("qg.keys64", keys_))
      return IndexError_WriteData;
    if (!buffer("qg.vectors", built_.data->vectors) ||
        !buffer("qg.offsets", built_.data->offsets) ||
        !buffer("qg.edges", built_.data->edges) ||
        !buffer("qg.router", built_.router) ||
        !part(Graph::kSnapshotSegment,
              [&](auto &write) { return built_.index->write_snapshot(write); }))
      return IndexError_WriteData;
    stats_.set_dumped_count(keys_.size());
    return 0;
  }

  int dump(const IndexDumper::Pointer &dumper) override {
    if (!dumper || !built_.index) return IndexError_InvalidArgument;
    int rc = IndexHelper::SerializeToDumper(meta_, dumper.get());
    if (rc) return rc;
    return serialize([&](const char *name, auto emit) {
      quantized_graph_detail::SnapshotWriter write{dumper};
      if (!emit(write)) return false;
      const size_t pad = (32 - write.bytes % 32) % 32;
      const char zeros[32]{};
      return (!pad || dumper->write(zeros, pad) == pad) &&
             dumper->append(name, write.bytes, pad, write.crc) == 0;
    });
  }

 private:
  bool compatible(const IndexHolder::Pointer &h) const {
    return initialized_ && h && h->count() && h->count() < UINT32_MAX &&
           h->dimension() == meta_.dimension() &&
           h->data_type() == IndexMeta::DT_FP32 &&
           h->element_size() == meta_.element_size() &&
           h->count() <= SIZE_MAX / meta_.element_size();
  }
  Stats stats_;
  IndexMeta meta_;
  Builder::Options options_;
  Builder::Result built_;
  std::vector<uint64_t> keys_;
  bool initialized_ = false;
};

class GraphContext final : public IndexContext {
 public:
  GraphContext(size_t dim, bool angular, IndexMetric::MatrixDistance dist)
      : dimension(dim), cosine(angular), query(dim, std::move(dist)) {}
  void set_topk(uint32_t k) override {
    k_ = k;
  }
  uint32_t topk() const override {
    return k_;
  }
  const IndexDocumentList &result() const override {
    return result(0);
  }
  const IndexDocumentList &result(size_t i) const override {
    static const IndexDocumentList empty;
    return i < docs.size() ? docs[i] : empty;
  }
  IndexDocumentList *mutable_result(size_t i) override {
    return i < docs.size() ? &docs[i] : nullptr;
  }
  int update(const ailego::Params &p) override {
    auto o = options;
    auto c = capacity;
    auto get = [&](const char *key, auto &v) {
      return !p.has(key) || p.get(key, &v);
    };
    if (!get("capacity", c) || !c || !get("margin", o.margin) ||
        !get("bucket_width", o.bucket_width) ||
        !get("route_beam", o.route_beam) ||
        !get("uncertainty_bonus", o.uncertainty_bonus) ||
        !get("max_expansions", o.max_expansions) || !std::isfinite(o.margin) ||
        (o.margin < 0 && o.margin != -1) ||
        !(o.bucket_width >= 0 && o.bucket_width <= 1) || o.route_beam > 4 ||
        !(o.uncertainty_bonus >= 0 && o.uncertainty_bonus <= 1))
      return IndexError_InvalidArgument;
    capacity = c;
    options = o;
    return 0;
  }
  size_t dimension;
  bool cosine;
  QuantizedGraphQuery query;
  QuantizedGraphSearchOptions options;
  uint32_t capacity = 128;
  std::vector<float> normalized;
  std::vector<IndexDocumentList> docs;

 private:
  uint32_t k_ = 10;
};

class QuantizedGraphSearcher final : public IndexSearcher {
 public:
  int init(const ailego::Params &p) override {
    params_ = p;
    return 0;
  }
  const ailego::Params &params() const override {
    return params_;
  }
  const Stats &stats() const override {
    return stats_;
  }
  const IndexMeta &meta() const override {
    return meta_;
  }
  int cleanup() override {
    graph_.reset();
    data_.reset();
    keys_.clear();
    stats_.clear();
    return 0;
  }
  int unload() override {
    return cleanup();
  }
  int load(IndexStorage::Pointer storage,
           IndexMetric::Pointer metric) override {
    if (!QuantizedGraphSupported()) return IndexError_Unsupported;
    if (!storage) return IndexError_InvalidArgument;
    auto &s = *storage;
    auto segment = s.get(Graph::kSnapshotSegment);
    quantized_graph_detail::SnapshotHeader header{};
    if (!segment ||
        !quantized_graph_detail::SnapshotReader{segment}(header.data(),
                                                         sizeof(header)) ||
        header[0] != quantized_graph_detail::kSnapshotMagic || header[2] == 0 ||
        header[2] > 4096 || header[4] > 1)
      return IndexError_InvalidFormat;
    const size_t dim = header[2];
    const bool cosine = header[4];
    IndexMeta meta(IndexMeta::DT_FP32, dim);
    if (cosine) meta.set_metric("Cosine", 0, ailego::Params{});
    // The snapshot fixes the metric. The query kernel needs squared L2 or
    // negative inner product on the normalized FP32 rows.
    if (metric && metric->name() != meta.metric_name())
      return IndexError_InvalidArgument;
    auto exact = IndexFactory::CreateMetric(cosine ? "InnerProduct"
                                                   : "SquaredEuclidean");
    if (!exact || exact->init(meta, {})) return IndexError_Unsupported;
    std::vector<uint64_t> keys;
    if (s.get("qg.keys")) {
      std::vector<uint32_t> compact;
      if (!read_part(s, "qg.keys", UINT32_MAX - 1, compact))
        return IndexError_InvalidFormat;
      keys.assign(compact.begin(), compact.end());
    } else if (!read_part(s, "qg.keys64", UINT32_MAX - 1, keys))
      return IndexError_InvalidFormat;
    const auto n = keys.size();
    auto data = std::make_shared<QuantizedGraphData>();
    data->dimension = dim;
    data->cosine = cosine;
    std::string router;
    if (!n || n > SIZE_MAX / (dim * sizeof(float)) ||
        !std::is_sorted(keys.begin(), keys.end()) ||
        std::adjacent_find(keys.begin(), keys.end()) != keys.end() ||
        !read_part(s, "qg.vectors", n * dim, data->vectors) ||
        data->vectors.size() != n * dim ||
        !read_part(s, "qg.offsets", n + 1, data->offsets) ||
        data->offsets.size() != n + 1 ||
        !read_part(s, "qg.edges", n * 256, data->edges) ||
        data->offsets.front() != 0 ||
        data->offsets.back() != data->edges.size() ||
        !read_part(s, "qg.router", size_t{1} << 30, router))
      return IndexError_InvalidFormat;
    for (size_t i = 0; i < n; ++i)
      if (data->offsets[i + 1] - data->offsets[i] > 256)
        return IndexError_InvalidFormat;
    for (auto edge : data->edges)
      if (edge >= n) return IndexError_InvalidFormat;
    std::shared_ptr<Graph> graph;
    int rc = Graph::load(*data, n, dim, cosine, dim * sizeof(float), router,
                         false, segment, graph);
    if (rc) return rc;
    distance_ = [dist = exact->distance(), cosine](const void *a, const void *b,
                                                   size_t d, float *v) {
      dist(a, b, d, v);
      if (cosine) *v += 1.f;
    };
    meta_ = std::move(meta);
    keys_ = std::move(keys);
    graph_ = std::move(graph);
    data_ = std::move(data);
    stats_.set_loaded_count(n);
    return 0;
  }
  Context::Pointer create_context() const override {
    if (!graph_) return nullptr;
    auto c = std::make_unique<GraphContext>(data_->dimension, data_->cosine,
                                            distance_);
    if (c->update(params_)) return nullptr;
    return c;
  }
  int search_impl(const void *v, const IndexQueryMeta &m,
                  Context::Pointer &ctx) const override {
    return search_impl(v, m, 1, ctx);
  }
  int search_impl(const void *v, const IndexQueryMeta &m, uint32_t count,
                  Context::Pointer &ctx) const override {
    auto *c = dynamic_cast<GraphContext *>(ctx.get());
    if (!graph_ || !c || (!v && count) || c->dimension != data_->dimension ||
        c->cosine != data_->cosine || m.dimension() != data_->dimension ||
        m.data_type() != IndexMeta::DT_FP32 ||
        m.element_size() != meta_.element_size())
      return IndexError_InvalidArgument;
    c->docs.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto &docs = c->docs[i];
      docs.clear();
      if (!c->topk()) continue;
      const float *q = static_cast<const float *>(v) + i * c->dimension;
      double norm = 0;
      const double limit = std::sqrt(double(std::numeric_limits<float>::max()) /
                                     (16 * c->dimension));
      for (size_t d = 0; d < c->dimension; ++d) {
        if (!(std::abs(q[d]) <= limit)) return IndexError_InvalidArgument;
        if (c->cosine) norm += double(q[d]) * q[d];
      }
      if (c->cosine) {
        if (!norm) return IndexError_InvalidArgument;
        c->normalized.assign(q, q + c->dimension);
        const auto scale = 1 / std::sqrt(norm);
        for (auto &x : c->normalized) x *= scale;
        q = c->normalized.data();
      }
      QuantizedGraphQuery::Filter filter;
      if (c->filter().is_valid())
        filter = [this, c](uint32_t id) { return c->filter()(keys_[id]); };
      int rc = c->query.reset(q, std::min<size_t>(c->topk(), keys_.size()),
                              c->capacity, c->options, 0, std::move(filter));
      if (!rc) rc = graph_->search(*data_, c->query);
      if (rc) return rc;
      for (auto &hit : c->query.results()) {
        if (docs.size() == c->topk()) break;
        if (!c->threshold_set() || hit.second <= c->threshold())
          docs.emplace_back(keys_[hit.first], hit.second);
      }
    }
    return 0;
  }

 private:
  Stats stats_;
  IndexMeta meta_;
  ailego::Params params_;
  std::vector<uint64_t> keys_;
  std::shared_ptr<const QuantizedGraphData> data_;
  std::shared_ptr<Graph> graph_;
  IndexMetric::MatrixDistance distance_;
};

// Like DiskANN, the streamer serves a sealed index. Collection buffers new
// documents in its writable segment and rebuilds the graph when sealing it.
class QuantizedGraphStreamer final : public IndexStreamer {
 public:
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    if (!valid_meta(meta)) return IndexError_InvalidArgument;
    meta_ = meta;
    return searcher_.init(params);
  }
  int open(IndexStorage::Pointer s) override {
    int rc = searcher_.load(std::move(s), nullptr);
    if (!rc && (searcher_.meta().dimension() != meta_.dimension() ||
                searcher_.meta().metric_name() != meta_.metric_name())) {
      close();
      return IndexError_InvalidFormat;
    }
    return rc;
  }
  int close() override {
    return searcher_.unload();
  }
  int cleanup() override {
    return close();
  }
  int flush(uint64_t) override {
    return 0;
  }
  const IndexMeta &meta() const override {
    return meta_;
  }
  const Stats &stats() const override {
    return searcher_.stats();
  }
  Context::Pointer create_context() const override {
    return searcher_.create_context();
  }
  int search_impl(const void *v, const IndexQueryMeta &m,
                  Context::Pointer &c) const override {
    return searcher_.search_impl(v, m, c);
  }
  int search_impl(const void *v, const IndexQueryMeta &m, uint32_t n,
                  Context::Pointer &c) const override {
    return searcher_.search_impl(v, m, n, c);
  }

 private:
  IndexMeta meta_;
  QuantizedGraphSearcher searcher_;
};
}  // namespace

INDEX_FACTORY_REGISTER_BUILDER(QuantizedGraphIndexBuilder);
INDEX_FACTORY_REGISTER_SEARCHER(QuantizedGraphSearcher);
INDEX_FACTORY_REGISTER_STREAMER(QuantizedGraphStreamer);

int AppendQuantizedGraph(IndexBuilder &builder, IndexStorage &s) {
  auto *b = dynamic_cast<QuantizedGraphIndexBuilder *>(&builder);
  if (!b) return IndexError_InvalidArgument;
  return b->serialize(
      [&](const char *name, auto emit) { return write_part(s, name, emit); });
}
#else
int AppendQuantizedGraph(IndexBuilder &, IndexStorage &) {
  return IndexError_Unsupported;
}
#endif
}  // namespace zvec::core
