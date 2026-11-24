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
#include "quantized_graph_searcher.h"
#include <cmath>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>
#include "quantized_graph_query.h"
#include "quantized_graph_search_context.h"

namespace zvec::core {
namespace {
template <class Buffer>
bool read_part(IndexStorage &s, const char *name, size_t max_count,
               Buffer &out) {
  auto seg = s.get(name);
  constexpr auto width = sizeof(typename Buffer::value_type);
  if (!seg || seg->data_size() % width || seg->data_size() / width > max_count)
    return false;
  out.resize(seg->data_size() / width);
  if (out.empty()) return seg->data_crc() == 0;
  core::quantized_graph_detail::SnapshotReader reader{seg};
  return reader(out.data(), seg->data_size()) && reader.crc == seg->data_crc();
}

class GraphContext final : public QuantizedGraphSearchContext {
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
  int configure(uint32_t c, float margin, float width, uint32_t expansions,
                uint32_t beam, float bonus) override {
    QuantizedGraphQuery::Options o;
    o.margin = margin;
    o.bucket_width = width;
    o.max_expansions = expansions;
    o.route_beam = beam;
    o.uncertainty_bonus = bonus;
    if (!c || !o.valid())
      return IndexError_InvalidArgument;
    capacity = c;
    options = o;
    return 0;
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
        !get("max_expansions", o.max_expansions))
      return IndexError_InvalidArgument;
    return configure(c, o.margin, o.bucket_width, o.max_expansions, o.route_beam,
                     o.uncertainty_bonus);
  }
  size_t dimension;
  bool cosine;
  QuantizedGraphQuery query;
  QuantizedGraphQuery::Options options;
  uint32_t capacity = 128;
  std::vector<float> normalized;
  std::vector<IndexDocumentList> docs;

 private:
  uint32_t k_ = 10;
};
}  // namespace
int QuantizedGraphSearcher::cleanup() {
  graph_.reset();
  data_.reset();
  keys_.clear();
  stats_.clear();
  return 0;
}

int QuantizedGraphBuilder::Result::load(
    IndexStorage &s, const ailego::Params &pages,
    std::vector<uint64_t> *external_keys) {
  auto segment = s.get(QuantizedGraphIndex::kSnapshotSegment);
  quantized_graph_detail::SnapshotHeader header{};
  if (!segment ||
      !quantized_graph_detail::SnapshotReader{segment}(header.data(),
                                                       sizeof(header)) ||
      header[0] != quantized_graph_detail::kSnapshotMagic || header[2] == 0 ||
      header[2] > 4096 || header[4] > 1)
    return IndexError_InvalidFormat;
  const size_t dim = header[2];
  const bool cosine = header[4];
  std::vector<uint64_t> keys;
  if (s.get(QuantizedGraphIndex::kKeysSegment)) {
    std::vector<uint32_t> compact;
    if (!read_part(s, QuantizedGraphIndex::kKeysSegment, UINT32_MAX - 1, compact))
      return IndexError_InvalidFormat;
    keys.assign(compact.begin(), compact.end());
  } else if (!read_part(s, QuantizedGraphIndex::kKeys64Segment, UINT32_MAX - 1, keys))
    return IndexError_InvalidFormat;
  const auto n = keys.size();
  if (!external_keys) {
    for (size_t i = 0; i < n; ++i)
      if (keys[i] != i) return IndexError_InvalidFormat;
  } else if (!unique_keys(keys)) {
    return IndexError_InvalidFormat;
  }
  auto data = std::make_shared<QuantizedGraphData>();
  data->dimension = dim;
  data->cosine = cosine;
  std::string router_bytes;
  if (!n || n > SIZE_MAX / (dim * sizeof(float)) ||
      !read_part(s, QuantizedGraphIndex::kVectorsSegment, n * dim, data->vectors) ||
      data->vectors.size() != n * dim ||
      !read_part(s, QuantizedGraphIndex::kOffsetsSegment, n + 1, data->offsets) ||
      data->offsets.size() != n + 1 ||
      !read_part(s, QuantizedGraphIndex::kEdgesSegment, n * 256, data->edges) ||
      data->offsets.front() != 0 ||
      data->offsets.back() != data->edges.size() ||
      !read_part(s, QuantizedGraphIndex::kRouterSegment, size_t{1} << 30, router_bytes))
    return IndexError_InvalidFormat;
  for (size_t i = 0; i < n; ++i)
    if (data->offsets[i + 1] - data->offsets[i] > 256)
      return IndexError_InvalidFormat;
  for (auto edge : data->edges)
    if (edge >= n) return IndexError_InvalidFormat;
  std::shared_ptr<QuantizedGraphIndex> graph;
  int rc = QuantizedGraphIndex::load(*data, n, dim, cosine, dim * sizeof(float),
                                     router_bytes, false, segment, graph, pages);
  if (rc) return rc;
  this->data = std::move(data);
  index = std::move(graph);
  router = std::move(router_bytes);
  codec.clear();
  build_params = {};
  if (external_keys) *external_keys = std::move(keys);
  return 0;
}

int QuantizedGraphBuilder::Result::open(
    const std::string &path, const ailego::Params &pages,
    std::vector<uint64_t> *keys) {
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  if (!storage) return IndexError_NoExist;
  int rc = storage->init({});
  if (rc || (rc = storage->open(path, false))) return rc;
  IndexMeta meta;
  if ((rc = IndexHelper::DeserializeFromStorage(storage.get(), &meta))) return rc;
  if (!QuantizedGraphSupportsMeta(meta) ||
      meta.searcher_name() != "QuantizedGraphSearcher")
    return IndexError_InvalidFormat;
  Result next;
  std::vector<uint64_t> mapping;
  if ((rc = next.load(*storage, pages, keys ? &mapping : nullptr))) return rc;
  if (meta.dimension() != next.data->dimension ||
      (meta.metric_name() == "Cosine") != next.data->cosine)
    return IndexError_InvalidFormat;
  if ((rc = storage->close())) return rc;
  next.build_params = meta.builder_params();
  *this = std::move(next);
  if (keys) *keys = std::move(mapping);
  return 0;
}

int QuantizedGraphSearcher::load(IndexStorage::Pointer storage,
                                 IndexMetric::Pointer metric) {
  if (!QuantizedGraphSupported()) return IndexError_Unsupported;
  if (!storage) return IndexError_InvalidArgument;
  QuantizedGraphBuilder::Result built;
  std::vector<uint64_t> keys;
  int rc = built.load(*storage, {}, &keys);
  if (rc) return rc;
  const size_t dim = built.data->dimension;
  const bool cosine = built.data->cosine;
  IndexMeta meta(IndexMeta::DT_FP32, dim);
  if (cosine) meta.set_metric("Cosine", 0, ailego::Params{});
  if (metric && metric->name() != meta.metric_name())
    return IndexError_InvalidArgument;
  auto exact =
      IndexFactory::CreateMetric(cosine ? "InnerProduct" : "SquaredEuclidean");
  if (!exact || exact->init(meta, {})) return IndexError_Unsupported;
  distance_ = [dist = exact->distance(), cosine](const void *a, const void *b,
                                                 size_t d, float *v) {
    dist(a, b, d, v);
    if (cosine) *v += 1.f;
  };
  meta_ = std::move(meta);
  keys_ = std::move(keys);
  graph_ = std::move(built.index);
  data_ = std::move(built.data);
  stats_.set_loaded_count(keys_.size());
  return 0;
}

IndexContext::Pointer QuantizedGraphSearcher::create_context() const {
  if (!graph_) return nullptr;
  auto c = std::make_unique<GraphContext>(data_->dimension, data_->cosine,
                                          distance_);
  if (c->update(params_)) return nullptr;
  return c;
}

int QuantizedGraphSearcher::search_impl(const void *v, const IndexQueryMeta &m,
                                        uint32_t count,
                                        Context::Pointer &ctx) const {
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

INDEX_FACTORY_REGISTER_SEARCHER(QuantizedGraphSearcher);
}  // namespace zvec::core
