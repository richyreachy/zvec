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

#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <zvec/core/framework/index_factory.h>
#include "algorithm/quantized_graph/quantized_graph_builder.h"
#include "algorithm/quantized_graph/quantized_graph_params.h"
#include "algorithm/quantized_graph/quantized_graph_query.h"

namespace py = pybind11;
namespace zvec {
using namespace core;

namespace {
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
using Array = py::array_t<float, py::array::c_style>;

void check(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}

void status(int code) {
  if (!code) return;
  const std::string message =
      "quantized graph: " + std::string(IndexError::What(code)) + " (" +
      std::to_string(code) + ")";
  if (code == IndexError_InvalidArgument) throw std::invalid_argument(message);
  throw std::runtime_error(message);
}

template <class T>
std::vector<T> read_part(const std::string &path, size_t max_elements) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  check(bool(stream), "open " + path);
  const auto bytes = stream.tellg();
  check(bytes >= 0 && uint64_t(bytes) % sizeof(T) == 0 &&
            uint64_t(bytes) / sizeof(T) <= max_elements,
        "invalid size: " + path);
  std::vector<T> part(size_t(bytes) / sizeof(T));
  stream.seekg(0);
  check(bool(stream.read(reinterpret_cast<char *>(part.data()), bytes)),
        "read " + path);
  return part;
}

bool cosine_metric(const std::string &name) {
  if (name == "cosine") return true;
  if (name == "l2") return false;
  throw std::invalid_argument("metric must be 'l2' or 'cosine'");
}

// Work on private copies so reading options never modifies caller dictionaries.
template <typename T>
void take(py::dict &options, const char *key, T &value) {
  if (options.contains(key)) value = options.attr("pop")(key).cast<T>();
}

void check_keys(const py::dict &options) {
  if (!options.empty())
    throw std::invalid_argument(
        "Unknown QG option: " +
        py::str(options.begin()->first).cast<std::string>());
}

ailego::Params page_params(py::dict opts) {
  auto d = py::dict(opts.attr("copy")());
  ailego::Params params;
  const auto &policy = quantized_graph_pages::kPolicy;
  if (d.contains(policy.c_str()))
    params.set(policy, d.attr("pop")(policy).cast<std::string>());
  for (const auto &key :
       {quantized_graph_pages::kMinimum, quantized_graph_pages::kFloor}) {
    if (d.contains(key.c_str()))
      params.set(key, d.attr("pop")(key).cast<double>());
  }
  check_keys(d);
  return params;
}

QuantizedGraphBuilder::Options build_options(const py::dict &options) {
  QuantizedGraphBuilder::Options o;
  auto d = py::dict(options.attr("copy")());
  take(d, "threads", o.threads);
  take(d, "passes", o.passes);
  take(d, "search_list_size", o.search_list_size);
  take(d, "seed", o.seed);
  take(d, "router_anchors", o.router_anchors);
  take(d, "router_dimension", o.router_dimension);
  take(d, "router_seed", o.router_seed);
  py::dict encoding;
  take(d, "encoding", encoding);
  check_keys(d);
  auto e = py::dict(encoding.attr("copy")());
  auto &enc = o.encoding;
  using Encoding = QuantizedGraphEncodingOptions;
  if (e.contains("codec")) {
    auto codec = e.attr("pop")("codec").cast<std::string>();
    if (!Encoding::parse_name(codec, enc.codec))
      throw std::invalid_argument("Unknown QG codec: " + codec);
  }
  if (e.contains("codebook")) {
    auto book = e.attr("pop")("codebook").cast<std::string>();
    if (!Encoding::parse_name(book, enc.codebook))
      throw std::invalid_argument("Unknown QG codebook: " + book);
  }
  take(e, "max_neighbors", enc.max_neighbors);
  take(e, "rotation_views", enc.rotation_views);
  take(e, "block_dimension", enc.block_dimension);
  take(e, "rank", enc.rank);
  take(e, "training_samples", enc.training_samples);
  take(e, "center_weight", enc.center_weight);
  take(e, "seed", enc.seed);
  check_keys(e);
  return o;
}

class PythonQuantizedGraph {
 public:
  PythonQuantizedGraph(const Array &rows, const std::string &metric,
                       const py::dict &config, const py::dict &pages) {
    if (rows.ndim() != 2 || !rows.shape(0) || rows.shape(0) >= UINT32_MAX ||
        !rows.shape(1) || rows.shape(1) > 4096)
      throw std::invalid_argument("invalid FP32 corpus shape");
    const bool cosine = cosine_metric(metric);
    check(QuantizedGraphSupported(), "QG unavailable on this CPU");
    auto options = build_options(config);
    options.page_params = page_params(pages);
    QuantizedGraphBuilder::Result built;
    py::gil_scoped_release release;
    std::vector<float> vectors(rows.data(), rows.data() + rows.size());
    status(QuantizedGraphBuilder::build(std::move(vectors), rows.shape(1),
                                        cosine, options, built));
    data_ = std::move(built.data);
    index_ = std::move(built.index);
    init_query();
  }

  PythonQuantizedGraph(const std::string &prefix, size_t n, size_t dim,
                       const std::string &metric, const py::dict &pages) {
    check(QuantizedGraphSupported(), "QG unavailable on this CPU");
    check(n > 0 && n < UINT32_MAX && dim > 0 && dim <= 4096,
          "invalid snapshot shape");
    const bool cosine = cosine_metric(metric);
    auto params = page_params(pages);
    py::gil_scoped_release release;
    auto data = std::make_shared<QuantizedGraphData>();
    data->dimension = dim;
    data->cosine = cosine;
    data->vectors = read_part<float>(prefix + ".vectors", n * dim);
    data->offsets = read_part<uint64_t>(prefix + ".offsets", n + 1);
    data->edges = read_part<uint32_t>(prefix + ".edges", n * 256);
    check(data->vectors.size() == n * dim && data->offsets.size() == n + 1 &&
              data->offsets.front() == 0 &&
              data->offsets.back() == data->edges.size(),
          "invalid CSR snapshot");
    for (size_t i = 0; i < n; ++i) {
      check(data->offsets[i] <= data->offsets[i + 1] &&
                data->offsets[i + 1] - data->offsets[i] <= 256,
            "invalid CSR degree");
    }
    for (auto id : data->edges) check(id < n, "invalid neighbor ID");
    auto bytes = read_part<char>(prefix + ".router", size_t{1} << 30);
    std::string router(bytes.begin(), bytes.end());
    auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
    check(bool(storage), "missing snapshot reader");
    status(storage->init({}));
    status(storage->open(prefix + ".qg", false));
    status(QuantizedGraphIndex::load(
        *data, n, dim, cosine, dim * sizeof(float), router, false,
        storage->get(QuantizedGraphIndex::kSnapshotSegment), index_, params));
    status(storage->close());
    data_ = std::move(data);
    init_query();
  }

  void configure(const py::dict &options) {
    auto d = py::dict(options.attr("copy")());
    QuantizedGraphSearchOptions search;
    size_t capacity = 128;
    take(d, "capacity", capacity);
    take(d, "margin", search.margin);
    take(d, "max_expansions", search.max_expansions);
    take(d, "route_beam", search.route_beam);
    take(d, "bucket_width", search.bucket_width);
    take(d, "uncertainty_bonus", search.uncertainty_bonus);
    check_keys(d);
    if (!capacity || capacity > UINT32_MAX)
      throw std::invalid_argument("capacity must be a positive uint32");
    std::lock_guard<std::mutex> lock(mutex_);
    options_ = search;
    capacity_ = capacity;
  }

  py::array_t<uint64_t> search(const Array &input, size_t k) {
    if (input.ndim() != 1 || size_t(input.size()) != data_->dimension || !k ||
        k > data_->size())
      throw std::invalid_argument("invalid query shape or k");

    py::array_t<uint64_t> result(k);
    auto *ids = result.mutable_data();
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      const float *q = input.data();
      std::vector<float> normalized;
      if (data_->cosine) {
        normalized.assign(q, q + data_->dimension);
        double norm = 0;
        for (float x : normalized) norm += double(x) * x;
        if (!(norm > 0 && std::isfinite(norm)))
          throw std::invalid_argument("invalid cosine query norm");
        const auto scale = 1 / std::sqrt(norm);
        for (float &x : normalized) x *= scale;
        q = normalized.data();
      } else if (!std::all_of(q, q + data_->dimension, [this](float x) {
                   return std::abs(x) <= l2_limit_;
                 })) {
        throw std::invalid_argument(
            "query coordinate exceeds the finite L2 range");
      }
      status(query_->reset(q, k, capacity_, options_));
      status(index_->search(*data_, *query_));
      const auto &hits = query_->results();
      check(hits.size() >= k, "insufficient results");
      for (size_t i = 0; i < k; ++i) ids[i] = hits[i].first;
    }
    return result;
  }

 private:
  void init_query() {
    l2_limit_ = std::sqrt(double(std::numeric_limits<float>::max()) /
                          (16 * data_->dimension));
    auto metric = IndexFactory::CreateMetric(
        data_->cosine ? "InnerProduct" : "SquaredEuclidean");
    check(bool(metric), "missing exact metric");
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP32, data_->dimension);
    status(metric->init(meta, {}));
    auto exact = [distance = metric->distance(), cosine = data_->cosine](
                     const void *a, const void *b, size_t dim, float *out) {
      distance(a, b, dim, out);
      if (cosine) *out += 1.f;
    };
    query_ = std::make_unique<QuantizedGraphQuery>(data_->dimension, exact);
  }

  std::shared_ptr<const QuantizedGraphData> data_;
  std::shared_ptr<QuantizedGraphIndex> index_;
  std::unique_ptr<QuantizedGraphQuery> query_;
  QuantizedGraphSearchOptions options_;
  double l2_limit_;
  std::mutex mutex_;
  size_t capacity_ = 128;
};
#else
struct PythonQuantizedGraph {};
#endif
}  // namespace

void RegisterQuantizedGraph(py::module_ &m) {
  auto graph = py::class_<PythonQuantizedGraph>(
      m, "QuantizedGraph",
      "Standalone FP32 graph with input row ordinals as IDs.");
  graph.def_static("supported", &QuantizedGraphSupported);
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
  graph
      .def_static(
          "build",
          [](const Array &rows, const std::string &metric,
             const py::dict &options, const py::dict &pages) {
            return std::make_unique<PythonQuantizedGraph>(rows, metric, options,
                                                          pages);
          },
          py::arg("vectors").noconvert(), py::kw_only(),
          py::arg("metric") = "l2", py::arg("options") = py::dict(),
          py::arg("page_params") = py::dict(),
          "Build from a contiguous float32 matrix. The graph owns its rows.")
      .def_static(
          "open",
          [](const std::string &prefix, size_t count, size_t dimension,
             const std::string &metric, const py::dict &pages) {
            return std::make_unique<PythonQuantizedGraph>(
                prefix, count, dimension, metric, pages);
          },
          py::arg("prefix"), py::kw_only(), py::arg("count"),
          py::arg("dimension"), py::arg("metric") = "l2",
          py::arg("page_params") = py::dict(),
          "Open an independent FP32 snapshot (.vectors, .edges, .offsets, "
          ".router, .qg).")
      .def("configure", &PythonQuantizedGraph::configure, py::arg("options"),
           "Replace search settings. Omitted options use native defaults.")
      .def("search", &PythonQuantizedGraph::search,
           py::arg("query").noconvert(), py::arg("k") = 10,
           "Return the nearest row IDs in distance order.");
#else
  auto unavailable = [](py::args, py::kwargs) {
    throw std::runtime_error(
        "This zvec build does not support quantized graphs");
  };
  graph.def_static("open", unavailable).def_static("build", unavailable);
#endif
}
}  // namespace zvec
