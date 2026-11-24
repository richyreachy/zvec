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
#include "quantized_graph_builder.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <atomic>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <type_traits>
#include <utility>
#include "quantized_graph_params.h"
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/turbo/turbo.h>

namespace zvec::core {
namespace {
template <class T>
bool read_build_param(const ailego::Params &params, const std::string &key,
                      T &out) {
  const auto *value = params.hypercube().get(key);
  if constexpr (std::is_enum_v<T>) {
    return value->compatible<std::string>() &&
           QuantizedGraphEncodingOptions::parse_name(
               value->unsafe_cast<std::string>(), out);
  } else {
    const bool unsigned_int = value->compatible<unsigned char>() ||
                              value->compatible<unsigned short>() ||
                              value->compatible<unsigned int>() ||
                              value->compatible<unsigned long>() ||
                              value->compatible<unsigned long long>();
    const bool signed_int = value->compatible<signed char>() ||
                            value->compatible<short>() || value->compatible<int>() ||
                            value->compatible<long>() ||
                            value->compatible<long long>();
    if constexpr (std::is_floating_point_v<T>) {
      return (signed_int || unsigned_int || value->compatible<double>() ||
              value->compatible<float>()) && params.get(key, &out);
    } else {
      uint64_t n;
      if (signed_int) {
        int64_t i;
        if (!params.get(key, &i) || i < 0) return false;
        n = static_cast<uint64_t>(i);
      } else if (!unsigned_int || !params.get(key, &n)) {
        return false;
      }
      if (n > std::numeric_limits<T>::max() || n > INT64_MAX) return false;
      out = static_cast<T>(n);
      return true;
    }
  }
}

bool read_build_params(const ailego::Params &params,
                       QuantizedGraphBuilder::Options &options) {
  size_t read = 0;
  auto load = [&](const std::string &key, auto &field) {
    if (params.has(key) && read_build_param(params, key, field)) ++read;
  };
  qg_build_params::build_fields(options, load);
  qg_build_params::encoding_fields(options.encoding, load);
  return read == params.hypercube().cubes().size() && options.valid();
}

template <class Emit>
bool write_part(IndexStorage &store, const char *name, Emit emit) {
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

// Vamana-style refinement over the owned corpus and temporary adjacency.
class VamanaGraph {
  using Candidate = std::pair<float, uint32_t>;

  struct Context {
    explicit Context(size_t count) : visited(count, 0) {}
    std::vector<uint32_t> visited;
    uint32_t generation = 0;
    std::vector<Candidate> frontier, best;
    std::vector<uint32_t> row;

    void reset() {
      if (++generation == 0) {
        std::fill(visited.begin(), visited.end(), 0);
        generation = 1;
      }
      frontier.clear();
      best.clear();
    }
  };

 public:
  VamanaGraph(QuantizedGraphData &data,
              const QuantizedGraphBuilder::Options &opts,
              turbo::DistanceFunc distance)
      : data_(data),
        opts_(opts),
        distance_(distance),
        rows_(data.size()),
        locks_(std::min<size_t>(data.size(), 4096)) {}

  int build() {
    const size_t n = data_.size();
    std::vector<double> mean(data_.dimension, 0);
    for (size_t id = 0; id < n; ++id)
      for (size_t d = 0; d < data_.dimension; ++d)
        mean[d] += data_.vectors[id * data_.dimension + d] / double(n);

    double best = std::numeric_limits<double>::infinity();
    for (uint32_t id = 0; id < n; ++id) {
      double d = 0;
      for (size_t j = 0; j < data_.dimension; ++j) {
        const double delta = data_.vectors[id * data_.dimension + j] - mean[j];
        d += delta * delta;
      }
      if (d < best) {
        best = d;
        entry_ = id;
      }
    }

    const size_t workers = std::min(n, opts_.threads);
    std::vector<Context> contexts;
    for (size_t i = 0; i < workers; ++i) contexts.emplace_back(n);
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), uint32_t{0});
    std::mt19937 random(opts_.seed);
    for (size_t pass = 0; pass < opts_.passes; ++pass) {
      std::shuffle(order.begin(), order.end(), random);
      const float alpha = pass == 0 ? 1.f : 1.2f;
      std::atomic<size_t> next{0};
      std::atomic<int> error{0};
      auto worker = [&](size_t part) {
        try {
          while (!error.load()) {
            const size_t i = next.fetch_add(1);
            if (i >= n) break;
            refine(order[i], alpha, contexts[part]);
          }
        } catch (const std::bad_alloc &) {
          error.store(IndexError_NoMemory);
        } catch (const std::exception &) {
          error.store(IndexError_Runtime);
        }
      };
      std::vector<std::thread> pool;
      try {
        for (size_t i = 0; i < workers; ++i) pool.emplace_back(worker, i);
      } catch (const std::exception &) {
        error.store(IndexError_NoMemory);
      }
      for (auto &thread : pool) thread.join();
      if (error.load()) return error.load();
    }

    data_.offsets.reserve(n + 1);
    // Retain reachability for duplicate vectors and separated clusters.
    data_.offsets.push_back(0);
    for (uint32_t id = 0; id < n; ++id) {
      auto &row = rows_[id];
      const uint32_t successor = (size_t(id) + 1) % n;
      if (n > 1 && std::find(row.begin(), row.end(), successor) == row.end()) {
        if (row.size() == opts_.encoding.max_neighbors) row.pop_back();
        row.push_back(successor);
      }
      data_.edges.insert(data_.edges.end(), row.begin(), row.end());
      data_.offsets.push_back(data_.edges.size());
    }
    return 0;
  }

 private:
  float distance(uint32_t a, uint32_t b) const {
    float out;
    distance_(data_.vectors.data() + size_t(a) * data_.dimension,
              data_.vectors.data() + size_t(b) * data_.dimension,
              data_.dimension, &out);
    return out;
  }

  void search(uint32_t id, Context &ctx) {
    ctx.reset();
    auto insert = [&](uint32_t v) {
      if (ctx.visited[v] == ctx.generation) return;
      ctx.visited[v] = ctx.generation;
      Candidate c{distance(id, v), v};
      if (ctx.best.size() == opts_.search_list_size && c > ctx.best.front())
        return;
      ctx.best.push_back(c);
      std::push_heap(ctx.best.begin(), ctx.best.end());
      if (ctx.best.size() > opts_.search_list_size) {
        std::pop_heap(ctx.best.begin(), ctx.best.end());
        ctx.best.pop_back();
      }
      ctx.frontier.push_back(c);
      std::push_heap(ctx.frontier.begin(), ctx.frontier.end(),
                     std::greater<Candidate>());
    };
    insert(entry_);
    while (!ctx.frontier.empty()) {
      const auto current = ctx.frontier.front();
      if (ctx.best.size() == opts_.search_list_size &&
          current.first > ctx.best.front().first)
        break;
      std::pop_heap(ctx.frontier.begin(), ctx.frontier.end(),
                    std::greater<Candidate>());
      ctx.frontier.pop_back();
      {
        std::lock_guard<std::mutex> lock(
            locks_[current.second % locks_.size()]);
        ctx.row = rows_[current.second];
      }
      for (uint32_t v : ctx.row) insert(v);
    }
  }

  std::vector<uint32_t> prune(uint32_t id, std::vector<Candidate> &candidates,
                              float alpha) const {
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [id](const auto &c) { return c.second == id; }),
        candidates.end());
    constexpr size_t kPruneCandidateLimit = 750;
    if (candidates.size() > kPruneCandidateLimit)
      candidates.resize(kPruneCandidateLimit);

    std::vector<float> occlusion(candidates.size(), 0);
    std::vector<uint32_t> result;
    const size_t degree = opts_.encoding.max_neighbors;
    for (float a = 1; a <= alpha + 1e-6f && result.size() < degree; a *= 1.2f) {
      for (size_t i = 0; i < candidates.size() && result.size() < degree; ++i) {
        if (occlusion[i] > a) continue;
        occlusion[i] = std::numeric_limits<float>::infinity();
        result.push_back(candidates[i].second);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
          if (occlusion[j] > alpha) continue;
          const float d = distance(candidates[i].second, candidates[j].second);
          occlusion[j] = d <= 0
                             ? std::numeric_limits<float>::infinity()
                             : std::max(occlusion[j], candidates[j].first / d);
        }
      }
    }
    return result;
  }

  void refine(uint32_t id, float alpha, Context &ctx) {
    search(id, ctx);
    {
      std::lock_guard<std::mutex> lock(locks_[id % locks_.size()]);
      for (uint32_t v : rows_[id]) ctx.best.emplace_back(distance(id, v), v);
      rows_[id] = prune(id, ctx.best, alpha);
      ctx.row = rows_[id];
    }
    for (uint32_t v : ctx.row) {
      std::lock_guard<std::mutex> lock(locks_[v % locks_.size()]);
      auto &row = rows_[v];
      if (std::find(row.begin(), row.end(), id) != row.end()) continue;
      if (row.size() < opts_.encoding.max_neighbors) {
        row.push_back(id);
      } else {
        std::vector<Candidate> candidates;
        for (uint32_t neighbor : row)
          candidates.emplace_back(distance(v, neighbor), neighbor);
        candidates.emplace_back(distance(v, id), id);
        row = prune(v, candidates, alpha);
      }
    }
  }

  QuantizedGraphData &data_;
  const QuantizedGraphBuilder::Options &opts_;
  turbo::DistanceFunc distance_;
  std::vector<std::vector<uint32_t>> rows_;
  std::vector<std::mutex> locks_;
  uint32_t entry_ = 0;
};

}  // namespace

int QuantizedGraphBuilder::build(std::vector<float> vectors, size_t dimension,
                                 bool cosine, const Options &options,
                                 Result &result) {
  if (!QuantizedGraphSupported()) return IndexError_Unsupported;
  if (!dimension || dimension > 4096 || vectors.empty() ||
      vectors.size() % dimension || vectors.size() / dimension >= UINT32_MAX ||
      !options.valid())
    return IndexError_InvalidArgument;

  try {
    Result next;
    auto owned = std::make_shared<QuantizedGraphData>();
    auto &data = *owned;
    data.dimension = dimension;
    data.cosine = cosine;
    data.vectors = std::move(vectors);
    const double limit =
        std::sqrt(double(std::numeric_limits<float>::max()) / (16 * dimension));
    for (size_t id = 0; id < data.size(); ++id) {
      float *v = data.vectors.data() + id * dimension;
      double norm = 0;
      for (size_t d = 0; d < dimension; ++d) {
        if (!std::isfinite(v[d]) || std::abs(double(v[d])) > limit)
          return IndexError_InvalidArgument;
        norm += double(v[d]) * v[d];
      }
      if (cosine) {
        if (norm == 0) return IndexError_InvalidArgument;
        const double scale = 1.0 / std::sqrt(norm);
        for (size_t d = 0; d < dimension; ++d) v[d] *= scale;
      }
    }

    auto distance = turbo::get_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kFp32,
        turbo::QuantizeType::kRaw);
    if (!distance) return IndexError_Unsupported;
    int ret = VamanaGraph(data, options, distance).build();
    if (ret != 0) return ret;
    auto encoding = options.encoding;
    encoding.threads = options.threads;
    ret = QuantizedGraphIndex::train_codec(data, data.size(), dimension, cosine,
                                           false, encoding, next.codec);
    if (ret != 0) return ret;
    if (options.router_anchors) {
      ret = QuantizedGraphIndex::train_router(
          data, data.size(), dimension, cosine, options.router_anchors,
          std::min(options.router_dimension, dimension), options.router_seed,
          next.router, false, encoding.rotation_views);
      if (ret != 0) return ret;
    }
    next.index = std::make_shared<QuantizedGraphIndex>(
        dimension, encoding.max_neighbors, cosine, dimension * sizeof(float),
        next.router, false, &next.codec);
    ret = next.index->prebuild(data, data.size(), options.threads,
                               options.page_params);
    if (ret != 0) return ret;
    next.data = std::move(owned);
    result = std::move(next);
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &error) {
    LOG_ERROR("QG raw-vector build failed: %s", error.what());
    return IndexError_Runtime;
  }
}

int QuantizedGraphBuilder::init(const IndexMeta &meta,
                                const ailego::Params &params) {
  Options options;
  if (!QuantizedGraphSupported()) return IndexError_Unsupported;
  if (!QuantizedGraphSupportsMeta(meta) ||
      !read_build_params(params, options))
    return IndexError_InvalidArgument;
  cleanup();
  meta_ = meta;
  meta_.set_builder(name().empty() ? "QuantizedGraphBuilder" : name(), 0,
                    params);
  meta_.set_searcher("QuantizedGraphSearcher", 0, ailego::Params{});
  meta_.set_streamer("QuantizedGraphStreamer", 0, ailego::Params{});
  options_ = std::move(options);
  initialized_ = true;
  return 0;
}

int QuantizedGraphBuilder::cleanup() {
  built_ = {};
  keys_.clear();
  stats_.clear();
  initialized_ = false;
  return 0;
}

bool QuantizedGraphBuilder::compatible(const IndexHolder::Pointer &h) const {
  return initialized_ && h && h->count() && h->count() < UINT32_MAX &&
         h->dimension() == meta_.dimension() &&
         h->data_type() == IndexMeta::DT_FP32 &&
         h->element_size() == meta_.element_size() &&
         h->count() <= SIZE_MAX / meta_.element_size();
}

int QuantizedGraphBuilder::train(IndexThreads::Pointer,
                                 IndexHolder::Pointer h) {
  return compatible(h) ? 0 : IndexError_InvalidArgument;
}

int QuantizedGraphBuilder::build(IndexThreads::Pointer,
                                 IndexHolder::Pointer h) {
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
  Result result;
  int rc = build(std::move(rows), dim, meta_.metric_name() == "Cosine",
                 options_, result);
  if (rc) return rc;
  keys_ = std::move(keys);
  built_ = std::move(result);
  stats_.set_built_count(n);
  return 0;
}

template <class Part>
static int serialize(const QuantizedGraphBuilder::Result &built,
                     const std::vector<uint64_t> &keys, Part part) {
  if (!built.index || !built.data || !built.data->size() ||
      (!keys.empty() && keys.size() != built.data->size()))
    return IndexError_InvalidArgument;
  auto buffer = [&](const char *name, const auto &values) {
    return part(name, [&](auto &write) {
      return write(values.data(), values.size() * sizeof(*values.data()));
    });
  };
  if (keys.empty() || *std::max_element(keys.begin(), keys.end()) <= UINT32_MAX) {
    std::vector<uint32_t> compact(built.data->size());
    if (keys.empty())
      std::iota(compact.begin(), compact.end(), uint32_t{0});
    else
      std::copy(keys.begin(), keys.end(), compact.begin());
    if (!buffer(QuantizedGraphIndex::kKeysSegment, compact)) return IndexError_WriteData;
  } else if (!buffer(QuantizedGraphIndex::kKeys64Segment, keys))
    return IndexError_WriteData;
  if (!buffer(QuantizedGraphIndex::kVectorsSegment, built.data->vectors) ||
      !buffer(QuantizedGraphIndex::kOffsetsSegment, built.data->offsets) ||
      !buffer(QuantizedGraphIndex::kEdgesSegment, built.data->edges) ||
      !buffer(QuantizedGraphIndex::kRouterSegment, built.router) ||
      !part(QuantizedGraphIndex::kSnapshotSegment,
            [&](auto &write) { return built.index->write_snapshot(write); }))
    return IndexError_WriteData;
  return 0;
}

static int dump_result(const QuantizedGraphBuilder::Result &built,
                       const std::vector<uint64_t> &keys, const IndexMeta &meta,
                       const IndexDumper::Pointer &dumper) {
  if (!dumper || !built.index) return IndexError_InvalidArgument;
  int rc = IndexHelper::SerializeToDumper(meta, dumper.get());
  if (rc) return rc;
  return serialize(built, keys, [&](const char *name, auto emit) {
    quantized_graph_detail::SnapshotWriter write{dumper};
    if (!emit(write)) return false;
    const size_t pad = (32 - write.bytes % 32) % 32;
    const char zeros[32]{};
    return (!pad || dumper->write(zeros, pad) == pad) &&
           dumper->append(name, write.bytes, pad, write.crc) == 0;
  });
}

int QuantizedGraphBuilder::dump(const IndexDumper::Pointer &dumper) {
  int rc = dump_result(built_, keys_, meta_, dumper);
  if (!rc) stats_.set_dumped_count(keys_.size());
  return rc;
}

int QuantizedGraphBuilder::dump(IndexStorage &storage) {
  int rc = serialize(built_, keys_, [&](const char *name, auto emit) {
    return write_part(storage, name, emit);
  });
  if (!rc) stats_.set_dumped_count(keys_.size());
  return rc;
}

bool QuantizedGraphBuilder::Result::unique_keys(const std::vector<uint64_t> &keys) {
  if (std::is_sorted(keys.begin(), keys.end()))
    return std::adjacent_find(keys.begin(), keys.end()) == keys.end();
  auto sorted = keys;
  std::sort(sorted.begin(), sorted.end());
  return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}

int QuantizedGraphBuilder::Result::save(
    const std::string &path, const std::vector<uint64_t> &keys) const {
  if (!data || !index || path.empty() ||
      (!keys.empty() && (keys.size() != data->size() || !unique_keys(keys))))
    return IndexError_InvalidArgument;
  IndexMeta meta(IndexMeta::DT_FP32, data->dimension);
  if (data->cosine) meta.set_metric("Cosine", 0, ailego::Params{});
  meta.set_builder("QuantizedGraphBuilder", 0, build_params);
  meta.set_searcher("QuantizedGraphSearcher", 0, ailego::Params{});
  meta.set_streamer("QuantizedGraphStreamer", 0, ailego::Params{});
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  if (!dumper) return IndexError_NoExist;
  int rc = dumper->init({});
  if (rc) return rc;

  std::string tmp = path + ".XXXXXX";
  int fd = mkstemp(tmp.data());
  if (fd < 0) return IndexError_CreateFile;
  struct Cleanup {
    const std::string &path;
    ~Cleanup() { std::remove(path.c_str()); }
  } cleanup{tmp};
  if (::close(fd) != 0) return IndexError_CloseFile;
  if ((rc = dumper->create(tmp)) || (rc = dump_result(*this, keys, meta, dumper)) ||
      (rc = dumper->close()))
    return rc;
  return std::rename(tmp.c_str(), path.c_str()) ? IndexError_WriteData : 0;
}

INDEX_FACTORY_REGISTER_BUILDER(QuantizedGraphBuilder);
}  // namespace zvec::core
