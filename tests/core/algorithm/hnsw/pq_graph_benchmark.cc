// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND.

// Reproducible graph-engine comparison, independent of HNSW build randomness.
// Usage: pq_graph_benchmark [vectors=4096] [queries=256] [dim=64]
//                          [training_trials=3] [timing_repeats=3]
// CSV goes to stdout. Dataset/graph/query seeds are fixed; existing KMC2
// training remains random, so run multiple independent codebooks.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "algorithm/quantized_graph/pq_graph_codec.h"
#include "algorithm/quantized_graph/pq_residual_graph_codec.h"
#include "algorithm/quantized_graph/quantized_graph.h"

namespace {
using namespace zvec::core;
using Clock = std::chrono::steady_clock;
using Ranked = std::pair<float, uint32_t>;
constexpr size_t kDegree = 32, kTopk = 10;
float distance(const float *a, const float *b, size_t dim, bool cosine) {
  float result = cosine ? 1 : 0;
  for (size_t d = 0; d < dim; ++d) {
    const float diff = a[d] - b[d];
    result += cosine ? -a[d] * b[d] : diff * diff;
  }
  return result;
}
void check(int ret) {
  if (ret) throw std::runtime_error("index error " + std::to_string(ret));
}
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start)
      .count();
}
struct Graph {
  struct Vector {
    const float *ptr = nullptr;
    const void *data() const {
      return ptr;
    }
  };
  size_t dim;
  std::vector<std::vector<float>> values;
  std::vector<std::vector<uint32_t>> links;
  bool valid(uint32_t id) const {
    return id < values.size();
  }
  int get_vector(uint32_t id, Vector &out) const {
    out.ptr = values[id].data();
    return 0;
  }
  const auto &neighbors(uint32_t id) const {
    return links[id];
  }
};
struct Query {
  struct Results {
    size_t capacity = 32;
    std::vector<Ranked> values;
    size_t limit() const {
      return capacity;
    }
    size_t size() const {
      return values.size();
    }
    void clear() {
      values.clear();
    }
    void emplace(uint32_t id, float d) {
      Ranked item{d, id};
      if (values.size() == capacity) {
        if (item >= values.front()) return;
        std::pop_heap(values.begin(), values.end());
        values.pop_back();
      }
      values.push_back(item);
      std::push_heap(values.begin(), values.end());
    }
  } results;
  const float *data = nullptr;
  size_t dim = 0, scans = 0, expands = 0;
  bool cosine = false;
  Results &reset_results() {
    scans = expands = 0;
    return results;
  }
  size_t topk() const {
    return kTopk;
  }
  const void *query() const {
    return data;
  }
  bool excluded(uint32_t) const {
    return false;
  }
  bool reach_scan_limit() const {
    return false;
  }
  void on_expand() {
    ++expands;
  }
  float distance(const void *v) {
    ++scans;
    return ::distance(data, static_cast<const float *>(v), dim, cosine);
  }
};
struct Workload {
  Graph graph;
  std::vector<std::vector<float>> queries;
  std::vector<std::vector<uint32_t>> truth;
};
Workload make_workload(size_t n, size_t nq, size_t dim, bool clustered,
                       bool cosine) {
  std::mt19937 rng(20261023);
  std::normal_distribution<float> normal(0, 1);
  std::vector<std::vector<float>> centers(16, std::vector<float>(dim));
  for (auto &c : centers)
    for (auto &x : c) x = normal(rng) * 2;
  auto make_vector = [&](size_t i) {
    std::vector<float> v(dim);
    for (size_t d = 0; d < dim; ++d)
      v[d] = normal(rng) + (clustered ? centers[i % centers.size()][d] : 0);
    if (cosine) {
      double norm = 0;
      for (float x : v) norm += double(x) * x;
      for (auto &x : v) x /= std::sqrt(norm);
    }
    return v;
  };
  Workload w;
  w.graph.dim = dim;
  for (size_t i = 0; i < n; ++i) w.graph.values.push_back(make_vector(i));
  for (size_t i = 0; i < nq; ++i) w.queries.push_back(make_vector(i));
  w.graph.links.resize(n);
  // Two ring edges ensure connectivity, 22 exact local neighbors, eight
  // long-range edges. Both codecs see precisely the same graph and entry.
  for (size_t i = 0; i < n; ++i) {
    auto &links = w.graph.links[i];
    links = {uint32_t((i + 1) % n), uint32_t((i + n - 1) % n)};
    std::vector<Ranked> ranked;
    for (size_t j = 0; j < n; ++j)
      if (j != i)
        ranked.emplace_back(distance(w.graph.values[i].data(),
                                     w.graph.values[j].data(), dim, cosine),
                            j);
    std::partial_sort(ranked.begin(), ranked.begin() + kDegree, ranked.end());
    for (size_t j = 0; links.size() < 24; ++j)
      if (std::find(links.begin(), links.end(), ranked[j].second) ==
          links.end())
        links.push_back(ranked[j].second);
    while (links.size() < kDegree) {
      uint32_t id = rng() % n;
      if (id != i && std::find(links.begin(), links.end(), id) == links.end())
        links.push_back(id);
    }
  }
  for (const auto &q : w.queries) {
    std::vector<Ranked> ranked;
    for (size_t j = 0; j < n; ++j)
      ranked.emplace_back(
          distance(q.data(), w.graph.values[j].data(), dim, cosine), j);
    std::partial_sort(ranked.begin(), ranked.begin() + kTopk, ranked.end());
    std::vector<uint32_t> ids;
    for (size_t k = 0; k < kTopk; ++k) ids.push_back(ranked[k].second);
    w.truth.push_back(std::move(ids));
  }
  return w;
}
template <class Codec>
void run(const Workload &w, const char *name, const char *dataset, bool cosine,
         size_t trial, size_t repeats) {
  Codec codec(w.graph.dim, cosine);
  auto start = Clock::now();
  check(codec.train(w.graph, w.graph.values.size(), 2));
  const double train_ms = elapsed(start) / 1000;
  const size_t payload = w.graph.values.size() * codec.batch_bytes();
  // Mirrors QuantizedGraph's documented 64-byte header/aligned fixed stride.
  const size_t stride =
      (64 + 4 * w.graph.dim + 4 * kDegree + codec.batch_bytes() + 63) &
      ~size_t{63};
  QuantizedGraph<Codec> graph(w.graph.dim, std::move(codec), kDegree);
  start = Clock::now();
  check(graph.prebuild(w.graph, w.graph.values.size(), 2));
  const double build_ms = elapsed(start) / 1000;
  Query q;
  q.dim = w.graph.dim;
  q.cosine = cosine;
  for (size_t ef : {16U, 32U, 64U, 128U, 256U, 512U, 1024U, 2048U, 4096U}) {
    q.results.capacity = ef;
    for (size_t i = 0; i < std::min(size_t{32}, w.queries.size()); ++i) {
      q.data = w.queries[i].data();
      check(graph.search(graph.entry(), w.graph, q));
    }
    std::vector<double> times;
    size_t hits = 0, scans = 0, expands = 0;
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
      for (size_t j = 0; j < w.queries.size(); ++j) {
        const size_t i = (j + repeat * 37) % w.queries.size();
        q.data = w.queries[i].data();
        start = Clock::now();
        check(graph.search(graph.entry(), w.graph, q));
        times.push_back(elapsed(start));
        scans += q.scans;
        expands += q.expands;
        auto result = q.results.values;
        std::sort(result.begin(), result.end());
        for (size_t k = 0; k < std::min(kTopk, result.size()); ++k)
          hits += std::find(w.truth[i].begin(), w.truth[i].end(),
                            result[k].second) != w.truth[i].end();
      }
    }
    const double total = std::accumulate(times.begin(), times.end(), 0.0);
    std::sort(times.begin(), times.end());
    auto percentile = [&](double p) {
      return times[size_t(p * (times.size() - 1))];
    };
    std::cout << dataset << ',' << (cosine ? "cosine" : "l2") << ',' << name
              << ',' << trial << ',' << ef << ',' << w.graph.values.size()
              << ',' << w.graph.dim << ',' << w.queries.size() << ','
              << train_ms << ',' << build_ms << ',' << payload << ','
              << stride * w.graph.values.size() << ','
              << double(hits) / (times.size() * kTopk) << ','
              << 1e6 * times.size() / total << ',' << percentile(.5) << ','
              << percentile(.95) << ',' << percentile(.99) << ','
              << double(scans) / times.size() << ','
              << double(expands) / times.size() << '\n';
    std::cout.flush();
  }
}
}  // namespace
int main(int argc, char **argv) {
  size_t n = argc > 1 ? std::stoul(argv[1]) : 4096;
  size_t nq = argc > 2 ? std::stoul(argv[2]) : 256;
  size_t dim = argc > 3 ? std::stoul(argv[3]) : 64;
  size_t trials = argc > 4 ? std::stoul(argv[4]) : 3;
  size_t repeats = argc > 5 ? std::stoul(argv[5]) : 3;
  if (n < 64 || !nq || !dim || dim > 4096 || !trials || !repeats) return 2;
  std::cout << std::fixed << std::setprecision(6)
            << "dataset,metric,codec,trial,ef,n,dim,nq,train_ms,prebuild_ms,"
               "code_bytes,arena_bytes_estimate,recall10,qps,p50_us,p95_us,p99_"
               "us,exact_per_query,expands_per_query\n";
  for (bool clustered : {false, true})
    for (bool cosine : {false, true}) {
      auto w = make_workload(n, nq, dim, clustered, cosine);
      const char *dataset = clustered ? "clustered" : "gaussian";
      for (size_t trial = 0; trial < trials; ++trial) {
        // Alternate order to reduce systematic thermal/order bias.
        if (trial % 2 == 0)
          run<PQGraphCodec>(w, "pq", dataset, cosine, trial, repeats);
        run<PQResidualGraphCodec>(w, "pq_residual", dataset, cosine, trial,
                                  repeats);
        if (trial % 2 != 0)
          run<PQGraphCodec>(w, "pq", dataset, cosine, trial, repeats);
      }
    }
}
