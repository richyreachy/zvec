// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <rabitqlib/third/Eigen/Eigenvalues>
#include <turbo/preprocessor/symqg_rotator/symqg_rotator.h>
#include "quantized_graph_build_options.h"
#include "quantized_graph_neighbors.h"
#include "quantized_graph_sq8_center.h"

namespace zvec::core {

// Database-only training for SymQG's affine / low-rank block codec. Nearby
// database vectors act as proxy queries; no query set or ground truth is read.
// Source adapters expose FP32 coordinates, widening stored FP16 on demand.
class SymQGCodecBuilder {
 public:
  struct Result {
    std::string data;
    size_t samples = 0;
    float lambda = 0;
    double captured_energy = 0;
  };

  template <class Graph>
  static Result train(const Graph &graph, size_t docs, size_t dimension,
                      bool cosine, const QuantizedGraphBuildOptions &options) {
    using Options = QuantizedGraphBuildOptions;
    if (!options.valid() || !dimension || dimension > 4096 || !docs ||
        docs >= UINT32_MAX)
      throw std::invalid_argument("invalid codec training options or shape");
    if (options.codec == Options::Codec::Native) return {};

    auto read_vector = [&](uint32_t id) {
      typename Graph::Vector raw;
      if (id >= docs || !graph.valid(id) || graph.get_vector(id, raw) ||
          !raw.data())
        throw std::domain_error("unreadable codec training vector");
      const auto *v = static_cast<const float *>(raw.data());
      std::vector<float> out(v, v + dimension);
      double norm = 0;
      for (float x : out) {
        if (!std::isfinite(x))
          throw std::domain_error("nonfinite codec training vector");
        norm += double(x) * x;
      }
      if (norm > std::numeric_limits<float>::max() / 16.)
        throw std::domain_error("codec training norm overflows FP32");
      return out;
    };
    // Use a single 32-neighbor batch for training regardless of the output
    // graph's degree. This keeps the estimator and its sampling cost stable.
    std::unordered_map<uint32_t, std::vector<uint32_t>> neighbor_cache;
    std::vector<float> selection_cache;
    auto neighbors = [&](uint32_t id) -> const std::vector<uint32_t> & {
      auto it = neighbor_cache.find(id);
      if (it != neighbor_cache.end()) return it->second;
      const auto center = read_vector(id);
      std::vector<uint32_t> selected;
      if (SelectQuantizedGraphNeighbors(graph, id, center.data(), dimension, 32,
                                        selected, selection_cache))
        throw std::domain_error("cannot sample codec training neighbors");
      return neighbor_cache.emplace(id, std::move(selected)).first->second;
    };
    struct Row {
      std::vector<float> center;
      std::vector<uint32_t> neighbors, proxies;
    };
    // Sample live internal IDs, not external keys or a dense-prefix assumption.
    // The temporary ID list costs four bytes per live node, never a database
    // vector copy, and also handles snapshots containing large ID holes.
    std::vector<uint32_t> live;
    for (uint32_t id = 0; id < docs; ++id)
      if (graph.valid(id)) live.push_back(id);
    if (live.empty()) throw std::domain_error("no live codec training vectors");
    std::vector<Row> rows;
    const size_t wanted = std::min(live.size(), options.training_samples);
    rows.reserve(wanted);
    std::unordered_set<uint32_t> used;
    std::mt19937 random(options.seed + uint32_t(dimension));
    const size_t attempts = std::min(live.size() * 4, wanted * 64);
    for (size_t attempt = 0; attempt < attempts && rows.size() < wanted;
         ++attempt) {
      const uint32_t id = live[random() % live.size()];
      if (used.count(id)) continue;
      Row row;
      row.neighbors = neighbors(id);
      for (size_t trial = 0;
           trial < 1024 && row.proxies.size() < 8 && !row.neighbors.empty();
           ++trial) {
        const auto &child =
            neighbors(row.neighbors[random() % row.neighbors.size()]);
        if (child.empty()) continue;
        const uint32_t target = child[random() % child.size()];
        if (target != id &&
            std::find(row.neighbors.begin(), row.neighbors.end(), target) ==
                row.neighbors.end() &&
            std::find(row.proxies.begin(), row.proxies.end(), target) ==
                row.proxies.end())
          row.proxies.push_back(target);
      }
      // Small or disconnected graphs may not have eight distinct two-hop
      // proxies. Their direct neighbors still provide valid training pairs.
      if (row.proxies.empty()) row.proxies = row.neighbors;
      row.center = read_vector(id);
      if (!cosine) {
        std::vector<uint8_t> codes(dimension);
        float error, norm, offset, scale;
        if (!SQ8Center::encode(row.center.data(), dimension, codes.data(),
                               error, norm, offset, scale))
          throw std::domain_error("invalid codec training center");
        SQ8Center::decode(codes.data(), dimension, row.center.data(), offset,
                          scale);
      }
      used.insert(id);
      rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::domain_error("no live codec training vectors");

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dimension);
    for (const auto &row : rows)
      for (size_t d = 0; d < dimension; ++d) mean[d] += row.center[d];
    mean /= double(rows.size());
    Eigen::MatrixXd centers(rows.size(), dimension);
    Eigen::MatrixXd neighbor_means(rows.size(), dimension);
    double b = 0, c = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto &row = rows[i];
      Eigen::VectorXd center(dimension);
      for (size_t d = 0; d < dimension; ++d) center[d] = row.center[d];
      centers.row(i) = center.transpose();
      neighbor_means.row(i) = center.transpose();
      if (row.neighbors.empty()) continue;
      Eigen::MatrixXd r(dimension, row.neighbors.size());
      for (size_t j = 0; j < row.neighbors.size(); ++j) {
        const auto vector = read_vector(row.neighbors[j]);
        for (size_t d = 0; d < dimension; ++d) r(d, j) = vector[d] - center[d];
      }
      neighbor_means.row(i) += r.rowwise().mean().transpose();
      const Eigen::VectorXd u = mean - center;
      const double uu = u.squaredNorm();
      const Eigen::VectorXd ru = r.transpose() * u;
      for (uint32_t proxy : row.proxies) {
        const auto vector = read_vector(proxy);
        Eigen::VectorXd s(dimension);
        for (size_t d = 0; d < dimension; ++d) s[d] = vector[d];
        if (cosine && s.norm() > 0) s.normalize();
        s -= center;
        const double ss = s.squaredNorm(), su = s.dot(u);
        const Eigen::VectorXd rs = r.transpose() * s;
        const double weight = 1. / (row.neighbors.size() * row.proxies.size());
        for (size_t j = 0; j < row.neighbors.size(); ++j) {
          const double rr = r.col(j).squaredNorm();
          // Squared triangle area A(lambda) = A(0) + 2 B lambda + C lambda^2.
          b += weight * (ru[j] * (rs[j] - ss) - (rr - rs[j]) * su);
          c += weight *
               (uu * (rr + ss - 2 * rs[j]) - (ru[j] - su) * (ru[j] - su));
        }
      }
    }
    if (!std::isfinite(b) || !std::isfinite(c))
      throw std::domain_error("nonfinite codec training objective");
    Result result;
    result.samples = rows.size();
    result.lambda = float(options.center_weight *
                          (c > 0 ? std::clamp(-b / c, 0., 1.) : 0.));
    const auto words = train_words(options);
    const uint32_t header[]{0x51474243, 1, uint32_t(dimension),
                            uint32_t(options.block_dimension),
                            options.codec == Options::Codec::LowRank ? 3u : 1u};
    auto put = [&](const void *data, size_t bytes) {
      result.data.append(static_cast<const char *>(data), bytes);
    };
    put(header, sizeof(header));
    put(&result.lambda, sizeof(result.lambda));
    std::vector<float> mean32(dimension);
    for (size_t d = 0; d < dimension; ++d) mean32[d] = float(mean[d]);
    put(mean32.data(), dimension * sizeof(float));
    put(words.data(), words.size() * sizeof(float));

    if (options.codec == Options::Codec::LowRank) {
      turbo::SymQGRotation rotation(dimension);
      const size_t padded = rotation.padded_dim();
      Eigen::MatrixXd delta(rows.size(), padded);
      std::vector<float> residual(dimension), rotated;
      for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t d = 0; d < dimension; ++d)
          residual[d] = float(neighbor_means(i, d) -
                              ((1. - result.lambda) * centers(i, d) +
                               result.lambda * mean32[d]));
        rotation.rotate(residual.data(), rotated);
        for (size_t d = 0; d < padded; ++d) delta(i, d) = rotated[d];
      }
      // Uncentered second moment: U minimizes E||delta - U U^T delta||^2.
      const Eigen::MatrixXd covariance =
          delta.transpose() * delta / rows.size();
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);
      if (eig.info() != Eigen::Success || !eig.eigenvectors().allFinite())
        throw std::runtime_error("codec eigendecomposition failed");
      const uint32_t rank = options.rank;
      std::vector<float> basis(rank * padded);
      double energy = 0;
      for (size_t j = 0; j < rank; ++j) {
        const auto axis = eig.eigenvectors().col(padded - 1 - j);
        Eigen::Index pivot;
        axis.cwiseAbs().maxCoeff(&pivot);
        const double sign = axis[pivot] < 0 ? -1 : 1;
        for (size_t d = 0; d < padded; ++d)
          basis[j * padded + d] = float(sign * axis[d]);
        energy += std::max(0., eig.eigenvalues()[padded - 1 - j]);
      }
      result.captured_energy =
          covariance.trace() > 0 ? energy / covariance.trace() : 0;
      put(&rank, sizeof(rank));
      put(basis.data(), basis.size() * sizeof(float));
    }
    return result;
  }

 private:
  static std::vector<float> train_words(
      const QuantizedGraphBuildOptions &options) {
    const size_t dim = options.block_dimension;
    std::vector<float> words(16 * dim);
    if (options.codebook == QuantizedGraphBuildOptions::Codebook::Hadamard) {
      for (size_t j = 0; j < 16; ++j)
        for (size_t d = 0; d < dim; ++d)
          words[j * dim + d] =
              (__builtin_popcount(unsigned(d & (j & 7))) & 1 ? -1.f : 1.f) *
              (j < 8 ? 1.f : -1.f);
      return words;
    }
    // An orthogonal random rotation motivates a standard Gaussian reference
    // distribution. Learn eight antipodal pairs with three deterministic
    // Lloyd restarts and select on an independent synthetic validation set.
    constexpr size_t kSamples = 65536, kIterations = 48;
    std::mt19937_64 random(uint64_t(options.seed) * 100 + 4 + dim);
    std::normal_distribution<double> normal;
    Eigen::MatrixXd train(kSamples, dim), validation(kSamples, dim);
    for (auto *matrix : {&train, &validation})
      for (size_t i = 0; i < kSamples; ++i)
        for (size_t d = 0; d < dim; ++d) (*matrix)(i, d) = normal(random);
    double best = std::numeric_limits<double>::infinity();
    for (size_t restart = 0; restart < 3; ++restart) {
      Eigen::MatrixXd centers(8, dim);
      for (size_t j = 0; j < 8; ++j) {
        centers.row(j) = train.row(random() % kSamples);
        centers.row(j) *= std::sqrt(dim * .7) / centers.row(j).norm();
      }
      for (size_t iteration = 0; iteration < kIterations; ++iteration) {
        const Eigen::MatrixXd dot = train * centers.transpose();
        const Eigen::VectorXd norms = centers.rowwise().squaredNorm();
        Eigen::MatrixXd sums = Eigen::MatrixXd::Zero(8, dim);
        size_t counts[8]{};
        for (size_t i = 0; i < kSamples; ++i) {
          size_t label = 0;
          double score = -std::numeric_limits<double>::infinity();
          for (size_t j = 0; j < 8; ++j) {
            const double value = 2 * std::abs(dot(i, j)) - norms[j];
            if (value > score) {
              score = value;
              label = j;
            }
          }
          sums.row(label) += (dot(i, label) >= 0 ? 1. : -1.) * train.row(i);
          ++counts[label];
        }
        for (size_t j = 0; j < 8; ++j) {
          if (counts[j])
            centers.row(j) = sums.row(j) / counts[j];
          else
            centers.row(j) = train.row(random() % kSamples);
        }
      }
      const Eigen::MatrixXd dot = validation * centers.transpose();
      const Eigen::VectorXd norms = centers.rowwise().squaredNorm();
      double loss = 0;
      for (size_t i = 0; i < kSamples; ++i) {
        double score = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < 8; ++j)
          score = std::max(score, 2 * std::abs(dot(i, j)) - norms[j]);
        loss += validation.row(i).squaredNorm() - score;
      }
      if (loss < best) {
        best = loss;
        for (size_t j = 0; j < 8; ++j)
          for (size_t d = 0; d < dim; ++d) {
            words[j * dim + d] = float(centers(j, d));
            words[(j + 8) * dim + d] = -words[j * dim + d];
          }
      }
    }
    return words;
  }
};
}  // namespace zvec::core
