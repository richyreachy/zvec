// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <turbo/distance/common/symqg_kernels.h>
#include "symqg_integer_projection.h"
#include "symqg_projection_assignment.h"
namespace zvec::turbo::research {
class SymQGProjection {
 public:
  struct Encoded {
    std::vector<uint8_t> labels;
    std::vector<double> beta, residual, anchor;
  };
  bool enabled() const {
    return projected_ != 0;
  }
  size_t dimension() const {
    return projected_;
  }
  void load(std::istream &in, size_t dimension) {
    uint32_t h[5]{};
    double f[2]{};
    in.read(reinterpret_cast<char *>(h), sizeof(h));
    in.read(reinterpret_cast<char *>(f), sizeof(f));
    if (!in || h[0] != 0x50524a31 || h[1] != dimension || h[2] == 0 ||
        h[2] > dimension || h[2] % 64 || h[3] > 64 || h[4] > 8 ||
        !std::isfinite(f[0]) || f[0] < 0 || !std::isfinite(f[1]) || f[1] < 1)
      throw std::runtime_error("invalid projection model");
    const uint64_t payload = uint64_t(h[1]) * h[2] * sizeof(float) +
                             uint64_t(h[1]) * h[3] * sizeof(double);
    if (payload > (4u << 20))
      throw std::runtime_error("projection model too large");
    full_ = dimension;
    projected_ = h[2];
    rank_ = h[3];
    sweeps_ = h[4];
    sigma_ = f[0];
    guard_ = f[1];
    matrix_.resize(full_ * projected_);
    basis_.resize(full_ * rank_);
    in.read(reinterpret_cast<char *>(matrix_.data()), matrix_.size() * 4);
    in.read(reinterpret_cast<char *>(basis_.data()), basis_.size() * 8);
    if (!in) throw std::runtime_error("truncated projection model");
    for (float x : matrix_)
      if (!std::isfinite(x))
        throw std::runtime_error("invalid projection matrix");
    for (double x : basis_)
      if (!std::isfinite(x))
        throw std::runtime_error("invalid projection metric");
    integer_query_ = std::getenv("QG_RESEARCH_PROJECTION_INT8") != nullptr;
    if (integer_query_) integer_.prepare(matrix_, projected_, full_);
    projected_basis_.assign(projected_ * rank_, 0);
    for (size_t p = 0; p < projected_; ++p)
      for (size_t d = 0; d < full_; ++d)
        for (size_t k = 0; k < rank_; ++k)
          projected_basis_[p * rank_ + k] +=
              double(matrix_[p * full_ + d]) * basis_[d * rank_ + k];
  }
  void project(const float *vector, std::vector<float> &out) const {
    out.resize(projected_);
    if (symqg_rank_project(matrix_.data(), vector, projected_, full_,
                           out.data()))
      return;
    for (size_t p = 0; p < projected_; ++p) {
      double v = 0;
      for (size_t d = 0; d < full_; ++d)
        v += double(matrix_[p * full_ + d]) * vector[d];
      out[p] = float(v);
    }
  }
  void project_query(const float *vector, std::vector<float> &out,
                     std::vector<uint8_t> &scratch) const {
    if (integer_query_ && integer_.project(vector, out, scratch)) return;
    project(vector, out);
  }
  Encoded encode(const float *vectors, size_t count,
                 const std::vector<float> &anchor,
                 const std::vector<float> &words, size_t block) const {
    if (count > 32 || projected_ % block)
      throw std::runtime_error("unsupported projected batch");
    Encoded out;
    out.labels.resize(count * projected_ / block);
    out.beta.resize(count);
    out.residual.resize(count * projected_);
    std::vector<float> pa, pv;
    project(anchor.data(), pa);
    out.anchor.assign(pa.begin(), pa.end());
    std::vector<double> norms(count), z(count * rank_),
        errors(count * projected_), stats(count * 4),
        w(words.begin(), words.end());
    for (size_t i = 0; i < count; ++i) {
      project(vectors + i * full_, pv);
      for (size_t p = 0; p < projected_; ++p)
        out.residual[i * projected_ + p] = double(pv[p]) - pa[p];
      for (size_t d = 0; d < full_; ++d) {
        const double r = double(vectors[i * full_ + d]) - anchor[d];
        norms[i] += r * r;
        for (size_t k = 0; k < rank_; ++k)
          z[i * rank_ + k] += r * basis_[d * rank_ + k];
      }
    }
    int rc = optimize_projected_radial(
        count, projected_, block, rank_, w.data(), projected_basis_.data(),
        sigma_, guard_, sweeps_, out.residual.data(), norms.data(), z.data(),
        out.labels.data(), out.beta.data(), errors.data(), stats.data());
    if (rc) throw std::runtime_error("projected radial assignment failed");
    return out;
  }

 private:
  bool integer_query_ = false;
  SymQGIntegerProjection integer_;
  size_t full_ = 0, projected_ = 0, rank_ = 0, sweeps_ = 0;
  double sigma_ = 0, guard_ = 1;
  std::vector<float> matrix_;
  std::vector<double> basis_, projected_basis_;
};
inline void ExtendProjectionModelForResearch(std::string &data) {
  const char *path = std::getenv("QG_RESEARCH_PROJECTION");
  if (!path) return;
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  auto n = f.tellg();
  if (!f || n < 36 || n > (4 << 20) || data.size() < 24)
    throw std::runtime_error("invalid projection extension");
  std::string bytes(size_t(n), '\0');
  f.seekg(0);
  f.read(bytes.data(), bytes.size());
  if (!f) throw std::runtime_error("truncated projection extension");
  if (data.size() + bytes.size() > (4u << 20))
    throw std::runtime_error("projection codec too large");
  uint32_t v = 4;
  std::memcpy(data.data() + 4, &v, 4);
  data += bytes;
}
}  // namespace zvec::turbo::research
