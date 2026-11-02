// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once
// Research prototype: covariance-weighted radial code assignment.
// Full-space radial calibration in a projected code space; FP64 assignment.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// C = sigma*I + B*B^T. B is row-major d by k. Initial encoding matches
// the fixed-alpha block encoder; block=4 hypercube words give sign RaBitQ.
// stats per row: initial/final Euclidean squared error, initial/final C loss.
namespace zvec::turbo::research {
inline int optimize_projected_radial(
    int n, int d, int block, int k, const double *words, const double *basis,
    double sigma, double guard, int sweeps, const double *residuals,
    const double *full_norm, const double *full_metric, uint8_t *labels,
    double *beta_out, double *errors, double *stats) {
  if (n < 0 || d <= 0 || block <= 0 || d % block || k < 0 || sigma < 0 ||
      guard < 1 || sweeps < 0)
    return -1;
  const int groups = d / block;
  std::vector<double> norms(16), projected(groups * 16 * k);
  for (int j = 0; j < 16; ++j)
    for (int h = 0; h < block; ++h)
      norms[j] += words[j * block + h] * words[j * block + h];
  for (int g = 0; g < groups; ++g)
    for (int j = 0; j < 16; ++j)
      for (int l = 0; l < k; ++l)
        for (int h = 0; h < block; ++h)
          projected[(g * 16 + j) * k + l] +=
              words[j * block + h] * basis[(g * block + h) * k + l];
  std::vector<double> dots(groups * 16), z(k), s(k), trial(k);
  for (int row = 0; row < n; ++row) {
    const double *r = residuals + row * d;
    double rr = full_norm[row], a = 0, bn = 0;
    std::fill(z.begin(), z.end(), 0);
    std::fill(s.begin(), s.end(), 0);
    for (int l = 0; l < k; ++l) z[l] = full_metric[row * k + l];
    const double alpha = std::sqrt(rr / d);
    for (int g = 0; g < groups; ++g) {
      double best = -std::numeric_limits<double>::infinity();
      int chosen = 0;
      for (int j = 0; j < 16; ++j) {
        double dot = 0;
        for (int h = 0; h < block; ++h)
          dot += r[g * block + h] * words[j * block + h];
        dots[g * 16 + j] = dot;
        const double score = 2 * alpha * dot - alpha * alpha * norms[j];
        if (score > best) {
          best = score;
          chosen = j;
        }
      }
      labels[row * groups + g] = chosen;
      a += dots[g * 16 + chosen];
      bn += norms[chosen];
      for (int l = 0; l < k; ++l) s[l] += projected[(g * 16 + chosen) * k + l];
    }
    if (rr > 0 && !(a > 0)) return -2;
    double beta = rr == 0 ? 0 : rr / a;
    auto loss = [&](double b, double norm, const std::vector<double> &v) {
      double value = sigma * std::max(0., b * b * norm - rr);
      for (int l = 0; l < k; ++l) {
        const double e = b * v[l] - z[l];
        value += e * e;
      }
      return value;
    };
    const double initial_e = std::max(0., beta * beta * bn - rr);
    double objective = loss(beta, bn, s);
    stats[row * 4] = initial_e;
    stats[row * 4 + 2] = objective;
    for (int sweep = 0; sweep < sweeps && rr > 0; ++sweep) {
      int changed = 0;
      for (int g = 0; g < groups; ++g) {
        const int old = labels[row * groups + g];
        int chosen = old;
        double best = objective, best_a = a, best_bn = bn;
        for (int j = 0; j < 16; ++j) {
          if (j == old) continue;
          const double aa = a - dots[g * 16 + old] + dots[g * 16 + j];
          if (!(aa > 1e-14 * std::sqrt(rr))) continue;
          const double norm = bn - norms[old] + norms[j], b = rr / aa;
          const double en = std::max(0., b * b * norm - rr);
          if (en > guard * initial_e + 1e-12 * std::max(rr, 1e-12)) continue;
          for (int l = 0; l < k; ++l)
            trial[l] = s[l] - projected[(g * 16 + old) * k + l] +
                       projected[(g * 16 + j) * k + l];
          const double value = loss(b, norm, trial);
          if (value < best - 1e-14 * std::max(objective, 1e-12)) {
            best = value;
            chosen = j;
            best_a = aa;
            best_bn = norm;
          }
        }
        if (chosen != old) {
          for (int l = 0; l < k; ++l)
            s[l] += projected[(g * 16 + chosen) * k + l] -
                    projected[(g * 16 + old) * k + l];
          a = best_a;
          bn = best_bn;
          beta = rr / a;
          objective = best;
          labels[row * groups + g] = chosen;
          ++changed;
        }
      }
      if (!changed) break;
    }
    beta_out[row] = beta;
    stats[row * 4 + 1] = std::max(0., beta * beta * bn - rr);
    stats[row * 4 + 3] = loss(beta, bn, s);
    for (int g = 0; g < groups; ++g)
      for (int h = 0; h < block; ++h)
        errors[row * d + g * block + h] =
            beta * words[labels[row * groups + g] * block + h] -
            r[g * block + h];
  }
  return 0;
}


}  // namespace zvec::turbo::research
