#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace independent76 {
inline void require(bool value, const char *reason) {
  if (!value) throw std::runtime_error(reason);
}
struct Problem {
  size_t d, p;
  double lambda;
  std::vector<double> q, diagonal;
  Problem(size_t dim, size_t proxies, const double *input, double lam)
      : d(dim),
        p(proxies),
        lambda(lam),
        q(input, input + dim * proxies),
        diagonal(dim, lam) {
    require(d > 0 && std::isfinite(lambda) && lambda >= 0, "invalid problem");
    for (size_t j = 0; j < p; ++j)
      for (size_t k = 0; k < d; ++k) {
        require(std::isfinite(q[j * d + k]), "nonfinite query");
        diagonal[k] += q[j * d + k] * q[j * d + k];
      }
  }
};
struct Edge {
  std::vector<double> r, s, g, qs, as;
  double norm = 0, alpha0 = 0;
  explicit Edge(const Problem &a, const double *input,
                const double *initial = nullptr)
      : r(input, input + a.d), s(a.d), g(a.d), qs(a.p), as(a.d) {
    double u = 0;
    for (size_t k = 0; k < a.d; ++k) {
      require(std::isfinite(r[k]), "nonfinite residual");
      s[k] = initial ? initial[k] : (r[k] > 0 ? 1. : -1.);
      require(s[k] == 1 || s[k] == -1, "invalid sign");
      norm += r[k] * r[k];
      u += r[k] * s[k];
      g[k] = a.lambda * r[k];
    }
    require(std::isfinite(norm) && (norm == 0 || (u > 0 && std::isfinite(u))),
            "nonpositive denominator");
    alpha0 = norm == 0 ? 0 : norm / u;
    for (size_t j = 0; j < a.p; ++j) {
      double qr = 0;
      for (size_t k = 0; k < a.d; ++k) qr += a.q[j * a.d + k] * r[k];
      for (size_t k = 0; k < a.d; ++k) g[k] += a.q[j * a.d + k] * qr;
    }
  }
  double alpha() const {
    if (norm == 0) return 0;
    double u = 0;
    for (size_t k = 0; k < r.size(); ++k) u += r[k] * s[k];
    require(u > 0 && std::isfinite(u), "invalid reconstructed denominator");
    double v = norm / u;
    require(std::isfinite(v), "nonfinite alpha");
    return v;
  }
  double objective(const Problem &a) const {
    double scale = alpha(), euclidean = 0, projected = 0;
    std::vector<double> error(a.d);
    for (size_t k = 0; k < a.d; ++k) {
      error[k] = scale * s[k] - r[k];
      euclidean += error[k] * error[k];
    }
    for (size_t j = 0; j < a.p; ++j) {
      double v = 0;
      for (size_t k = 0; k < a.d; ++k) v += a.q[j * a.d + k] * error[k];
      projected += v * v;
    }
    double value = projected + a.lambda * euclidean;
    require(std::isfinite(value), "nonfinite objective");
    return value;
  }
  void costs(const Problem &a, double max_alpha, double *output) {
    std::fill_n(output, a.d, std::numeric_limits<double>::infinity());
    if (norm == 0) return;
    double u = 0, B = a.lambda * a.d, C = 0;
    for (size_t k = 0; k < a.d; ++k) {
      u += r[k] * s[k];
      C += s[k] * g[k];
      as[k] = a.lambda * s[k];
    }
    for (size_t j = 0; j < a.p; ++j) {
      double v = 0;
      for (size_t k = 0; k < a.d; ++k) v += a.q[j * a.d + k] * s[k];
      qs[j] = v;
      B += v * v;
      for (size_t k = 0; k < a.d; ++k) as[k] += a.q[j * a.d + k] * v;
    }
    require(u > 0, "invalid current denominator");
    double alpha = norm / u;
    for (size_t k = 0; k < a.d; ++k) {
      double next_u = u - 2 * s[k] * r[k];
      if (!(next_u > 0)) continue;
      double next_alpha = norm / next_u;
      if (!(next_alpha <= max_alpha)) continue;
      double next_B = B - 4 * s[k] * as[k] + 4 * a.diagonal[k];
      double next_C = C - 2 * s[k] * g[k];
      double delta = next_alpha * next_alpha * next_B -
                     2 * next_alpha * next_C -
                     (alpha * alpha * B - 2 * alpha * C);
      require(std::isfinite(delta), "nonfinite finite delta");
      output[k] = delta;
    }
  }
};
struct Result {
  std::vector<int8_t> signs;
  std::vector<double> alphas, objectives;
  std::vector<uint32_t> moves;
};
inline Result optimize(const Problem &a, const double *residuals,
                       size_t lanes) {
  require(lanes > 0, "empty block");
  std::vector<Edge> edges;
  edges.reserve(lanes);
  std::vector<double> values(lanes), costs(a.d);
  double total = 0;
  for (size_t i = 0; i < lanes; ++i) {
    edges.emplace_back(a, residuals + i * a.d);
    values[i] = edges.back().objective(a);
    total += values[i];
  }
  Result result;
  result.objectives.push_back(total);
  for (size_t sweep = 0; sweep < 2; ++sweep)
    for (size_t i = 0; i < lanes; ++i)
      for (size_t flip = 0; flip < 4; ++flip) {
        auto &e = edges[i];
        e.costs(a, e.alpha0 * 1.02, costs.data());
        size_t k = std::min_element(costs.begin(), costs.end()) - costs.begin();
        if (!std::isfinite(costs[k]) ||
            costs[k] >= -1e-13 * std::max(1., total))
          break;
        e.s[k] *= -1;
        double changed = e.objective(a), next_total = 0;
        for (size_t lane = 0; lane < lanes; ++lane)
          next_total += lane == i ? changed : values[lane];
        require(
            std::abs((next_total - total) - costs[k]) <
                1e-9 * std::max({1., std::abs(total), std::abs(next_total)}),
            "finite delta/direct mismatch");
        if (next_total >= total) {
          e.s[k] *= -1;
          break;
        }
        values[i] = changed;
        total = next_total;
        result.objectives.push_back(total);
        result.moves.insert(result.moves.end(),
                            {uint32_t(sweep), uint32_t(i), uint32_t(k)});
      }
  for (const auto &e : edges) {
    double alpha = e.alpha();
    require(alpha <= 1.02 * e.alpha0 + 1e-15, "scale cap");
    result.alphas.push_back(alpha);
    for (double s : e.s) result.signs.push_back(int8_t(s));
  }
  return result;
}
}  // namespace independent76
