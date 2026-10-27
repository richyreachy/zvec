#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#if defined(QG_RADIAL_STATS)
#include <atomic>
#include <cstdio>
#include <cstdlib>
#endif

namespace zvec::turbo {
struct RadialCalibration {
  enum Kind : size_t { kRegular, kZeroResidual, kZeroCenter, kRadialFallback };
  struct Factors {
    double a, b;
    Kind kind;
  };
  // h' = a*h+b*c, constrained by r.h'=r.r and c.h'=c.r.
  // Near singular systems use a bounded radial reconstruction, which need
  // not satisfy both constraints unless r and c are exactly collinear.
  static Factors fit(double rr, double cc, double cr, double ch, double rh) {
    if (!(rr > 0)) return {0., 0., kZeroResidual};
    if (!(cc > 1e-24)) return {rh > 1e-24 ? rr / rh : 0., 0., kZeroCenter};
    const double determinant = rh * cc - ch * cr;
    const double scale = std::max(
        {std::abs(rh * cc), std::abs(ch * cr), std::abs(rr * cc), 1e-30});
    if (std::abs(determinant) > 1e-8 * scale) {
      const double a = (rr * cc - cr * cr) / determinant;
      const double b = (rh * cr - ch * rr) / determinant;
      if (std::isfinite(a) && std::isfinite(b) && std::abs(a) <= 4. &&
          std::abs(b) <= 4.)
        return {a, b, kRegular};
    }
    return {0., cr / cc, kRadialFallback};
  }
#if defined(QG_RADIAL_STATS)
  inline static std::array<std::atomic<uint64_t>, 4> counts{};
  struct Reporter {
    ~Reporter() {
      const char *path = std::getenv("QG_RADIAL_STATS_FILE");
      if (!path) return;
      FILE *file = std::fopen(path, "w");
      if (!file) return;
      std::fprintf(file,
                   "{\"regular\":%llu,\"zero_residual\":%llu,"
                   "\"zero_center\":%llu,\"radial_fallback\":%llu}\n",
                   (unsigned long long)counts[0].load(),
                   (unsigned long long)counts[1].load(),
                   (unsigned long long)counts[2].load(),
                   (unsigned long long)counts[3].load());
      std::fclose(file);
    }
  };
  inline static Reporter reporter;
  static void record(const std::array<uint64_t, 4> &batch) {
    (void)reporter;
    for (size_t j = 0; j < batch.size(); ++j)
      if (batch[j]) counts[j].fetch_add(batch[j], std::memory_order_relaxed);
  }
#endif
};
}  // namespace zvec::turbo
