// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_error.h>
#include "algorithm/hnsw/hnsw_params.h"
#if defined(__linux__)
#include <linux/fs.h>
#include <linux/mman.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace zvec::core::quantized_graph_pages {

constexpr size_t kPage = 2 * 1024 * 1024;
constexpr size_t kUnknown = std::numeric_limits<size_t>::max();
using Clock = std::chrono::steady_clock;

// Probe the owned arena, not process-wide AnonHugePages (which includes query
// buffers). Older headers/kernels and restricted procfs yield unknown coverage.
class System {
 public:
  System() = default;
  System(const System &) = delete;
  System &operator=(const System &) = delete;

  ~System() {
#if defined(__linux__)
    if (fd_ >= 0) ::close(fd_);
#endif
  }

  size_t coverage(uintptr_t start, size_t bytes) {
    if (!bytes) return 0;
#if defined(__linux__) && defined(PAGEMAP_SCAN)
    if (fd_ == -2) fd_ = ::open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) return kUnknown;

    size_t covered = 0;
    const uintptr_t end = start + bytes;
    while (start < end) {
      page_region regions[32]{};
      pm_scan_arg scan{};
      scan.size = sizeof(scan);
      scan.start = start;
      scan.end = end;
      scan.vec = reinterpret_cast<uintptr_t>(regions);
      scan.vec_len = 32;
      scan.category_mask = PAGE_IS_HUGE;
      scan.return_mask = PAGE_IS_HUGE;

      const int count = ::ioctl(fd_, PAGEMAP_SCAN, &scan);
      if (count < 0 || scan.walk_end <= start || scan.walk_end > end)
        return kUnknown;

      for (int i = 0; i < count; ++i)
        covered += regions[i].end - regions[i].start;
      start = scan.walk_end;
    }

    return covered;
#else
    (void)start;
    return kUnknown;
#endif
  }

  int collapse(uintptr_t start) {
#if defined(__linux__) && defined(MADV_COLLAPSE)
    return ::madvise(reinterpret_cast<void *>(start), kPage, MADV_COLLAPSE)
               ? errno
               : 0;
#else
    (void)start;
    return ENOTSUP;
#endif
  }

  Clock::time_point now() const {
    return Clock::now();
  }

 private:
#if defined(__linux__)
  int fd_ = -2;
#endif
};

struct Result {
  size_t eligible = 0, before = kUnknown, after = kUnknown;
  size_t attempts = 0, failures = 0;
  int last_error = 0;
  bool budget_exhausted = false;

  bool ready(double minimum = 1.0) const {
    return after != kUnknown &&
           (after == eligible ||
            (minimum < 1.0 && eligible &&
             static_cast<double>(after) / eligible >= minimum));
  }
};

inline bool parse_min_coverage(const char *value, double &minimum) {
  minimum = 1.0;
  if (!value) return true;

  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end || errno == ERANGE ||
      !(parsed > 0.0 && parsed <= 1.0))
    return false;

  minimum = parsed;
  return true;
}

inline bool read_coverage(const ailego::Params &params, const std::string &key,
                          double &minimum) {
  std::string value;
  if (params.get<std::string>(key, &value))
    return parse_min_coverage(value.c_str(), minimum);
  return params.get(key, &minimum) && minimum > 0.0 && minimum <= 1.0;
}

inline bool read_min_coverage(const ailego::Params &params, double &minimum) {
  const auto &key = PARAM_HNSW_QUANTIZED_GRAPH_THP_MIN_COVERAGE;
  return params.has(key)
             ? read_coverage(params, key, minimum)
             : parse_min_coverage(std::getenv("QG_THP_MIN_COVERAGE"), minimum);
}

inline bool read_coverage_floor(const ailego::Params &params, double minimum,
                                double &floor) {
  floor = minimum;  // No fallback unless explicitly configured for this index.
  const auto &key = PARAM_HNSW_QUANTIZED_GRAPH_THP_MIN_COVERAGE_FLOOR;
  return !params.has(key) ||
         (read_coverage(params, key, floor) && floor <= minimum);
}

// Select the highest admitted tier after one bounded preparation pass. Steps
// are five percentage points; the final step is clamped to the configured
// floor. Computing in percent avoids cumulative subtraction at 95/90/85/80.
inline double admission_minimum(const Result &r, double target, double floor) {
  if (r.after == kUnknown) return target;

  double minimum = target;
  for (unsigned step = 1; minimum > floor && !r.ready(minimum); ++step) {
    const double next = (target * 100.0 - 5.0 * step) / 100.0;
    minimum = next > floor ? next : floor;
  }

  return minimum;
}

// Keep transient resource failures distinguishable from unsupported policy or
// invalid input. Callers can release the failed index and retry with backoff.
inline int admission_error(const Result &r, bool required,
                           double minimum = 1.0) {
  if (!required || r.ready(minimum)) return 0;
  if (r.after == kUnknown) return IndexError_Unsupported;

  switch (r.last_error) {
    case ENOMEM:
      return IndexError_NoMemory;
    case EPERM:
    case EACCES:
      return IndexError_Denied;
    case EINVAL:
    case ENOSYS:
    case ENOTSUP:
      return IndexError_Unsupported;
    default:
      return IndexError_NoReady;
  }
}

// Called only after all arena writes/validation and before publication. One
// pass, at most 32 failed collapses, and a soft five-second budget: a blocking
// kernel call itself cannot be interrupted by this deadline. Never touch the
// partial trailing 2 MiB or discard/rewrite any vector data.
template <class Ops>
Result prepare(uintptr_t start, size_t bytes, Ops &ops, bool collapse = true) {
  Result r;
  r.eligible = bytes / kPage * kPage;
  const auto deadline = ops.now() + std::chrono::seconds(5);
  r.before = ops.coverage(start, r.eligible);

  if (collapse && r.before != r.eligible) {
    for (size_t offset = 0; offset < r.eligible; offset += kPage) {
      if (ops.now() >= deadline || r.failures >= 32) {
        r.budget_exhausted = true;
        break;
      }

      if (ops.coverage(start + offset, kPage) == kPage) continue;

      ++r.attempts;
      const int error = ops.collapse(start + offset);
      if (error) {
        ++r.failures;
        r.last_error = error;
        if (error == EINVAL || error == ENOSYS || error == ENOTSUP ||
            error == EPERM || error == EACCES)
          break;
      }
    }
  }

  // A successful hint/collapse is not a coverage measurement.
  r.after = ops.coverage(start, r.eligible);
  return r;
}

// proxima.hnsw.quantized_graph.thp_policy: prepare (default), hint (legacy
// advice only), require (reject coverage below the minimum or unverifiable
// coverage). thp_min_coverage is a fraction in (0, 1], default 1 (100%).
// Optional thp_min_coverage_floor enables five-percentage-point fallback, down
// to that floor (inclusive). It must be in (0, thp_min_coverage]. Omitting it
// preserves strict single-threshold admission. It has no environment fallback.
// Explicit Params override the corresponding QG_THP_POLICY/QG_THP_MIN_COVERAGE
// environment fallback. Configuration is per index and is not persisted.
// The minimum applies only to require and does not change preparation. The
// denominator excludes the partial trailing 2 MiB block. This is an admission
// check, not a lifetime reservation: benchmarks must verify mappings after
// runs. Callers may retry NoMemory/NoReady with a fresh index and bounded
// backoff; the library never exits the host process or retries loading.
inline int prepare_arena(void *arena, size_t bytes,
                         const ailego::Params &params = {}) {
  const char *policy = std::getenv("QG_THP_POLICY");
  std::string configured_policy;
  if (params.has(PARAM_HNSW_QUANTIZED_GRAPH_THP_POLICY)) {
    if (!params.get<std::string>(PARAM_HNSW_QUANTIZED_GRAPH_THP_POLICY,
                                 &configured_policy))
      return IndexError_InvalidArgument;
    policy = configured_policy.c_str();
  }

  if (!policy) policy = "prepare";
  const bool required = std::strcmp(policy, "require") == 0;
  const bool hint = std::strcmp(policy, "hint") == 0;
  if (!required && !hint && std::strcmp(policy, "prepare") != 0) {
    LOG_ERROR(
        "Invalid quantized graph THP policy: %s (prepare, hint or require)",
        policy);
    return IndexError_InvalidArgument;
  }

  double target = 1.0, floor = 1.0;
  if (required && (!read_min_coverage(params, target) ||
                   !read_coverage_floor(params, target, floor))) {
    LOG_ERROR(
        "Invalid quantized graph THP minimum coverage "
        "(expected 0 < floor <= minimum <= 1)");
    return IndexError_InvalidArgument;
  }

  const auto begin = Clock::now();
  System ops;
  const auto r = prepare(reinterpret_cast<uintptr_t>(arena), bytes, ops, !hint);

  const double minimum = admission_minimum(r, target, floor);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      Clock::now() - begin)
                      .count();
  const double percent =
      r.after == kUnknown ? -1.0
                          : (r.eligible ? 100.0 * r.after / r.eligible : 100.0);

  LOG_INFO(
      "Quantized graph THP policy=%s arena_bytes=%zu eligible_bytes=%zu "
      "before_bytes=%zu after_bytes=%zu coverage_known=%d ready=%d "
      "coverage_pct=%.6f min_coverage=%.17g full_coverage=%d "
      "target_coverage=%.17g coverage_floor=%.17g downgraded=%d "
      "attempts=%zu failures=%zu errno=%d budget_exhausted=%d cost_ms=%lld",
      policy, bytes, r.eligible, r.before == kUnknown ? 0 : r.before,
      r.after == kUnknown ? 0 : r.after, int(r.after != kUnknown),
      int(r.ready(minimum)), percent, minimum, int(r.ready()), target, floor,
      int(minimum < target), r.attempts, r.failures, r.last_error,
      int(r.budget_exhausted), static_cast<long long>(ms));

  const int error = admission_error(r, required, minimum);
  if (!r.ready())
    LOG_WARN(
        "Quantized graph THP coverage incomplete or unknown "
        "(coverage_pct=%.6f min_coverage=%.17g target_coverage=%.17g "
        "coverage_floor=%.17g downgraded=%d coverage_known=%d); %s",
        percent, minimum, target, floor, int(minimum < target),
        int(r.after != kUnknown),
        error ? "rejecting load" : "continuing with ordinary-page fallback");

  return error;
}

}  // namespace zvec::core::quantized_graph_pages
