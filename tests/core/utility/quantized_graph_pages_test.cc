// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "algorithm/quantized_graph/quantized_graph_pages.h"
#include <algorithm>
#include <new>
#include <vector>
#include <gtest/gtest.h>

namespace zvec::core::quantized_graph_pages {
namespace {
struct Fake {
  std::vector<bool> huge;
  std::vector<uintptr_t> calls;
  int error = 0;
  bool known = true, promote = true;
  int delay_seconds = 0;
  Clock::time_point time{};
  size_t coverage(uintptr_t start, size_t bytes) const {
    if (!bytes) return 0;
    if (!known) return kUnknown;
    size_t total = 0;
    for (size_t i = start / kPage; i < (start + bytes) / kPage; ++i)
      if (huge.at(i)) total += kPage;
    return total;
  }
  int collapse(uintptr_t start) {
    calls.push_back(start);
    time += std::chrono::seconds(delay_seconds);
    if (!error && promote) huge.at(start / kPage) = true;
    return error;
  }
  Clock::time_point now() const {
    return time;
  }
};
}  // namespace

TEST(QuantizedGraphPages, SkipsCoveredPagesAndPartialTail) {
  Fake f{{true, false, true, false}};
  auto r = prepare(0, 3 * kPage + 4096, f);
  EXPECT_EQ(r.before, 2 * kPage);
  EXPECT_EQ(r.after, 3 * kPage);
  EXPECT_EQ(r.eligible, 3 * kPage);
  EXPECT_TRUE(r.ready());
  EXPECT_EQ(f.calls, std::vector<uintptr_t>{kPage});
  EXPECT_FALSE(f.huge.back());
}

TEST(QuantizedGraphPages, AlreadyHugeNeedsNoCollapse) {
  Fake f{{true, true}};
  EXPECT_TRUE(prepare(0, 2 * kPage, f).ready());
  EXPECT_TRUE(f.calls.empty());
}

TEST(QuantizedGraphPages, AdviceOnlyPreservesLegacyMappings) {
  Fake f{{true, false}};
  auto r = prepare(0, 2 * kPage, f, false);
  EXPECT_EQ(r.after, kPage);
  EXPECT_FALSE(r.ready());
  EXPECT_TRUE(f.calls.empty());
}

TEST(QuantizedGraphPages, SuccessfulSyscallDoesNotProveCoverage) {
  Fake f{{false, false}};
  f.promote = false;
  auto r = prepare(0, 2 * kPage, f);
  EXPECT_EQ(r.failures, 0);
  EXPECT_EQ(r.attempts, 2);
  EXPECT_FALSE(r.ready());
}

TEST(QuantizedGraphPages, UnknownCoverageFailsClosed) {
  Fake f{{false}};
  f.known = false;
  auto r = prepare(0, kPage, f);
  EXPECT_EQ(r.after, kUnknown);
  EXPECT_FALSE(r.ready());
}

TEST(QuantizedGraphPages, UnsupportedOperationStopsImmediately) {
  for (int error : {EINVAL, ENOSYS, ENOTSUP, EPERM, EACCES}) {
    Fake f{{false, false}};
    f.error = error;
    auto r = prepare(0, 2 * kPage, f);
    EXPECT_EQ(r.attempts, 1);
    EXPECT_EQ(r.last_error, error);
    EXPECT_FALSE(r.ready());
  }
}

TEST(QuantizedGraphPages, AllocationFailureHasFiniteAttemptBudget) {
  Fake f{std::vector<bool>(100, false)};
  f.error = ENOMEM;
  auto r = prepare(0, 100 * kPage, f);
  EXPECT_EQ(r.attempts, 32);
  EXPECT_EQ(r.failures, 32);
  EXPECT_TRUE(r.budget_exhausted);
  EXPECT_FALSE(r.ready());
}

TEST(QuantizedGraphPages, SoftDeadlineStopsBetweenKernelCalls) {
  Fake f{std::vector<bool>(100, false)};
  f.delay_seconds = 3;
  auto r = prepare(0, 100 * kPage, f);
  EXPECT_EQ(r.attempts, 2);
  EXPECT_EQ(r.after, 2 * kPage);
  EXPECT_TRUE(r.budget_exhausted);
}

TEST(QuantizedGraphPages, SmallArenaNeedsNoHugePages) {
  Fake f{{false}};
  f.known = false;
  EXPECT_TRUE(prepare(0, kPage - 1, f).ready());
  EXPECT_TRUE(f.calls.empty());
}

TEST(QuantizedGraphPages, CallerCanDistinguishRetryableFailures) {
  Result r;
  r.eligible = kPage;
  r.after = 0;
  r.last_error = ENOMEM;
  EXPECT_EQ(int(IndexError_NoMemory), admission_error(r, true));
  EXPECT_EQ(0, admission_error(r, false));
  r.last_error = EAGAIN;
  EXPECT_EQ(int(IndexError_NoReady), admission_error(r, true));
  r.last_error = 0;  // Deadline expired with incomplete coverage.
  EXPECT_EQ(int(IndexError_NoReady), admission_error(r, true));
  r.after = kPage;
  EXPECT_EQ(0, admission_error(r, true));
}

TEST(QuantizedGraphPages, UnsupportedOrDeniedIsNotReportedAsNoMemory) {
  Result r;
  r.eligible = kPage;
  r.after = 0;
  for (int error : {EINVAL, ENOSYS, ENOTSUP}) {
    r.last_error = error;
    EXPECT_EQ(int(IndexError_Unsupported), admission_error(r, true));
  }
  for (int error : {EPERM, EACCES}) {
    r.last_error = error;
    EXPECT_EQ(int(IndexError_Denied), admission_error(r, true));
  }
  r.after = kUnknown;
  r.last_error = ENOMEM;
  EXPECT_EQ(int(IndexError_Unsupported), admission_error(r, true));
}

TEST(QuantizedGraphPages, LivePreparationPreservesEveryByte) {
  constexpr size_t bytes = 2 * kPage + 4096;
  auto *data = static_cast<unsigned char *>(
      ::operator new(bytes, std::align_val_t{kPage}));
  for (size_t i = 0; i < bytes; ++i) data[i] = i * 37 + 19;
#if defined(__linux__)
  ::madvise(data, bytes, MADV_HUGEPAGE);
#endif
  System ops;
  const auto r = prepare(reinterpret_cast<uintptr_t>(data), bytes, ops);
  EXPECT_EQ(r.eligible, 2 * kPage);
  EXPECT_LE(r.attempts, 2);
  EXPECT_TRUE(r.after == kUnknown || r.after <= r.eligible);
  // Availability depends on the host; preservation does not.
  bool identical = true;
  for (size_t i = 0; i < bytes; ++i)
    if (data[i] != static_cast<unsigned char>(i * 37 + 19)) identical = false;
  EXPECT_TRUE(identical);
  ::operator delete(data, std::align_val_t{kPage});
}

#if defined(__linux__)
TEST(QuantizedGraphPages, StrictModeRejectsUnavailablePagesBeforeQueries) {
  const char *old = std::getenv("QG_THP_POLICY");
  const bool was_set = old != nullptr;
  const std::string saved = old ? old : "";
  auto *data = static_cast<unsigned char *>(
      ::operator new(kPage, std::align_val_t{kPage}));
  ASSERT_EQ(0, ::madvise(data, kPage, MADV_NOHUGEPAGE));
  std::memset(data, 123, kPage);
  ::setenv("QG_THP_POLICY", "require", 1);
  EXPECT_NE(0, prepare_arena(data, kPage));
  ::setenv("QG_THP_POLICY", "prepare", 1);
  EXPECT_EQ(0, prepare_arena(data, kPage));
  ::setenv("QG_THP_POLICY", "hint", 1);
  EXPECT_EQ(0, prepare_arena(data, kPage));
  ::setenv("QG_THP_POLICY", "typo", 1);
  EXPECT_EQ(int(IndexError_InvalidArgument), prepare_arena(data, kPage));
  EXPECT_TRUE(std::all_of(data, data + kPage,
                          [](unsigned char c) { return c == 123; }));
  ::operator delete(data, std::align_val_t{kPage});
  if (was_set)
    ::setenv("QG_THP_POLICY", saved.c_str(), 1);
  else
    ::unsetenv("QG_THP_POLICY");
}
#endif
}  // namespace zvec::core::quantized_graph_pages
