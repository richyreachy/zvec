// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include "algorithm/hnsw/hnsw_streamer.h"
#include "tests/test_util.h"

namespace zvec::core_interface {
namespace {
class FailingStreamer : public core::HnswStreamer {
 public:
  using core::HnswStreamer::open;
  explicit FailingStreamer(int error) : error_(error) {}
  int open(core::IndexStorage::Pointer, bool) override {
    return error_;
  }
  int close() override {
    return 0;
  }

 private:
  int error_;
};

class FailingIndex : public HNSWIndex {
 public:
  using HNSWIndex::init;
  explicit FailingIndex(int error) : error_(error) {}

 protected:
  int prepare_streamer_open(const StorageOptions &) override {
    streamer_ = error_ ? std::make_shared<FailingStreamer>(error_) : nullptr;
    return 0;
  }

 private:
  int error_;
};
}  // namespace

TEST(IndexOpenError, PreservesResourceAndPermanentFailures) {
  const std::string path = "index_open_error.index";
  for (int error :
       {int(core::IndexError_NoMemory), int(core::IndexError_NoReady),
        int(core::IndexError_Unsupported), int(core::IndexError_Denied),
        int(core::IndexError_InvalidArgument),
        int(core::IndexError_InvalidFormat), 0}) {
    test_util::RemoveTestFiles(path + "*");
    {
      FailingIndex index(error);
      HNSWIndexParam param(MetricType::kL2sq, 4, 16, 100);
      param.data_type = DataType::DT_FP32;
      ASSERT_EQ(0, index.init(param));
      EXPECT_EQ(error ? error : int(core::IndexError_Runtime),
                index.open(path, {StorageOptions::StorageType::kMMAP, true}));
      // Destruction after failed open releases storage before an outer retry.
    }
    test_util::RemoveTestFiles(path + "*");
  }
}
}  // namespace zvec::core_interface
