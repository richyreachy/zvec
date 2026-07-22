// Copyright 2025-present the zvec project
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

#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_builder.h"
#include "diskann_distance_estimator.h"
#include "diskann_holder.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

constexpr size_t kDim = 64;

class DiskAnnRabitqTest : public testing::Test {
 protected:
  void SetUp() override {
    LoggerBroker::SetLevel(Logger::LEVEL_INFO);
    index_meta_ptr_.reset(new IndexMeta(IndexMeta::DataType::DT_FP32, kDim));
    index_meta_ptr_->set_metric("SquaredEuclidean", 0, Params());
  }

  void TearDown() override {
    string cmd = "rm -rf " + dir_;
    system(cmd.c_str());
  }

  static string dir_;
  static shared_ptr<IndexMeta> index_meta_ptr_;
};

string DiskAnnRabitqTest::dir_("DiskAnnRabitqTest");
shared_ptr<IndexMeta> DiskAnnRabitqTest::index_meta_ptr_;

// Verify that the RaBitQ distance estimator factory is registered.
TEST_F(DiskAnnRabitqTest, FactoryRegistration) {
  EXPECT_TRUE(DiskAnnDistanceEstimator::has_factory("rabitq"))
      << "RaBitQ distance estimator factory must be registered";
  auto estimator = DiskAnnDistanceEstimator::create("rabitq");
  ASSERT_NE(estimator, nullptr);
}

// Build a small DiskAnn index using the RaBitQ quantizer and verify it
// completes successfully.
TEST_F(DiskAnnRabitqTest, BuildWithRabitq) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(kDim);
  constexpr size_t kDocCnt = 100;
  for (size_t i = 0; i < kDocCnt; ++i) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;
  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 50);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);
  // Use RaBitQ quantizer instead of default PQ.
  params.set("zvec.diskann.builder.quantizer", string("rabitq"));

  ASSERT_EQ(0, builder->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  // Dump the index to verify it can be serialized.
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = dir_ + "/BuildWithRabitq";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Verify stats.
  auto &stats = builder->stats();
  ASSERT_EQ(kDocCnt, stats.trained_count());
  ASSERT_EQ(kDocCnt, stats.built_count());
  ASSERT_EQ(kDocCnt, stats.dumped_count());
}
