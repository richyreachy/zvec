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

#pragma once

#include <memory>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_stats.h>
#include "quantizer/quantizer.h"

namespace zvec {
namespace turbo {

using namespace zvec::core;

//! RaBitQ Quantizer
//!
//! Implements the RaBitQ (Random Bit Quantization) algorithm as a turbo
//! Quantizer.  RaBitQ uses random rotation + 1-bit (plus optional extra-bit)
//! quantization to estimate distances with sub-linear error, making it
//! suitable for approximate nearest neighbor search in compressed space.
//!
//! The quantizer must be trained (require_train() == true) before use.
//! Training performs K-means clustering on the input data to obtain
//! centroids, then creates a random rotation matrix (FHT-Kac or dense).
//!
//! Quantized datapoint layout (quantized_datapoint_vector_length() bytes):
//!   [4B cluster_id][bin_data][ex_data]
//!
//! Quantized query layout (quantized_query_vector_length() bytes):
//!   [rotated_query][query_bin][delta|vl|k1xsumq|kbxsumq][q_to_centroids]
//!
//! All rabitqlib types are hidden behind a pimpl (Impl) so that consumers
//! of this header do not transitively include rabitqlib/Eigen headers.
class RabitQuantizer : public Quantizer {
 public:
  RabitQuantizer();
  ~RabitQuantizer() override;

  // Non-copyable
  RabitQuantizer(const RabitQuantizer &) = delete;
  RabitQuantizer &operator=(const RabitQuantizer &) = delete;

  //! Initialize with index metadata and parameters.
  //! Recognized params:
  //!   proxima.rabitq.num_clusters  (default 16)
  //!   proxima.rabitq.total_bits    (default 7, range [1, 9])
  //!   proxima.rabitq.rotator.type  ("fht" or "matrix", default "fht")
  //!   proxima.rabitq.sample_count (0 = all)
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  const IndexMeta &meta() const override;

  DataType input_data_type() const override;

  QuantizeType type() const override;

  int dim() const override;

  bool require_train() const override;

  //! Train with K-means clustering on the provided data.
  int train(const void *data, size_t num, size_t stride) override;

  //! Train with K-means clustering using an IndexHolder.
  int train(IndexHolder::Pointer holder) override;

  size_t quantized_datapoint_vector_length() const override;

  size_t quantized_query_vector_length() const override;

  void quantize_data(const void *input, void *output) const override;

  void quantize_query(const void *input, void *output) const override;

  float calc_distance_dp_query(const void *dp,
                               const void *query) const override;

  void calc_distance_dp_query_batch(const void *const *dp_list, int dp_num,
                                    const void *query,
                                    float *dist_list) const override;

  float calc_distance_dp_query_unquantized(const void *dp,
                                           const void *query) const override;

  void calc_distance_dp_query_batch_unquantized(
      const void *const *dp_list, int dp_num, const void *query,
      float *dist_list) const override;

  float calc_distance_dp_dp(const void *dp1, const void *dp2) const override;

  int serialize(std::string *out) const override;

  int deserialize(std::string &in) override;

  int deserialize(const void *data, size_t len) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace turbo
}  // namespace zvec
