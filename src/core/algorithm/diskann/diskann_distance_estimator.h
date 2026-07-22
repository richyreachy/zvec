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

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_threads.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

//! Abstract distance estimator for DiskAnn graph traversal.
//!
//! DiskAnn uses a compressed representation of all data points (stored on
//! disk) to compute approximate distances during beam search.  Historically
//! this was Product Quantization (PQ).  This interface abstracts the
//! distance-estimation layer so that alternative quantizers — notably the
//! turbo RaBitQ quantizer — can be plugged in without modifying the search
//! loop.
//!
//! Two roles:
//!  * Build time: train_and_quantize() trains the quantizer and produces
//!    a serialized quantizer blob + a flat array of per-vector codes.
//!  * Search time: load() restores the quantizer from the blob; then for
//!    each query, preprocess_query() prepares a query representation and
//!    compute_dists() evaluates distances for a batch of candidate IDs.
class DiskAnnDistanceEstimator {
 public:
  using Pointer = std::shared_ptr<DiskAnnDistanceEstimator>;
  using Factory = std::function<Pointer()>;

  virtual ~DiskAnnDistanceEstimator() = default;

  // -----------------------------------------------------------------------
  // Build-time API
  // -----------------------------------------------------------------------

  //! Train the quantizer and quantize all data points.
  //! On success, |serialized_quantizer| holds the quantizer parameters
  //! (codebook, rotation, etc.) and |quantized_data| holds a flat array of
  //! per-vector codes (doc_cnt * code_size() bytes, row-major).
  virtual int train_and_quantize(IndexThreads::Pointer threads,
                                 IndexHolder::Pointer holder,
                                 const IndexMeta &meta,
                                 std::vector<uint8_t> &serialized_quantizer,
                                 std::vector<uint8_t> &quantized_data) = 0;

  // -----------------------------------------------------------------------
  // Search-time API
  // -----------------------------------------------------------------------

  //! Restore the quantizer from a serialized blob and attach the flat
  //! array of per-vector codes.
  virtual int load(const IndexMeta &meta, const uint8_t *quantizer_params,
                   size_t params_size, const uint8_t *quantized_data,
                   size_t data_size, size_t doc_cnt) = 0;

  //! Prepare the query for distance computation.
  //! |query| points to the raw fp32/fp16 query vector (element_size bytes).
  //! |dist_buffer| is a workspace of dist_buffer_size() bytes.
  //! The contents of |dist_buffer| are consumed by compute_dists().
  virtual int preprocess_query(const void *query, size_t query_size,
                               void *dist_buffer, size_t buffer_size) = 0;

  //! Compute approximate distances for |id_num| data points identified by
  //! |ids|.  Results are written to |dists| (id_num floats).
  //! |dist_buffer| is the workspace populated by preprocess_query().
  //! |coord_buffer| is a workspace of coord_buffer_size(id_num) bytes.
  virtual void compute_dists(uint32_t id_num, const diskann_id_t *ids,
                             const void *dist_buffer, void *coord_buffer,
                             float *dists) = 0;

  // -----------------------------------------------------------------------
  // Size queries (used by DiskAnnContext for buffer allocation)
  // -----------------------------------------------------------------------

  //! Bytes of workspace needed for preprocess_query / compute_dists.
  virtual size_t dist_buffer_size() const = 0;

  //! Bytes of coordinate workspace needed for compute_dists with
  //! at most |max_ids| candidates.
  virtual size_t coord_buffer_size(uint32_t max_ids) const = 0;

  //! Bytes per quantized data point (for storage layout).
  virtual size_t code_size() const = 0;

  // -----------------------------------------------------------------------
  // Factory registry
  // -----------------------------------------------------------------------

  static void register_factory(const std::string &name, Factory factory);
  static Pointer create(const std::string &name);
  static bool has_factory(const std::string &name);

 private:
  static std::map<std::string, Factory> &registry();
};

}  // namespace core
}  // namespace zvec
