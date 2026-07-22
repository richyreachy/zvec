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

#include <cstring>
#include "diskann_distance_estimator.h"
#include "diskann_pq_table.h"
#include "diskann_pq_trainer.h"

namespace zvec {
namespace core {

//! PQ-based implementation of DiskAnnDistanceEstimator.
//! Wraps the existing DiskAnnPqTrainer + PQTable.
class PQDistanceEstimator : public DiskAnnDistanceEstimator {
 public:
  PQDistanceEstimator() = default;
  ~PQDistanceEstimator() override = default;

  int train_and_quantize(IndexThreads::Pointer threads,
                         IndexHolder::Pointer holder, const IndexMeta &meta,
                         std::vector<uint8_t> &serialized_quantizer,
                         std::vector<uint8_t> &quantized_data) override {
    // Determine chunk_num: default to dim/2
    uint32_t chunk_num = meta.dimension() / 2;
    if (chunk_num == 0) chunk_num = 1;
    if (chunk_num > meta.dimension()) chunk_num = meta.dimension();

    DiskAnnPqTrainer trainer(PQTable::kMaxTrainSampleCount);

    std::vector<uint8_t> full_pivot_data;
    std::vector<uint8_t> centroid;
    std::vector<uint32_t> chunk_offsets;
    std::vector<uint8_t> pq_codes;

    int ret =
        trainer.train_quantized_data(threads, holder, meta, full_pivot_data,
                                     centroid, chunk_offsets, chunk_num);
    if (ret != 0) return ret;

    ret = trainer.generate_quantized_data(threads, holder, meta, centroid,
                                          pq_codes, chunk_num);
    if (ret != 0) return ret;

    // Serialize quantizer params: [DiskAnnPqMeta][full_pivot_data][centroid]
    // [chunk_offsets]
    chunk_num_ = chunk_num;
    meta_ = meta;

    DiskAnnPqMeta pq_meta;
    pq_meta.full_pivot_data_size = full_pivot_data.size();
    pq_meta.centroid_data_size = centroid.size();
    pq_meta.chunk_num = chunk_num;

    serialized_quantizer.clear();
    serialized_quantizer.insert(
        serialized_quantizer.end(), reinterpret_cast<uint8_t *>(&pq_meta),
        reinterpret_cast<uint8_t *>(&pq_meta) + sizeof(DiskAnnPqMeta));
    serialized_quantizer.insert(serialized_quantizer.end(),
                                full_pivot_data.begin(), full_pivot_data.end());
    serialized_quantizer.insert(serialized_quantizer.end(), centroid.begin(),
                                centroid.end());
    serialized_quantizer.insert(
        serialized_quantizer.end(),
        reinterpret_cast<uint8_t *>(chunk_offsets.data()),
        reinterpret_cast<uint8_t *>(chunk_offsets.data() +
                                    chunk_offsets.size()));

    quantized_data = std::move(pq_codes);

    return 0;
  }

  int load(const IndexMeta &meta, const uint8_t *quantizer_params,
           size_t params_size, const uint8_t *quantized_data, size_t data_size,
           size_t doc_cnt) override {
    (void)data_size;
    meta_ = meta;

    if (params_size < sizeof(DiskAnnPqMeta)) {
      LOG_ERROR("PQDistanceEstimator: params too small: %zu", params_size);
      return IndexError_InvalidFormat;
    }

    DiskAnnPqMeta pq_meta;
    memcpy(&pq_meta, quantizer_params, sizeof(DiskAnnPqMeta));

    chunk_num_ = static_cast<uint32_t>(pq_meta.chunk_num);

    const uint8_t *ptr = quantizer_params + sizeof(DiskAnnPqMeta);

    std::vector<uint8_t> full_pivot_data(ptr,
                                         ptr + pq_meta.full_pivot_data_size);
    ptr += pq_meta.full_pivot_data_size;

    std::vector<uint8_t> centroid(ptr, ptr + pq_meta.centroid_data_size);
    ptr += pq_meta.centroid_data_size;

    std::vector<uint32_t> chunk_offsets(chunk_num_ + 1);
    memcpy(chunk_offsets.data(), ptr, (chunk_num_ + 1) * sizeof(uint32_t));

    std::vector<uint8_t> pq_data(quantized_data,
                                 quantized_data + doc_cnt * chunk_num_);

    pq_table_ = std::make_shared<PQTable>(meta, chunk_num_);
    pq_table_->init(full_pivot_data, centroid, chunk_offsets, pq_data);

    return 0;
  }

  int preprocess_query(const void *query, size_t query_size, void *dist_buffer,
                       size_t buffer_size) override {
    (void)query_size;
    (void)buffer_size;
    // PQTable modifies the query in-place (subtracts centroid), so we need
    // a writable copy.  The caller passes query_rotated which is writable.
    return pq_table_->preprocess_pq_dist_table(
        const_cast<void *>(query), static_cast<float *>(dist_buffer));
  }

  void compute_dists(uint32_t id_num, const diskann_id_t *ids,
                     const void *dist_buffer, void *coord_buffer,
                     float *dists) override {
    pq_table_->compute_dists(
        id_num, ids, chunk_num_,
        static_cast<float *>(const_cast<void *>(dist_buffer)), coord_buffer,
        dists);
  }

  size_t dist_buffer_size() const override {
    return PQTable::kPQCentroidNum * chunk_num_ * sizeof(float);
  }

  size_t coord_buffer_size(uint32_t max_ids) const override {
    return max_ids * chunk_num_ * sizeof(uint8_t);
  }

  size_t code_size() const override {
    return chunk_num_;
  }

 private:
  IndexMeta meta_;
  uint32_t chunk_num_{0};
  PQTable::Pointer pq_table_;
};

}  // namespace core
}  // namespace zvec
