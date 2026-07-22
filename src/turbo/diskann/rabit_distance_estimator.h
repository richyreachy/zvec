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
#include <memory>
#include <string>
#include <vector>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include "core/algorithm/diskann/diskann_distance_estimator.h"
#include "quantizer/quantizer.h"

namespace zvec {
namespace turbo {

//! RaBitQ-based implementation of DiskAnnDistanceEstimator.
//!
//! Uses the turbo RabitQuantizer for training, quantization, and distance
//! estimation.  This allows DiskAnn to use RaBitQ's random-bit quantization
//! instead of PQ for approximate distance computation during graph traversal.
//!
//! The serialized quantizer format is:
//!   [uint32_t magic=0x52544A51][uint32_t dp_len][uint32_t qp_len]
//!   [serialized turbo quantizer bytes]
//!
//! The quantized data is a flat array of dp_len-byte rows, one per vector.
class RabitDistanceEstimator : public core::DiskAnnDistanceEstimator {
 public:
  RabitDistanceEstimator() = default;
  ~RabitDistanceEstimator() override = default;

  int train_and_quantize(core::IndexThreads::Pointer threads,
                         core::IndexHolder::Pointer holder,
                         const core::IndexMeta &meta,
                         std::vector<uint8_t> &serialized_quantizer,
                         std::vector<uint8_t> &quantized_data) override {
    (void)threads;

    // Create the turbo RabitQuantizer via the global factory.
    auto quantizer = core::IndexFactory::CreateQuantizer("RabitQuantizer");
    if (!quantizer) {
      LOG_ERROR("RabitDistanceEstimator: Failed to create RabitQuantizer");
      return IndexError_NoExist;
    }

    // Configure with default parameters suitable for DiskAnn.
    ailego::Params params;
    params.set("proxima.rabitq.num_clusters", static_cast<uint32_t>(16));
    params.set("proxima.rabitq.total_bits", static_cast<uint32_t>(7));

    int ret = quantizer->init(meta, params);
    if (ret != 0) {
      LOG_ERROR("RabitDistanceEstimator: quantizer init failed, ret=%d", ret);
      return ret;
    }

    // Train the quantizer.
    ret = quantizer->train(holder);
    if (ret != 0) {
      LOG_ERROR("RabitDistanceEstimator: quantizer train failed, ret=%d", ret);
      return ret;
    }

    dp_len_ = quantizer->quantized_datapoint_vector_length();
    qp_len_ = quantizer->quantized_query_vector_length();
    dim_ = static_cast<uint32_t>(meta.dimension());
    meta_ = meta;

    // Serialize the quantizer.
    std::string ser;
    ret = quantizer->serialize(&ser);
    if (ret != 0) {
      LOG_ERROR("RabitDistanceEstimator: serialize failed, ret=%d", ret);
      return ret;
    }

    // Build the serialized quantizer blob.
    serialized_quantizer.clear();
    uint32_t magic = 0x52544A51u;  // 'QJTR'
    uint32_t dp_len_u32 = static_cast<uint32_t>(dp_len_);
    uint32_t qp_len_u32 = static_cast<uint32_t>(qp_len_);
    serialized_quantizer.insert(
        serialized_quantizer.end(), reinterpret_cast<uint8_t *>(&magic),
        reinterpret_cast<uint8_t *>(&magic) + sizeof(magic));
    serialized_quantizer.insert(
        serialized_quantizer.end(), reinterpret_cast<uint8_t *>(&dp_len_u32),
        reinterpret_cast<uint8_t *>(&dp_len_u32) + sizeof(dp_len_u32));
    serialized_quantizer.insert(
        serialized_quantizer.end(), reinterpret_cast<uint8_t *>(&qp_len_u32),
        reinterpret_cast<uint8_t *>(&qp_len_u32) + sizeof(qp_len_u32));
    serialized_quantizer.insert(serialized_quantizer.end(), ser.begin(),
                                ser.end());

    // Keep a reference for quantizing data.
    quantizer_ = quantizer;

    // Quantize all data points.
    size_t doc_cnt = holder->count();
    quantized_data.resize(doc_cnt * dp_len_);

    auto iter = holder->create_iterator();
    if (!iter) {
      LOG_ERROR("RabitDistanceEstimator: create_iterator failed");
      return IndexError_Runtime;
    }

    size_t idx = 0;
    while (iter->is_valid()) {
      quantizer->quantize_data(iter->data(),
                               quantized_data.data() + idx * dp_len_);
      iter->next();
      idx++;
    }

    if (idx != doc_cnt) {
      LOG_ERROR(
          "RabitDistanceEstimator: doc count mismatch, expected=%zu, "
          "actual=%zu",
          doc_cnt, idx);
      return IndexError_Runtime;
    }

    LOG_INFO(
        "RabitDistanceEstimator: trained %zu vectors, dp_len=%zu, qp_len=%zu",
        doc_cnt, dp_len_, qp_len_);

    return 0;
  }

  int load(const core::IndexMeta &meta, const uint8_t *quantizer_params,
           size_t params_size, const uint8_t *quantized_data, size_t data_size,
           size_t doc_cnt) override {
    (void)data_size;
    meta_ = meta;
    dim_ = static_cast<uint32_t>(meta.dimension());

    if (params_size < 12) {
      LOG_ERROR("RabitDistanceEstimator: params too small: %zu", params_size);
      return core::IndexError_InvalidFormat;
    }

    // Read header.
    uint32_t magic, dp_len_u32, qp_len_u32;
    memcpy(&magic, quantizer_params, sizeof(magic));
    memcpy(&dp_len_u32, quantizer_params + 4, sizeof(dp_len_u32));
    memcpy(&qp_len_u32, quantizer_params + 8, sizeof(qp_len_u32));

    if (magic != 0x52544A51u) {
      LOG_ERROR("RabitDistanceEstimator: bad magic: 0x%08x", magic);
      return core::IndexError_InvalidFormat;
    }

    dp_len_ = dp_len_u32;
    qp_len_ = qp_len_u32;

    // Deserialize the turbo quantizer.
    auto quantizer = core::IndexFactory::CreateQuantizer("RabitQuantizer");
    if (!quantizer) {
      LOG_ERROR("RabitDistanceEstimator: CreateQuantizer failed");
      return core::IndexError_NoExist;
    }

    std::string ser(reinterpret_cast<const char *>(quantizer_params) + 12,
                    params_size - 12);
    int ret = quantizer->deserialize(ser);
    if (ret != 0) {
      LOG_ERROR("RabitDistanceEstimator: deserialize failed, ret=%d", ret);
      return ret;
    }

    quantizer_ = quantizer;
    quantized_data_ = quantized_data;
    doc_cnt_ = doc_cnt;

    // Allocate query buffer.
    query_buf_.resize(qp_len_);

    LOG_INFO(
        "RabitDistanceEstimator: loaded %zu vectors, dp_len=%zu, qp_len=%zu",
        doc_cnt, dp_len_, qp_len_);

    return 0;
  }

  int preprocess_query(const void *query, size_t query_size, void *dist_buffer,
                       size_t buffer_size) override {
    (void)query_size;
    (void)buffer_size;
    // Quantize the query into our internal buffer, then copy it into
    // dist_buffer for the caller to pass to compute_dists.
    quantizer_->quantize_query(query, query_buf_.data());

    // Copy quantized query into dist_buffer (it's larger than needed).
    size_t copy_len = std::min(qp_len_, buffer_size);
    memcpy(dist_buffer, query_buf_.data(), copy_len);

    return 0;
  }

  void compute_dists(uint32_t id_num, const core::diskann_id_t *ids,
                     const void *dist_buffer, void *coord_buffer,
                     float *dists) override {
    (void)coord_buffer;
    // Build pointer array into quantized_data_ for the requested IDs.
    // Use the quantized query from dist_buffer.
    const void *qp = dist_buffer;

    for (uint32_t i = 0; i < id_num; ++i) {
      core::diskann_id_t id = ids[i];
      if (id < doc_cnt_) {
        const void *dp = quantized_data_ + id * dp_len_;
        dists[i] = quantizer_->calc_distance_dp_query(dp, qp);
      } else {
        dists[i] = std::numeric_limits<float>::max();
      }
    }
  }

  size_t dist_buffer_size() const override {
    // Must be at least qp_len_ (quantized query size).
    return qp_len_;
  }

  size_t coord_buffer_size(uint32_t max_ids) const override {
    // Not used for RaBitQ (we compute distances directly).
    (void)max_ids;
    return 1;
  }

  size_t code_size() const override {
    return dp_len_;
  }

 private:
  core::IndexMeta meta_;
  uint32_t dim_{0};
  size_t dp_len_{0};
  size_t qp_len_{0};
  size_t doc_cnt_{0};

  std::shared_ptr<Quantizer> quantizer_;
  const uint8_t *quantized_data_{nullptr};
  std::vector<uint8_t> query_buf_;
};

//! Static factory registration so DiskAnn core code can create
//! RabitDistanceEstimator instances via DiskAnnDistanceEstimator::create().
static bool s_rabit_registered = []() {
  core::DiskAnnDistanceEstimator::register_factory(
      "rabitq", []() -> core::DiskAnnDistanceEstimator::Pointer {
        return std::make_shared<RabitDistanceEstimator>();
      });
  return true;
}();

}  // namespace turbo
}  // namespace zvec
