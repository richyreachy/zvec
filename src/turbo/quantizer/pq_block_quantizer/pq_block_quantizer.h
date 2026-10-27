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

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>
#include <turbo/distance/common/fast_scan_common.h>
#include <turbo/quantizer/pq_fast_quantizer/pq_fast_quantizer.h>
#include <zvec/core/framework/index_error.h>

namespace zvec::turbo {

// Global PQ4 codebook, eight coordinates per subquantizer, 32-lane FastScan.
// The model and codes are derived caches. Train before publishing the codec;
// all encoding and query operations thereafter are read-only on the model.
class PQBlockQuantizer {
 public:
  static constexpr size_t kBatchSize = turbo::kFastScanBlockSize;
  using Query = std::vector<char>;

  explicit PQBlockQuantizer(size_t dimension, bool cosine = false)
      : dimension_(dimension), chunks_((dimension + 7) / 8), cosine_(cosine) {}
  size_t encoded_dim() const {
    return chunks_ * 8;
  }
  size_t batch_bytes() const {
    return turbo::fast_scan_packed_block_size(chunks_);
  }
  bool trained() const {
    return quantizer_ != nullptr;
  }

  void transform(const void *vector, std::vector<float> &out) const {
    const auto *values = static_cast<const float *>(vector);
    out.assign(encoded_dim(), 0.0f);
    std::copy_n(values, dimension_, out.data());
  }

  // The immutable model defines the codebook shared by all encoded blocks.
  // Returns null until training succeeds; valid until reset/retraining.
  const PqFastQuantizer *model() const {
    return quantizer_.get();
  }

  // Training consumes already padded FP32 samples; the caller owns sampling
  // policy and supplies at least 16 vectors for the 16-centroid codebook.
  void reset_training() {
    quantizer_.reset();
  }
  int train(const core::IndexHolder::Pointer &holder, size_t threads) {
    reset_training();
    if (!holder || holder->data_type() != core::IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != encoded_dim() || holder->count() < 16) {
      return core::IndexError_Mismatch;
    }
    auto quantizer = std::make_shared<turbo::PqFastQuantizer>();
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DataType::DT_FP32, encoded_dim());
    // Cosine coordinates are already normalized by HNSW. Use -dot + 1,
    // not half the reconstructed L2: PQ centroids need not have unit norm.
    meta.set_metric(cosine_ ? "InnerProduct" : "SquaredEuclidean", 0,
                    ailego::Params());
    ailego::Params params;
    params.set("num_chunk", static_cast<uint32_t>(chunks_));
    int ret = quantizer->init(meta, params);
    if (ret != 0) return ret;
    ret = quantizer->train(holder, static_cast<int>(std::max(
                                       size_t{1}, std::min(threads, chunks_))));
    if (ret != 0) return ret;
    quantizer_ = std::move(quantizer);
    return 0;
  }

  void encode(const float *, const float *vectors, size_t count,
              char *codes) const {
    const size_t stride = quantizer_->quantized_datapoint_vector_length();
    std::vector<char> plain(std::max(size_t{1}, count) * stride);
    for (size_t i = 0; i < count; ++i)
      quantizer_->quantize_data(vectors + i * encoded_dim(),
                                plain.data() + i * stride);
    const int ret = quantizer_->pack_codes(plain.data(), count, stride, codes);
    assert(ret == 0);
    (void)ret;
  }
  void prepare_query(const void *vector, Query &query) const {
    std::vector<float> padded;
    transform(vector, padded);
    query.resize(quantizer_->quantized_query_vector_length());
    quantizer_->quantize_query(padded.data(), query.data());
  }
  void scan(const char *codes, Query &query, float, float *distances) const {
    quantizer_->calc_distance_packed_block(codes, kBatchSize, query.data(),
                                           distances);
    if (cosine_)
      for (size_t i = 0; i < kBatchSize; ++i) distances[i] += 1.0f;
  }

 private:
  size_t dimension_;
  size_t chunks_;
  bool cosine_;
  std::shared_ptr<const turbo::PqFastQuantizer> quantizer_;
};
}  // namespace zvec::turbo
