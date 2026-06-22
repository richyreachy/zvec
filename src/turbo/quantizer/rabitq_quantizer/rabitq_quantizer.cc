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

#include "quantizer/rabitq_quantizer/rabitq_quantizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace turbo {

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------
float RabitqQuantizer::ip_float_code(const float *q, const uint8_t *code,
                                     size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum += q[i] * static_cast<float>(code[i]);
  }
  return sum;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

int RabitqQuantizer::init(const IndexMeta &meta, const ailego::Params &params) {
#if !RABITQ_SUPPORTED
  LOG_ERROR("RaBitQ quantizer is not supported on this platform");
  return IndexError_Unsupported;
#else
  original_meta_ = meta;

  original_dim_ = meta.dimension();
  padded_dim_ = ((original_dim_ + 63) / 64) * 64;


  // Read total_bits from params (default = 2)
  uint32_t bits = kDefaultTotalBits;
  params.get(RABITQ_TOTAL_BITS_KEY, &bits);
  if (bits < 1 || bits > 8) {
    LOG_ERROR("RaBitQ total_bits must be in [1, 8], got %u", bits);
    return IndexError_InvalidArgument;
  }
  total_bits_ = bits;

  // Determine metric type
  auto metric_name = meta.metric_name();
  if (metric_name == "SquaredEuclidean" || metric_name == "Euclidean") {
    metric_type_ = MetricType::kSquaredEuclidean;
  } else if (metric_name == "InnerProduct") {
    metric_type_ = MetricType::kInnerProduct;
  } else if (metric_name == "Cosine") {
    // Cosine is handled as normalized inner product
    metric_type_ = MetricType::kInnerProduct;
  } else {
    LOG_WARN("RaBitQ: unsupported metric '%s', defaulting to L2",
             metric_name.c_str());
    metric_type_ = MetricType::kSquaredEuclidean;
  }

  // Create FHT rotator with padded dimension
  rotator_ = CreateRotator(RotatorType::kFht, static_cast<int>(original_dim_));
  if (!rotator_) {
    LOG_ERROR("RaBitQ: failed to create FHT rotator for dim=%u", original_dim_);
    return IndexError_Runtime;
  }

  // Precompute RaBitQ config for faster quantization (multi-bit)
  if (total_bits_ > 1) {
    rabitq_config_ = rabitqlib::quant::faster_config(padded_dim_, total_bits_);
  }

  meta_.set_meta(IndexMeta::DataType::DT_RABITQ, 1, padded_dim_);

  LOG_DEBUG(
      "RaBitQ quantizer initialized: dim=%u, padded_dim=%u, total_bits=%u",
      original_dim_, padded_dim_, total_bits_);
  return 0;
#endif
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

void RabitqQuantizer::train(const void *data, size_t num, size_t stride) {
#if RABITQ_SUPPORTED
  if (!rotator_) return;
  // FHT rotator training just initializes the random sign vector (no data
  // needed)
  rotator_->train(data, num, stride);
  LOG_DEBUG("RaBitQ rotator trained with %zu vectors", num);
#else
  (void)data;
  (void)num;
  (void)stride;
#endif
}

int RabitqQuantizer::train(IndexHolder::Pointer holder) {
#if RABITQ_SUPPORTED
  if (!holder || holder->dimension() != original_dim_) {
    return IndexError_Mismatch;
  }
  if (!rotator_) return IndexError_Runtime;
  // Train rotator (FHT does not use data, but the interface is kept generic)
  rotator_->train(nullptr, 0, 0);
  return 0;
#else
  (void)holder;
  return IndexError_Unsupported;
#endif
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

void RabitqQuantizer::quantize_data(const void *input, void *output) const {
#if RABITQ_SUPPORTED
  const float *vec = reinterpret_cast<const float *>(input);
  auto *out = reinterpret_cast<char *>(output);

  // Step 1: Apply FHT rotation (pads to padded_dim and rotates)
  thread_local std::vector<float> rotated;
  rotated.resize(padded_dim_, 0.0f);
  rotator_->apply(vec, rotated.data());

  // Step 2: Quantize using RaBitQ full_single (centroid = 0)
  thread_local std::vector<uint8_t> code;
  code.resize(padded_dim_, 0);

  float f_add = 0.0f, f_rescale = 0.0f, f_error = 0.0f;

  rabitqlib::MetricType rq_metric =
      (metric_type_ == MetricType::kSquaredEuclidean) ? rabitqlib::METRIC_L2
                                                      : rabitqlib::METRIC_IP;

  rabitqlib::quant::quantize_full_single<float, uint8_t>(
      rotated.data(), padded_dim_, total_bits_, code.data(), f_add, f_rescale,
      f_error, rq_metric, rabitq_config_);

  // Step 3: Pack into output buffer
  // Layout: [codes (padded_dim B)] [f_add (4B)] [f_rescale (4B)] [f_error (4B)]
  std::memcpy(out, code.data(), padded_dim_);
  float *factors = reinterpret_cast<float *>(out + padded_dim_);
  factors[0] = f_add;
  factors[1] = f_rescale;
  factors[2] = f_error;
#else
  (void)input;
  (void)output;
#endif
}

void RabitqQuantizer::quantize_query(const void *input, void *output) const {
#if RABITQ_SUPPORTED
  const float *vec = reinterpret_cast<const float *>(input);
  auto *out = reinterpret_cast<float *>(output);

  // Step 1: Apply FHT rotation (pads to padded_dim and rotates)
  rotator_->apply(vec, out);

  // Step 2: Compute sum and squared norm of the rotated query
  float sum_q = 0.0f;
  float norm_sq_q = 0.0f;
  for (uint32_t i = 0; i < padded_dim_; ++i) {
    sum_q += out[i];
    norm_sq_q += out[i] * out[i];
  }

  // Step 3: Append sum and norm_sq after the rotated query
  out[padded_dim_] = sum_q;
  out[padded_dim_ + 1] = norm_sq_q;
#else
  (void)input;
  (void)output;
#endif
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

float RabitqQuantizer::calc_distance_dp_query(const void *dp,
                                              const void *query) const {
#if RABITQ_SUPPORTED
  const auto *dp_bytes = reinterpret_cast<const char *>(dp);
  const auto *q_floats = reinterpret_cast<const float *>(query);

  // Parse data layout
  const auto *code = reinterpret_cast<const uint8_t *>(dp_bytes);
  const float *factors =
      reinterpret_cast<const float *>(dp_bytes + padded_dim_);
  float f_add = factors[0];
  float f_rescale = factors[1];
  // factors[2] = f_error (unused in point estimate)

  // Parse query layout
  const float *rotated_q = q_floats;
  float sum_q = q_floats[padded_dim_];
  float norm_sq_q = q_floats[padded_dim_ + 1];

  // Compute ip(q, code)
  float ip = ip_float_code(rotated_q, code, padded_dim_);

  // Centering constant
  float cb_val = cb();

  // Distance estimation
  float est_dist = f_add + f_rescale * (ip + cb_val * sum_q);

  // For L2, add query squared norm
  if (metric_type_ == MetricType::kSquaredEuclidean) {
    est_dist += norm_sq_q;
  }

  return est_dist;
#else
  (void)dp;
  (void)query;
  return 0.0f;
#endif
}

void RabitqQuantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                   int dp_num,
                                                   const void *query,
                                                   float *dist_list) const {
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = calc_distance_dp_query(dp_list[i], query);
  }
}

float RabitqQuantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  // Quantize the raw query on-the-fly then compute distance
  std::string qbuf;
  qbuf.resize(padded_dim_, '\0');
  quantize_query(query, &qbuf[0]);
  return calc_distance_dp_query(dp, qbuf.data());
}

void RabitqQuantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  thread_local std::string qbuf;
  qbuf.resize(padded_dim_, '\0');
  quantize_query(query, &qbuf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, qbuf.data(), dist_list);
}

float RabitqQuantizer::calc_distance_dp_dp(const void *dp1,
                                           const void *dp2) const {
#if RABITQ_SUPPORTED
  // For dp-dp distance, treat dp2's code as a pseudo-query.
  // Convert dp2's codes to float and use dp1's factors for estimation.
  const auto *dp1_bytes = reinterpret_cast<const char *>(dp1);
  const auto *dp2_bytes = reinterpret_cast<const char *>(dp2);

  const auto *code1 = reinterpret_cast<const uint8_t *>(dp1_bytes);
  const float *factors1 =
      reinterpret_cast<const float *>(dp1_bytes + padded_dim_);
  float f_add_1 = factors1[0];
  float f_rescale_1 = factors1[1];

  const auto *code2 = reinterpret_cast<const uint8_t *>(dp2_bytes);
  const float *factors2 =
      reinterpret_cast<const float *>(dp2_bytes + padded_dim_);
  float f_add_2 = factors2[0];
  float f_rescale_2 = factors2[1];

  // Convert code2 to float pseudo-query
  float ip_1 = 0.0f, sum_code2 = 0.0f, norm_sq_code2 = 0.0f;
  float ip_2 = 0.0f, sum_code1 = 0.0f, norm_sq_code1 = 0.0f;
  for (uint32_t i = 0; i < padded_dim_; ++i) {
    float c1 = static_cast<float>(code1[i]);
    float c2 = static_cast<float>(code2[i]);
    ip_1 += c2 * static_cast<float>(code1[i]);
    ip_2 += c1 * static_cast<float>(code2[i]);
    sum_code1 += c1;
    sum_code2 += c2;
    norm_sq_code1 += c1 * c1;
    norm_sq_code2 += c2 * c2;
  }

  float cb_val = cb();

  // Asymmetric estimate from dp1's perspective (dp2 as pseudo-query)
  float est1 = f_add_1 + f_rescale_1 * (ip_1 + cb_val * sum_code2);
  if (metric_type_ == MetricType::kSquaredEuclidean) {
    est1 += norm_sq_code2;
  }

  // Asymmetric estimate from dp2's perspective (dp1 as pseudo-query)
  float est2 = f_add_2 + f_rescale_2 * (ip_2 + cb_val * sum_code1);
  if (metric_type_ == MetricType::kSquaredEuclidean) {
    est2 += norm_sq_code1;
  }

  // Average both directions for a symmetric estimate
  return 0.5f * (est1 + est2);
#else
  (void)dp1;
  (void)dp2;
  return 0.0f;
#endif
}

// ---------------------------------------------------------------------------
// Legacy interface
// ---------------------------------------------------------------------------

int RabitqQuantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                              std::string *out, IndexQueryMeta *ometa) const {
#if RABITQ_SUPPORTED
  if (!query || !out || !ometa) {
    return IndexError_InvalidArgument;
  }
  if (qmeta.data_type() != IndexMeta::DataType::DT_FP32) {
    return IndexError_Unsupported;
  }

  ometa->set_meta(IndexMeta::DataType::DT_RABITQ, 1, padded_dim_);
  out->resize(padded_dim_, '\0');
  quantize_query(query, &(*out)[0]);
  return 0;
#else
  (void)query;
  (void)qmeta;
  (void)out;
  (void)ometa;
  return IndexError_Unsupported;
#endif
}

int RabitqQuantizer::dequantize(const void * /*in*/,
                                const IndexQueryMeta & /*qmeta*/,
                                std::string * /*out*/) const {
  // RaBitQ is a lossy asymmetric quantizer; exact dequantization is not
  // supported. Callers should retain original vectors if reconstruction is
  // needed.
  return IndexError_NotImplemented;
}

DistanceImpl RabitqQuantizer::distance(const void *query,
                                       const IndexQueryMeta &qmeta) const {
#if RABITQ_SUPPORTED
  std::string qbuf;
  IndexQueryMeta ometa;
  if (this->quantize(query, qmeta, &qbuf, &ometa) != 0) {
    return DistanceImpl{};
  }

  // Create a lambda-based DistanceFunc that captures this quantizer
  auto dim = padded_dim_;
  auto metric = metric_type_;
  auto cb_val = cb();

  DistanceFunc func = [dim, metric, cb_val](const void *m, const void *q,
                                            size_t /*dim_arg*/, float *out) {
    const auto *dp_bytes = reinterpret_cast<const char *>(m);
    const auto *q_floats = reinterpret_cast<const float *>(q);

    const auto *code = reinterpret_cast<const uint8_t *>(dp_bytes);
    const float *factors = reinterpret_cast<const float *>(dp_bytes + dim);
    float f_add = factors[0];
    float f_rescale = factors[1];

    const float *rotated_q = q_floats;
    float sum_q = q_floats[dim];
    float norm_sq_q = q_floats[dim + 1];

    float ip = ip_float_code(rotated_q, code, dim);
    float est_dist = f_add + f_rescale * (ip + cb_val * sum_q);
    if (metric == MetricType::kSquaredEuclidean) {
      est_dist += norm_sq_q;
    }
    *out = est_dist;
  };

  return DistanceImpl(std::move(func), std::move(qbuf), padded_dim_);
#else
  (void)query;
  (void)qmeta;
  return DistanceImpl{};
#endif
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

int RabitqQuantizer::serialize(std::string *out) const {
#if RABITQ_SUPPORTED
  if (!out) return IndexError_InvalidArgument;

  // Serialize rotator
  std::string rotator_blob;
  if (rotator_) {
    int rc = rotator_->serialize(&rotator_blob);
    if (rc != 0) return rc;
  }

  // Payload: [total_bits (4B)] [rotator_blob]
  const uint32_t payload_size =
      sizeof(uint32_t) + static_cast<uint32_t>(rotator_blob.size());
  out->resize(sizeof(QuantizerSerHeader) + payload_size);

  auto *header = reinterpret_cast<QuantizerSerHeader *>(&(*out)[0]);
  header->magic = kQuantizerMagic;
  header->version = kQuantizerSerVersion;
  header->quant_type = static_cast<uint16_t>(type_);
  header->dim = original_dim_;
  header->metric = static_cast<uint32_t>(metric_type_);
  header->payload_size = payload_size;
  header->reserved = 0;

  char *payload = &(*out)[sizeof(QuantizerSerHeader)];
  std::memcpy(payload, &total_bits_, sizeof(uint32_t));
  if (!rotator_blob.empty()) {
    std::memcpy(payload + sizeof(uint32_t), rotator_blob.data(),
                rotator_blob.size());
  }
  return 0;
#else
  (void)out;
  return IndexError_Unsupported;
#endif
}

int RabitqQuantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

int RabitqQuantizer::deserialize(const void *data, size_t len) {
#if RABITQ_SUPPORTED
  if (!data || len < sizeof(QuantizerSerHeader)) {
    return IndexError_InvalidArgument;
  }

  const auto *header = reinterpret_cast<const QuantizerSerHeader *>(data);
  if (header->magic != kQuantizerMagic ||
      header->version != kQuantizerSerVersion ||
      header->payload_size < sizeof(uint32_t) ||
      len < sizeof(QuantizerSerHeader) + header->payload_size) {
    return IndexError_InvalidArgument;
  }

  if (header->dim != original_dim_ ||
      header->metric != static_cast<uint32_t>(metric_type_)) {
    return IndexError_InvalidArgument;
  }

  const char *payload =
      reinterpret_cast<const char *>(data) + sizeof(QuantizerSerHeader);

  // Read total_bits
  std::memcpy(&total_bits_, payload, sizeof(uint32_t));
  if (total_bits_ < 1 || total_bits_ > 8) {
    return IndexError_InvalidArgument;
  }

  // Read rotator blob
  rotator_.reset();
  if (header->payload_size > sizeof(uint32_t)) {
    const char *rot_ptr = payload + sizeof(uint32_t);
    const size_t rot_len = header->payload_size - sizeof(uint32_t);
    rotator_ = CreateRotatorFromBlob(rot_ptr, rot_len);
    if (!rotator_) {
      return IndexError_InvalidArgument;
    }
  }

  padded_dim_ = ((original_dim_ + 63) / 64) * 64;
  if (total_bits_ > 1) {
    rabitq_config_ = rabitqlib::quant::faster_config(padded_dim_, total_bits_);
  }

  return 0;
#else
  (void)data;
  (void)len;
  return IndexError_Unsupported;
#endif
}

INDEX_FACTORY_REGISTER_QUANTIZER(RabitqQuantizer);

}  // namespace turbo
}  // namespace zvec
