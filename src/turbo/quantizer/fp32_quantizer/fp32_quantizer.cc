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

#include "quantizer/fp32_quantizer/fp32_quantizer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "core/quantizer/record_quantizer.h"

namespace zvec {
namespace turbo {

int Fp32Quantizer::init(const IndexMeta &meta,
                        const ailego::Params & /*params*/) {
  meta_ = meta;

  meta_.set_meta(IndexMeta::DataType::DT_FP32, meta.dimension());

  auto metric_name = meta.metric_name();
  if (metric_name == "Cosine") {
    extra_meta_size_ = EXTRA_META_SIZE_COSINE;
    meta_.set_extra_meta_size(extra_meta_size_);
  }

  // `meta.dimension()` may be either the raw dim (when the caller passes a
  // bare meta, e.g. unit tests) or the inflated storage dim (data + extras)
  // when an upstream converter such as CosineConverter has already appended
  // a norm float and set meta.extra_meta_size(). Distinguish via the input
  // meta's extra_meta_size: if it is already set, the dim is inflated and
  // we strip the extras to recover the raw dim; otherwise the dim is raw.
  if (meta.extra_meta_size() > 0) {
    original_dim_ =
        meta.dimension() -
        meta.extra_meta_size() / IndexMeta::UnitSizeof(meta.data_type());
  } else {
    original_dim_ = meta.dimension();
  }

  // Cache the distance dispatch for the new Quantizer interface.
  dp_query_func_ =
      get_distance_func(metric_from_name(metric_name), DataType::kFp32,
                        QuantizeType::kDefault, CpuArchType::kAuto);
  dp_query_batch_func_ =
      get_batch_distance_func(metric_from_name(metric_name), DataType::kFp32,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  return 0;
}

int Fp32Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  if (qmeta.unit_size() != sizeof(float)) {
    return IndexError_Unsupported;
  }

  // qmeta.dimension() may be the inflated (data + extras) dimension when the
  // caller uses meta_.dimension() directly (e.g. HnswDistCalculator). Use the
  // raw original dim we recorded at init() to avoid over-reading the query.
  size_t raw_dim = (original_dim_ != 0 && qmeta.dimension() >= original_dim_)
                       ? original_dim_
                       : qmeta.dimension();
  size_t byte_size = raw_dim * sizeof(float);
  out->resize(byte_size);
  std::memcpy(&(*out)[0], query, byte_size);

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_FP32, raw_dim,
                  static_cast<uint32_t>(type_), extra_meta_size_);

  return 0;
}

int Fp32Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                              std::string *out) const {
  size_t byte_size = qmeta.dimension() * sizeof(float);
  out->resize(byte_size);
  std::memcpy(out->data(), in, byte_size);
  return 0;
}

DistanceImpl Fp32Quantizer::distance(const void *query,
                                     const IndexQueryMeta &qmeta) const {
  auto metric = metric_from_name(meta_.metric_name());
  auto func = get_distance_func(metric, DataType::kFp32, QuantizeType::kDefault,
                                CpuArchType::kAuto);
  if (!func) {
    return DistanceImpl{};
  }
  auto batch_func = get_batch_distance_func(
      metric, DataType::kFp32, QuantizeType::kDefault, CpuArchType::kAuto);

  // The query is assumed to be already quantized — copy it directly.
  std::string quantized_query(static_cast<const char *>(query),
                              qmeta.element_size());
  return DistanceImpl(std::move(func), std::move(batch_func),
                      std::move(quantized_query), original_dim_);
}

void Fp32Quantizer::quantize_one(const void *input, void *output) const {
  std::memcpy(output, input,
              static_cast<size_t>(original_dim_) * sizeof(float));
}

float Fp32Quantizer::calc_distance_dp_query(const void *dp,
                                            const void *query) const {
  float d = 0.0f;
  if (dp_query_func_) {
    dp_query_func_(dp, query, original_dim_, &d);
  }
  return d;
}

void Fp32Quantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                 int dp_num, const void *query,
                                                 float *dist_list) const {
  if (dp_query_batch_func_) {
    dp_query_batch_func_(const_cast<const void **>(dp_list), query,
                         static_cast<size_t>(dp_num), original_dim_, dist_list);
    return;
  }
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = calc_distance_dp_query(dp_list[i], query);
  }
}

float Fp32Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  return calc_distance_dp_query(dp, buf.data());
}

void Fp32Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::string buf(quantized_length(), '\0');
  quantize_one(query, &buf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, buf.data(), dist_list);
}

float Fp32Quantizer::calc_distance_dp_dp(const void *dp1,
                                         const void *dp2) const {
  return calc_distance_dp_query(dp1, dp2);
}

INDEX_FACTORY_REGISTER_QUANTIZER(Fp32Quantizer);

}  // namespace turbo
}  // namespace zvec
