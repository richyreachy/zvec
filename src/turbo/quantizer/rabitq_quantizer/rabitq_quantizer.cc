// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "rabitq_quantizer.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <rabitqlib/quantization/rabitq_impl.hpp>
#include <zvec/core/framework/index_factory.h>

namespace zvec::turbo {
namespace {
float ReadFloat(const void *data, size_t offset) {
  float value;
  std::memcpy(&value, static_cast<const char *>(data) + offset, sizeof(value));
  return value;
}
bool FiniteInput(const void *data, size_t dim) {
  for (size_t i = 0; i < dim; ++i)
    if (!std::isfinite(ReadFloat(data, i * sizeof(float)))) return false;
  return true;
}
}  // namespace

int RabitqQuantizer::init(const IndexMeta &meta, const ailego::Params &params) {
  int64_t bits = 7;
  if (params.has(RABITQ_TOTAL_BITS) && !params.get(RABITQ_TOTAL_BITS, &bits))
    return kErrInvalidArgument;
  const auto metric = metric_from_name(meta.metric_name());
  if (meta.data_type() != IndexMeta::DT_FP32 || meta.dimension() < 2 ||
      meta.dimension() > 4095 || bits < 1 || bits > 8 ||
      meta.element_size() != static_cast<size_t>(meta.dimension()) * sizeof(float) ||
      (metric != MetricType::kSquaredEuclidean &&
       metric != MetricType::kInnerProduct && metric != MetricType::kCosine))
    return kErrInvalidArgument;
  dim_ = static_cast<int>(meta.dimension());
  bits_ = static_cast<int>(bits);
  metric_ = metric;
  rotator_ = FhtRotator::create(dim_);
  if (!rotator_) return kErrRuntime;
  rescale_ =
      bits_ > 1
          ? rabitqlib::quant::rabitq_impl::ex_bits::get_const_scaling_factors(
                dim_, bits_ - 1)
          : -1;
  meta_ = meta;
  // DT_BINARY rounds to whole 32-bit words. Describe our exact byte layout
  // with DT_UINT8 instead; original dimension is owned by the quantizer.
  meta_.set_meta(IndexMeta::DT_UINT8,
                 static_cast<uint32_t>(quantized_datapoint_vector_length()));
  ailego::Params encoding;
  encoding.set(RABITQ_TOTAL_BITS, bits_);
  meta_.set_quantizer("RabitqQuantizer", 0, encoding);
  return 0;
}

float RabitqQuantizer::rotate(const void *input,
                              std::vector<float> *out) const {
  std::vector<float> raw(dim_);
  std::memcpy(raw.data(), input, dim_ * sizeof(float));
  double norm2 = 0;
  for (float v : raw) norm2 += static_cast<double>(v) * v;
  const float norm = static_cast<float>(std::sqrt(norm2));
  if (metric_ == MetricType::kCosine && norm > 0)
    for (float &v : raw) v /= norm;
  out->resize(dim_);
  rotator_->apply(raw.data(), out->data());
  return norm;
}

void RabitqQuantizer::quantize_data(const void *input, void *output) const {
  std::vector<float> rotated;
  const float original_norm = rotate(input, &rotated);
  std::vector<uint8_t> codes(dim_, 0);
  // The library produces the extra bits (including the negative-coordinate
  // complement). Add the sign bit to obtain the complete midpoint code.
  if (bits_ > 1 && original_norm > 0) {
    rabitqlib::quant::rabitq_impl::ex_bits::ex_bits_code<float, uint8_t>(
        rotated.data(), dim_, bits_ - 1, codes.data(), rescale_);
  }
  const float midpoint = ((1u << bits_) - 1) * 0.5f;
  double norm2 = 0, dot = 0;
  std::memset(output, 0, quantized_datapoint_vector_length());
  auto *packed = static_cast<uint8_t *>(output) + 3 * sizeof(float);
  for (int i = 0; i < dim_; ++i) {
    codes[i] += static_cast<uint8_t>((rotated[i] >= 0) << (bits_ - 1));
    norm2 += static_cast<double>(rotated[i]) * rotated[i];
    dot += static_cast<double>(rotated[i]) * (codes[i] - midpoint);
    const size_t pos = static_cast<size_t>(i) * bits_;
    const unsigned shift = pos % 8;
    packed[pos / 8] |= codes[i] << shift;
    if (shift + bits_ > 8) packed[pos / 8 + 1] |= codes[i] >> (8 - shift);
  }
  // RaBitQ's unbiased inner-product estimator: ||x||² / <x, code>.
  // Keep the original norm independently for vector reconstruction (Cosine).
  const float factors[] = {static_cast<float>(norm2),
                           dot > 0 ? static_cast<float>(norm2 / dot) : 0.0f,
                           original_norm};
  std::memcpy(output, factors, sizeof(factors));
}

void RabitqQuantizer::quantize_query(const void *input, void *output) const {
  std::vector<float> rotated;
  rotate(input, &rotated);
  double norm2 = 0;
  for (float v : rotated) norm2 += static_cast<double>(v) * v;
  rotated.push_back(static_cast<float>(norm2));
  std::memcpy(output, rotated.data(), quantized_query_vector_length());
}

unsigned RabitqQuantizer::code(const void *data, int i) const {
  const auto *packed = static_cast<const uint8_t *>(data) + 3 * sizeof(float);
  const size_t pos = static_cast<size_t>(i) * bits_;
  const unsigned shift = pos % 8;
  unsigned value = packed[pos / 8] >> shift;
  if (shift + bits_ > 8)
    value |= static_cast<unsigned>(packed[pos / 8 + 1]) << (8 - shift);
  return value & ((1u << bits_) - 1);
}

float RabitqQuantizer::calc_distance_dp_query(const void *dp,
                                              const void *q) const {
  const float midpoint = ((1u << bits_) - 1) * 0.5f;
  double dot = 0;
  for (int i = 0; i < dim_; ++i)
    dot += (code(dp, i) - midpoint) *
           static_cast<double>(ReadFloat(q, i * sizeof(float)));
  const float ip = static_cast<float>(dot * ReadFloat(dp, sizeof(float)));
  if (metric_ == MetricType::kSquaredEuclidean)
    return ReadFloat(dp, 0) + ReadFloat(q, dim_ * sizeof(float)) - 2 * ip;
  return metric_ == MetricType::kCosine ? 1 - ip : -ip;
}

void RabitqQuantizer::calc_distance_dp_query_batch(const void *const *dp, int n,
                                                   const void *q,
                                                   float *out) const {
  for (int i = 0; i < n; ++i) out[i] = calc_distance_dp_query(dp[i], q);
}

float RabitqQuantizer::calc_distance_dp_query_unquantized(const void *dp,
                                                          const void *q) const {
  std::string query(quantized_query_vector_length(), '\0');
  quantize_query(q, query.data());
  return calc_distance_dp_query(dp, query.data());
}

void RabitqQuantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp, int n, const void *q, float *out) const {
  if (n <= 0) return;
  std::string query(quantized_query_vector_length(), '\0');
  quantize_query(q, query.data());
  calc_distance_dp_query_batch(dp, n, query.data(), out);
}

float RabitqQuantizer::calc_distance_dp_dp(const void *, const void *) const {
  // HNSW rejects inserts without an original-vector provider. There is no
  // symmetric code distance: RaBitQ search uses an asymmetric estimator.
  return std::numeric_limits<float>::infinity();
}

bool RabitqQuantizer::valid_input(const IndexQueryMeta &meta) const {
  return meta.data_type() == IndexMeta::DT_FP32 &&
         meta.dimension() == static_cast<uint32_t>(dim_) &&
         meta.element_size() == dim_ * sizeof(float);
}

int RabitqQuantizer::quantize(const void *data, const IndexQueryMeta &meta,
                              std::string *out, IndexQueryMeta *ometa) const {
  if (!data || !out || !ometa || !valid_input(meta) || !FiniteInput(data, dim_))
    return kErrInvalidArgument;
  out->resize(quantized_query_vector_length());
  quantize_query(data, out->data());
  ometa->set_meta(IndexMeta::DT_FP32, dim_, static_cast<uint32_t>(type_),
                  sizeof(float));
  return 0;
}

int RabitqQuantizer::quantize_datapoint(const void *data,
                                        const IndexQueryMeta &meta,
                                        std::string *out,
                                        IndexQueryMeta *ometa) const {
  if (!data || !out || !ometa || !valid_input(meta) || !FiniteInput(data, dim_))
    return kErrInvalidArgument;
  out->resize(quantized_datapoint_vector_length());
  quantize_data(data, out->data());
  ometa->set_meta(IndexMeta::DT_UINT8, static_cast<uint32_t>(out->size()),
                  static_cast<uint32_t>(type_), 0);
  return 0;
}

int RabitqQuantizer::dequantize(const void *data, const IndexQueryMeta &meta,
                                std::string *out) const {
  if (!data || !out ||
      meta.element_size() != quantized_datapoint_vector_length())
    return kErrInvalidArgument;
  std::vector<float> rotated(dim_), raw(dim_);
  const float midpoint = ((1u << bits_) - 1) * 0.5f;
  double code_norm2 = 0;
  for (int i = 0; i < dim_; ++i) {
    rotated[i] = code(data, i) - midpoint;
    code_norm2 += static_cast<double>(rotated[i]) * rotated[i];
  }
  // For reconstruction use the least-squares scale, rather than the unbiased
  // estimator scale used for search: <x, code> / ||code||².
  const float unbiased_scale = ReadFloat(data, sizeof(float));
  float scale = unbiased_scale > 0
                    ? ReadFloat(data, 0) / (unbiased_scale * code_norm2)
                    : 0;
  if (metric_ == MetricType::kCosine)
    scale *= ReadFloat(data, 2 * sizeof(float));
  for (float &v : rotated) v *= scale;
  rotator_->apply_inverse(rotated.data(), raw.data());
  out->assign(reinterpret_cast<const char *>(raw.data()), dim_ * sizeof(float));
  return 0;
}

DistanceImpl RabitqQuantizer::distance(const void *query,
                                       const IndexQueryMeta &meta) const {
  if (!query || meta.element_size() != quantized_query_vector_length())
    return {};
  auto single = [this](const void *dp, const void *q, size_t, float *out) {
    *out = calc_distance_dp_query(dp, q);
  };
  return DistanceImpl(
      single, {},
      std::string(static_cast<const char *>(query), meta.element_size()), dim_);
}

int RabitqQuantizer::serialize(std::string *out) const {
  if (!out || !rotator_) return kErrInvalidArgument;
  std::string rotation;
  int ret = rotator_->serialize(&rotation);
  if (ret != 0) return ret;
  QuantizerSerHeader header{};
  header.magic = kQuantizerMagic;
  header.version = kQuantizerSerVersion;
  header.quant_type = static_cast<uint16_t>(type_);
  header.dim = dim_;
  header.metric = static_cast<uint32_t>(metric_);
  header.data_type = static_cast<uint16_t>(DataType::kUint8);
  header.payload_size = sizeof(uint32_t) + rotation.size();
  const uint32_t bits = bits_;
  out->assign(reinterpret_cast<const char *>(&header), sizeof(header));
  out->append(reinterpret_cast<const char *>(&bits), sizeof(bits));
  out->append(rotation);
  return 0;
}

int RabitqQuantizer::deserialize(const void *data, size_t len) {
  if (!data || !rotator_ || len < sizeof(QuantizerSerHeader) + sizeof(uint32_t))
    return kErrInvalidArgument;
  QuantizerSerHeader header;
  std::memcpy(&header, data, sizeof(header));
  uint32_t bits;
  const char *payload = static_cast<const char *>(data) + sizeof(header);
  std::memcpy(&bits, payload, sizeof(bits));
  if (header.magic != kQuantizerMagic ||
      header.version != kQuantizerSerVersion ||
      header.quant_type != static_cast<uint16_t>(type_) ||
      header.dim != static_cast<uint32_t>(dim_) ||
      header.metric != static_cast<uint32_t>(metric_) ||
      header.data_type != static_cast<uint16_t>(DataType::kUint8) ||
      header.reserved != 0 || bits != static_cast<uint32_t>(bits_) ||
      header.payload_size != len - sizeof(header))
    return kErrInvalidArgument;
  const size_t rotation_size = sizeof(RotatorSerHeader) + 4 * ((dim_ + 7) / 8);
  if (len - sizeof(header) - sizeof(bits) != rotation_size)
    return kErrInvalidArgument;
  auto rotation = FhtRotator::from_blob(payload + sizeof(bits), rotation_size);
  if (!rotation || rotation->in_dim() != dim_) return kErrInvalidArgument;
  rotator_ = std::move(rotation);
  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(RabitqQuantizer);
}  // namespace zvec::turbo
