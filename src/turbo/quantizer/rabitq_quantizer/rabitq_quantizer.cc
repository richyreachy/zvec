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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <rabitqlib/quantization/rabitq_impl.hpp>
#if RABITQ_SUPPORTED
#include <rabitqlib/utils/space.hpp>
#include <rabitqlib/utils/warmup_space.hpp>
#endif
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_features.h>
#include <zvec/core/framework/index_threads.h>

namespace zvec::turbo {
namespace {
// Version the record/query layout independently of the common quantizer header.
#if RABITQ_SUPPORTED
// Bumped from 0x52425132: records and query blobs now follow rabitqlib's
// BinDataMap/ExDataMap layout so the dispatched SIMD estimators can consume
// them in place. Indexes written with the scalar layout must be rebuilt.
constexpr uint32_t kSplitFormat = 0x52425133;
constexpr size_t kBinPrefix = 8;  // u32 cluster + u32 zero pad for alignment
#else
constexpr uint32_t kSplitFormat = 0x52425132;
#endif
constexpr size_t kRecordHeader = 9 * sizeof(float);
uint32_t ReadUint(const void *data, size_t offset = 0) {
  uint32_t value;
  std::memcpy(&value, static_cast<const char *>(data) + offset, sizeof(value));
  return value;
}
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
  int64_t clusters = 16, samples = 0;
  if ((params.has(RABITQ_NUM_CLUSTERS) &&
       !params.get(RABITQ_NUM_CLUSTERS, &clusters)) ||
      (params.has(RABITQ_SAMPLE_COUNT) &&
       !params.get(RABITQ_SAMPLE_COUNT, &samples)) ||
      clusters < 1 || clusters > 65536 || samples < 0)
    return kErrInvalidArgument;
  const auto metric = metric_from_name(meta.metric_name());
  if (meta.data_type() != IndexMeta::DT_FP32 || meta.dimension() < 2 ||
      meta.dimension() > 4095 || bits < 1 || bits > 9 ||
      meta.element_size() !=
          static_cast<size_t>(meta.dimension()) * sizeof(float) ||
      (metric != MetricType::kSquaredEuclidean &&
       metric != MetricType::kInnerProduct && metric != MetricType::kCosine))
    return kErrInvalidArgument;
  dim_ = static_cast<int>(meta.dimension());
  bits_ = static_cast<int>(bits);
  metric_ = metric;
  padded_dim_ = (dim_ + 63) / 64 * 64;
  num_clusters_ = static_cast<uint32_t>(clusters);
  sample_count_ = static_cast<size_t>(samples);
  centroids_.clear();
  rotator_ = FhtRotator::create(padded_dim_);
  if (!rotator_) return kErrRuntime;
  rescale_ =
      bits_ > 1
          ? rabitqlib::quant::rabitq_impl::ex_bits::get_const_scaling_factors(
                padded_dim_, bits_ - 1)
          : -1;
  // The query is always 4-bit quantized (3 extra bits). get_const_scaling_
  // factors() costs ~10ms (random matrix + heap sweeps), so it must never run
  // per query; it is deterministic and cheap to recompute here.
  query_rescale_ =
      rabitqlib::quant::rabitq_impl::ex_bits::get_const_scaling_factors(
          padded_dim_, 3);
#if RABITQ_SUPPORTED
  if (bits_ > 1)
    ex_ipfunc_ =
        rabitqlib::select_excode_ipfunc(static_cast<size_t>(bits_ - 1));
#endif
  meta_ = meta;
  // DT_BINARY rounds to whole 32-bit words. Describe our exact byte layout
  // with DT_UINT8 instead; original dimension is owned by the quantizer.
  meta_.set_meta(IndexMeta::DT_UINT8,
                 static_cast<uint32_t>(quantized_datapoint_vector_length()));
  ailego::Params encoding;
  encoding.set(RABITQ_TOTAL_BITS, bits_);
  encoding.set(RABITQ_NUM_CLUSTERS, num_clusters_);
  encoding.set(RABITQ_SAMPLE_COUNT, sample_count_);
  meta_.set_quantizer("RabitqQuantizer", 0, encoding);
  return 0;
}

float RabitqQuantizer::rotate(const void *input,
                              std::vector<float> *out) const {
  std::vector<float> raw(padded_dim_, 0);
  std::memcpy(raw.data(), input, dim_ * sizeof(float));
  double norm2 = 0;
  for (float v : raw) norm2 += static_cast<double>(v) * v;
  const float norm = static_cast<float>(std::sqrt(norm2));
  if (metric_ == MetricType::kCosine && norm > 0)
    for (float &v : raw) v /= norm;
  out->resize(padded_dim_);
  rotator_->apply(raw.data(), out->data());
  return norm;
}

int RabitqQuantizer::train(IndexHolder::Pointer holder) {
  if (!rotator_ || !require_train() || !holder || holder->count() == 0 ||
      holder->data_type() != IndexMeta::DT_FP32 ||
      holder->dimension() != static_cast<size_t>(dim_) ||
      holder->element_size() != dim_ * sizeof(float))
    return kErrInvalidArgument;
  IndexMeta training_meta(IndexMeta::DT_FP32, dim_);
  training_meta.set_metric(metric_ == MetricType::kSquaredEuclidean
                               ? "SquaredEuclidean"
                               : "InnerProduct",
                           0, ailego::Params{});
  const size_t count = sample_count_ == 0
                           ? holder->count()
                           : std::min(sample_count_, holder->count());
  auto samples = std::make_shared<SampleIndexFeatures<CompactIndexFeatures>>(
      training_meta, count);
  auto it = holder->create_iterator();
  if (!it) return kErrInvalidArgument;
  for (; it->is_valid(); it->next()) {
    if (!it->data() || !FiniteInput(it->data(), dim_))
      return kErrInvalidArgument;
    std::vector<float> value(dim_);
    std::memcpy(value.data(), it->data(), dim_ * sizeof(float));
    if (metric_ == MetricType::kCosine) {
      double norm2 = 0;
      for (float v : value) norm2 += static_cast<double>(v) * v;
      if (norm2 > 0)
        for (float &v : value) v /= std::sqrt(norm2);
    }
    samples->emplace(value.data());
  }
  if (samples->count() == 0) return kErrInvalidArgument;
  std::vector<float> centroids;
  auto append = [&](const void *value) {
    std::vector<float> padded(padded_dim_, 0), rotated(padded_dim_);
    std::memcpy(padded.data(), value, dim_ * sizeof(float));
    rotator_->apply(padded.data(), rotated.data());
    centroids.insert(centroids.end(), rotated.begin(), rotated.end());
  };
  if (samples->count() <= num_clusters_) {
    for (size_t i = 0; i < samples->count(); ++i) {
      std::vector<float> center(dim_);
      std::memcpy(center.data(), samples->element(i), dim_ * sizeof(float));
      // Match spherical k-means centroids for IP/cosine even when there are
      // fewer samples than requested centers and clustering is unnecessary.
      if (metric_ != MetricType::kSquaredEuclidean) {
        double norm2 = 0;
        for (float v : center) norm2 += static_cast<double>(v) * v;
        if (norm2 > 0)
          for (float &v : center) v /= std::sqrt(norm2);
      }
      append(center.data());
    }
  } else {
    // Reuse the same centroid trainer as the dedicated HNSW-RaBitQ converter.
    auto cluster = IndexFactory::CreateCluster("OptKmeansCluster");
    if (!cluster) return kErrRuntime;
    int ret = cluster->init(training_meta, ailego::Params{});
    if (ret != 0) return ret;
    ret = cluster->mount(samples);
    if (ret != 0) return ret;
    cluster->suggest(num_clusters_);
    IndexCluster::CentroidList centers;
    ret = cluster->cluster(std::make_shared<SingleQueueIndexThreads>(0, false),
                           centers);
    if (ret != 0) return ret;
    if (centers.empty() || centers.size() > num_clusters_) return kErrRuntime;
    for (const auto &center : centers) append(center.feature());
  }
  centroids_ = std::move(centroids);
  return 0;
}

uint32_t RabitqQuantizer::nearest_centroid(const std::vector<float> &x) const {
  uint32_t best = 0;
  double best_distance = std::numeric_limits<double>::infinity();
  for (size_t c = 0; c < centroids_.size() / padded_dim_; ++c) {
    double distance = 0;
    for (int i = 0; i < padded_dim_; ++i) {
      const double v = centroids_[c * padded_dim_ + i];
      distance += metric_ == MetricType::kSquaredEuclidean
                      ? (x[i] - v) * (x[i] - v)
                      : -x[i] * v;
    }
    if (distance < best_distance) {
      best_distance = distance;
      best = c;
    }
  }
  return best;
}

#if RABITQ_SUPPORTED
void RabitqQuantizer::quantize_data(const void *input, void *output) const {
  std::vector<float> rotated;
  const float original_norm = rotate(input, &rotated);
  const uint32_t cluster = nearest_centroid(rotated);
  const float *centroid = centroids_.data() + cluster * padded_dim_;
  std::vector<int> binary(padded_dim_);
  std::vector<uint8_t> extra(padded_dim_, 0);
  const auto metric = metric_ == MetricType::kSquaredEuclidean
                          ? rabitqlib::METRIC_L2
                          : rabitqlib::METRIC_IP;
  float bin_add, bin_scale, bin_error;
  rabitqlib::quant::rabitq_impl::one_bit::one_bit_code_with_factor(
      rotated.data(), centroid, padded_dim_, binary.data(), bin_add, bin_scale,
      bin_error, metric);
  double norm2 = 0;
  for (int i = 0; i < padded_dim_; ++i) {
    const double r = rotated[i] - centroid[i];
    norm2 += r * r;
  }
  // The upstream formula is undefined for a zero residual and can round
  // slightly negative inside sqrt. Both mean a zero error radius here.
  if (!std::isfinite(bin_error)) bin_error = 0;
  float full_add = bin_add, full_scale = bin_scale, full_error;
  if (bits_ > 1 && norm2 > 0) {
    rabitqlib::quant::rabitq_impl::ex_bits::ex_bits_code_with_factor(
        rotated.data(), centroid, padded_dim_, bits_ - 1, extra.data(),
        full_add, full_scale, full_error, metric, rescale_);
  }
  double code_norm2 = 0, dot = 0;
  const float midpoint = ((1u << bits_) - 1) * 0.5f;
  for (int i = 0; i < padded_dim_; ++i) {
    const float code = (binary[i] << (bits_ - 1)) + extra[i] - midpoint;
    dot += (rotated[i] - centroid[i]) * static_cast<double>(code);
    code_norm2 += static_cast<double>(code) * code;
  }
  // Layout: [u32 cluster][u32 pad][BinDataMap][ExDataMap][orig_norm,
  // resid_scale]. BinDataMap/ExDataMap mirror rabitqlib's data_layout.hpp so
  // estimate() can feed records straight into the dispatched SIMD kernels.
  std::memset(output, 0, quantized_datapoint_vector_length());
  auto *out = static_cast<char *>(output);
  std::memcpy(out, &cluster, sizeof(cluster));
  char *bin_data = out + kBinPrefix;
  rabitqlib::pack_binary(binary.data(), reinterpret_cast<uint64_t *>(bin_data),
                         static_cast<size_t>(padded_dim_));
  std::memcpy(bin_data + padded_dim_ / 8, &bin_add, sizeof(bin_add));
  std::memcpy(bin_data + padded_dim_ / 8 + 4, &bin_scale, sizeof(bin_scale));
  std::memcpy(bin_data + padded_dim_ / 8 + 8, &bin_error, sizeof(bin_error));
  char *ex_data = bin_data + padded_dim_ / 8 + 12;
  const size_t ex_bytes =
      bits_ > 1 ? static_cast<size_t>(padded_dim_) * (bits_ - 1) / 8 : 0;
  if (bits_ > 1) {
    rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
        extra.data(), reinterpret_cast<uint8_t *>(ex_data), padded_dim_,
        static_cast<size_t>(bits_ - 1));
    std::memcpy(ex_data + ex_bytes, &full_add, sizeof(full_add));
    std::memcpy(ex_data + ex_bytes + 4, &full_scale, sizeof(full_scale));
  }
  const float tail[] = {original_norm, static_cast<float>(dot / code_norm2)};
  std::memcpy(ex_data + ex_bytes + 8, tail, sizeof(tail));
}
#else
void RabitqQuantizer::quantize_data(const void *input, void *output) const {
  std::vector<float> rotated;
  const float original_norm = rotate(input, &rotated);
  const uint32_t cluster = nearest_centroid(rotated);
  const float *centroid = centroids_.data() + cluster * padded_dim_;
  std::vector<int> binary(padded_dim_);
  std::vector<uint8_t> extra(padded_dim_, 0);
  const auto metric = metric_ == MetricType::kSquaredEuclidean
                          ? rabitqlib::METRIC_L2
                          : rabitqlib::METRIC_IP;
  float bin_add, bin_scale, bin_error;
  rabitqlib::quant::rabitq_impl::one_bit::one_bit_code_with_factor(
      rotated.data(), centroid, padded_dim_, binary.data(), bin_add, bin_scale,
      bin_error, metric);
  double norm2 = 0;
  for (int i = 0; i < padded_dim_; ++i) {
    const double r = rotated[i] - centroid[i];
    norm2 += r * r;
  }
  // The upstream formula is undefined for a zero residual and can round
  // slightly negative inside sqrt. Both mean a zero error radius here.
  if (!std::isfinite(bin_error)) bin_error = 0;
  float full_add = bin_add, full_scale = bin_scale, full_error;
  if (bits_ > 1 && norm2 > 0) {
    rabitqlib::quant::rabitq_impl::ex_bits::ex_bits_code_with_factor(
        rotated.data(), centroid, padded_dim_, bits_ - 1, extra.data(),
        full_add, full_scale, full_error, metric, rescale_);
  }
  double code_norm2 = 0, dot = 0;
  const float midpoint = ((1u << bits_) - 1) * 0.5f;
  std::memset(output, 0, quantized_datapoint_vector_length());
  auto *signs = static_cast<uint8_t *>(output) + kRecordHeader;
  auto *packed = signs + padded_dim_ / 8;
  for (int i = 0; i < padded_dim_; ++i) {
    signs[i / 8] |= binary[i] << (i % 8);
    if (bits_ > 1) {
      const size_t pos = static_cast<size_t>(i) * (bits_ - 1);
      const unsigned shift = pos % 8;
      packed[pos / 8] |= extra[i] << shift;
      if (shift + bits_ - 1 > 8) packed[pos / 8 + 1] |= extra[i] >> (8 - shift);
    }
    const float code = (binary[i] << (bits_ - 1)) + extra[i] - midpoint;
    dot += (rotated[i] - centroid[i]) * static_cast<double>(code);
    code_norm2 += static_cast<double>(code) * code;
  }
  const float factors[] = {original_norm,
                           static_cast<float>(norm2),
                           static_cast<float>(dot / code_norm2),
                           bin_add,
                           bin_scale,
                           bin_error,
                           full_add,
                           full_scale};
  std::memcpy(output, &cluster, sizeof(cluster));
  std::memcpy(static_cast<char *>(output) + sizeof(cluster), factors,
              sizeof(factors));
}
#endif

#if RABITQ_SUPPORTED
void RabitqQuantizer::quantize_query(const void *input, void *output) const {
  std::vector<float> rotated;
  rotate(input, &rotated);
  std::memset(output, 0, quantized_query_vector_length());
  auto *out = static_cast<char *>(output);
  std::memcpy(out, rotated.data(),
              static_cast<size_t>(padded_dim_) * sizeof(float));
  auto *query_bin =
      reinterpret_cast<uint64_t *>(out + padded_dim_ * sizeof(float));
  std::vector<uint8_t> codes(padded_dim_, 0);
  double norm2 = 0;
  float sumq = 0;  // float accumulation matches rabitqlib's query wrapper
  for (int i = 0; i < padded_dim_; ++i) {
    norm2 += static_cast<double>(rotated[i]) * rotated[i];
    sumq += rotated[i];
  }
  float delta = 0, vl = 0;
  if (norm2 > 0) {
    std::vector<float> zero(padded_dim_, 0);
    rabitqlib::quant::rabitq_impl::total_bits::rabitq_scalar_impl(
        rotated.data(), zero.data(), padded_dim_, 4, codes.data(), delta, vl,
        query_rescale_);
  }
  // Same per-query precompute as rabitqlib::SplitSingleQuery: bit-plane
  // transposed 4-bit codes plus the folded query-sum corrections.
  rabitqlib::new_transpose_bin_512(codes.data(), query_bin, padded_dim_, 4);
  auto *scalars = reinterpret_cast<float *>(
      out + static_cast<size_t>(padded_dim_) * sizeof(float) +
      static_cast<size_t>(padded_dim_) * 4 / 8);
  const float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
  const float c_b = -static_cast<float>((1 << bits_) - 1) / 2.F;
  scalars[0] = delta;
  scalars[1] = vl;
  scalars[2] = sumq * c_1;
  scalars[3] = sumq * c_b;
  float *g_add = scalars + 4;
  float *g_error = g_add + num_clusters_;
  for (size_t c = 0; c < centroids_.size() / padded_dim_; ++c) {
    double distance = 0, dot = 0;
    for (int i = 0; i < padded_dim_; ++i) {
      const double center = centroids_[c * padded_dim_ + i];
      distance += (rotated[i] - center) * (rotated[i] - center);
      dot += rotated[i] * center;
    }
    g_add[c] = static_cast<float>(
        metric_ == MetricType::kSquaredEuclidean ? distance : -dot);
    g_error[c] = static_cast<float>(std::sqrt(distance));
  }
}
#else
void RabitqQuantizer::quantize_query(const void *input, void *output) const {
  std::vector<float> rotated;
  rotate(input, &rotated);
  std::vector<float> values(quantized_query_vector_length() / sizeof(float), 0);
  std::copy(rotated.begin(), rotated.end(), values.begin());
  std::vector<float> zero(padded_dim_, 0);
  std::vector<uint8_t> codes(padded_dim_, 0);
  double norm2 = 0;
  for (float v : rotated) norm2 += static_cast<double>(v) * v;
  float delta = 0, vl = 0;
  if (norm2 > 0) {
    rabitqlib::quant::rabitq_impl::total_bits::rabitq_scalar_impl(
        rotated.data(), zero.data(), padded_dim_, 4, codes.data(), delta, vl,
        query_rescale_);
  }
  // Scalar equivalent of SplitSingleQuery's 4-bit warmup. Keep the FP32
  // rotated query separately for full estimates and the original sum
  // correction.
  for (int i = 0; i < padded_dim_; ++i) {
    values[padded_dim_ + i] = codes[i] * delta + vl;
    values[2 * padded_dim_] += rotated[i];
  }
  for (size_t c = 0; c < centroids_.size() / padded_dim_; ++c) {
    double distance = 0, dot = 0;
    for (int i = 0; i < padded_dim_; ++i) {
      const double center = centroids_[c * padded_dim_ + i];
      distance += (rotated[i] - center) * (rotated[i] - center);
      dot += rotated[i] * center;
    }
    values[2 * padded_dim_ + 1 + c] =
        metric_ == MetricType::kSquaredEuclidean ? distance : -dot;
    values[2 * padded_dim_ + 1 + num_clusters_ + c] = std::sqrt(distance);
  }
  std::memcpy(output, values.data(), quantized_query_vector_length());
}
#endif

#if !RABITQ_SUPPORTED
unsigned RabitqQuantizer::sign(const void *data, int i) const {
  const auto *packed = static_cast<const uint8_t *>(data) + kRecordHeader;
  return (packed[i / 8] >> (i % 8)) & 1;
}

unsigned RabitqQuantizer::code(const void *data, int i) const {
  if (bits_ == 1) return sign(data, i);
  const auto *packed =
      static_cast<const uint8_t *>(data) + kRecordHeader + padded_dim_ / 8;
  const size_t pos = static_cast<size_t>(i) * (bits_ - 1);
  const unsigned shift = pos % 8;
  unsigned value = packed[pos / 8] >> shift;
  if (shift + bits_ - 1 > 8)
    value |= static_cast<unsigned>(packed[pos / 8 + 1]) << (8 - shift);
  return (sign(data, i) << (bits_ - 1)) | (value & ((1u << (bits_ - 1)) - 1));
}
#endif

#if RABITQ_SUPPORTED
float RabitqQuantizer::estimate(const void *dp, const void *q, bool full,
                                float *lower) const {
  const uint32_t cluster = ReadUint(dp);
  if (cluster >= centroids_.size() / padded_dim_) {
    *lower = std::numeric_limits<float>::infinity();
    return *lower;
  }
  // Mirrors rabitqlib::split_single_estdist / split_single_fulldist, reading
  // the precomputed query state and the in-place BinDataMap/ExDataMap record.
  const auto *query = static_cast<const char *>(q);
  const float *rotated_query = reinterpret_cast<const float *>(query);
  const auto *query_bin = reinterpret_cast<const uint64_t *>(
      query + static_cast<size_t>(padded_dim_) * sizeof(float));
  const auto *scalars = reinterpret_cast<const float *>(
      query + static_cast<size_t>(padded_dim_) * sizeof(float) +
      static_cast<size_t>(padded_dim_) * 4 / 8);
  const float *g_add = scalars + 4;
  const float *g_error = g_add + num_clusters_;
  const char *bin_data = static_cast<const char *>(dp) + kBinPrefix;
  const auto *bin_code = reinterpret_cast<const uint64_t *>(bin_data);
  const float f_add = ReadFloat(bin_data, padded_dim_ / 8);
  const float f_rescale = ReadFloat(bin_data, padded_dim_ / 8 + 4);
  const float f_error = ReadFloat(bin_data, padded_dim_ / 8 + 8);
  float score;
  if (!full) {
    const float ip_x0_qr = rabitqlib::warmup_ip_x0_q_512(
        bin_code, query_bin, scalars[0], scalars[1], padded_dim_, 4);
    score = f_add + g_add[cluster] + f_rescale * (ip_x0_qr + scalars[2]);
    *lower = score - f_error * g_error[cluster];
  } else {
    const char *ex_data = bin_data + padded_dim_ / 8 + 12;
    const size_t ex_bytes = static_cast<size_t>(padded_dim_) * (bits_ - 1) / 8;
    const float ip_x0_qr =
        rabitqlib::mask_ip_x0_q(rotated_query, bin_code, padded_dim_);
    score = ReadFloat(ex_data, ex_bytes) + g_add[cluster] +
            ReadFloat(ex_data, ex_bytes + 4) *
                (static_cast<float>(1u << (bits_ - 1)) * ip_x0_qr +
                 ex_ipfunc_(rotated_query,
                            reinterpret_cast<const uint8_t *>(ex_data),
                            static_cast<size_t>(padded_dim_)) +
                 scalars[3]);
    *lower = score -
             f_error * g_error[cluster] / static_cast<float>(1u << (bits_ - 1));
  }
  // Library IP estimators use 1-IP. Turbo's internal IP space uses -IP.
  if (metric_ == MetricType::kInnerProduct) {
    score -= 1;
    *lower -= 1;
  }
  return score;
}
#else
float RabitqQuantizer::estimate(const void *dp, const void *q, bool full,
                                float *lower) const {
  const uint32_t cluster = ReadUint(dp);
  if (cluster >= centroids_.size() / padded_dim_) {
    *lower = std::numeric_limits<float>::infinity();
    return *lower;
  }
  double dot = 0;
  for (int i = 0; i < padded_dim_; ++i) {
    dot += (full ? code(dp, i) : sign(dp, i)) *
           static_cast<double>(
               ReadFloat(q, (full ? i : padded_dim_ + i) * sizeof(float)));
  }
  dot -= (full ? ((1u << bits_) - 1) * 0.5f : 0.5f) *
         ReadFloat(q, 2 * padded_dim_ * sizeof(float));
  const float g_add =
      ReadFloat(q, (2 * padded_dim_ + 1 + cluster) * sizeof(float));
  const float g_error = ReadFloat(
      q, (2 * padded_dim_ + 1 + num_clusters_ + cluster) * sizeof(float));
  const float add = ReadFloat(dp, (full ? 7 : 4) * sizeof(float));
  const float scale = ReadFloat(dp, (full ? 8 : 5) * sizeof(float));
  // Library IP estimators use 1-IP. Turbo's internal IP space uses -IP.
  const float score = add + g_add + scale * dot -
                      (metric_ == MetricType::kInnerProduct ? 1 : 0);
  *lower = score - ReadFloat(dp, 6 * sizeof(float)) * g_error /
                       (full ? (1u << (bits_ - 1)) : 1);
  return score;
}
#endif

DistanceEstimate RabitqQuantizer::estimate_distance_dp_query(
    const void *dp, const void *q) const {
  float lower;
  const float distance = estimate(dp, q, false, &lower);
  // With no extra bits the coarse score is already the final score.
  return {distance, bits_ == 1 ? distance : lower};
}

float RabitqQuantizer::calc_distance_dp_query(const void *dp,
                                              const void *q) const {
  float lower;
  return estimate(dp, q, bits_ > 1, &lower);
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
  if (!data || !out || !ometa || !valid_input(meta) || require_train() ||
      !FiniteInput(data, dim_))
    return kErrInvalidArgument;
  out->resize(quantized_query_vector_length());
  quantize_query(data, out->data());
  *ometa = quantized_query_meta();
  return 0;
}

int RabitqQuantizer::quantize_datapoint(const void *data,
                                        const IndexQueryMeta &meta,
                                        std::string *out,
                                        IndexQueryMeta *ometa) const {
  if (!data || !out || !ometa || !valid_input(meta) || require_train() ||
      !FiniteInput(data, dim_))
    return kErrInvalidArgument;
  out->resize(quantized_datapoint_vector_length());
  quantize_data(data, out->data());
  ometa->set_meta(IndexMeta::DT_UINT8, static_cast<uint32_t>(out->size()),
                  static_cast<uint32_t>(type_), 0);
  return 0;
}

#if RABITQ_SUPPORTED
int RabitqQuantizer::dequantize(const void *data, const IndexQueryMeta &meta,
                                std::string *out) const {
  if (!data || !out ||
      meta.element_size() != quantized_datapoint_vector_length())
    return kErrInvalidArgument;
  const uint32_t cluster = ReadUint(data);
  if (cluster >= centroids_.size() / padded_dim_) return kErrInvalidArgument;
  const char *bin_data = static_cast<const char *>(data) + kBinPrefix;
  const char *ex_data = bin_data + padded_dim_ / 8 + 12;
  const size_t ex_bytes =
      bits_ > 1 ? static_cast<size_t>(padded_dim_) * (bits_ - 1) / 8 : 0;
  const float orig_norm = ReadFloat(ex_data, ex_bytes + 8);
  const float scale = ReadFloat(ex_data, ex_bytes + 12);
  std::vector<float> rotated(padded_dim_), raw(padded_dim_);
  const float midpoint = ((1u << bits_) - 1) * 0.5f;
  const auto *words = reinterpret_cast<const uint64_t *>(bin_data);
  std::vector<uint8_t> extra(padded_dim_, 0);
  if (bits_ > 1) {
    // rabitqlib ships no unpacker for its SIMD packing patterns; recover the
    // per-dimension extra codes with the dispatched ex-code inner product on
    // unit vectors. Dequantize is a cold path, never used during search.
    std::vector<float> unit(padded_dim_, 0);
    const auto *packed = reinterpret_cast<const uint8_t *>(ex_data);
    for (int d = 0; d < padded_dim_; ++d) {
      if (d > 0) unit[d - 1] = 0;
      unit[d] = 1;
      extra[d] = static_cast<uint8_t>(
          ex_ipfunc_(unit.data(), packed, static_cast<size_t>(padded_dim_)));
    }
  }
  const float *centroid = centroids_.data() + cluster * padded_dim_;
  for (int i = 0; i < padded_dim_; ++i) {
    const unsigned sign =
        static_cast<unsigned>((words[i / 64] >> (63 - i % 64)) & 1);
    rotated[i] =
        centroid[i] + scale * (((sign << (bits_ - 1)) | extra[i]) - midpoint);
  }
  rotator_->apply_inverse(rotated.data(), raw.data());
  if (metric_ == MetricType::kCosine)
    for (float &v : raw) v *= orig_norm;
  out->assign(reinterpret_cast<const char *>(raw.data()), dim_ * sizeof(float));
  return 0;
}
#else
int RabitqQuantizer::dequantize(const void *data, const IndexQueryMeta &meta,
                                std::string *out) const {
  if (!data || !out ||
      meta.element_size() != quantized_datapoint_vector_length())
    return kErrInvalidArgument;
  const uint32_t cluster = ReadUint(data);
  if (cluster >= centroids_.size() / padded_dim_) return kErrInvalidArgument;
  std::vector<float> rotated(padded_dim_), raw(padded_dim_);
  const float midpoint = ((1u << bits_) - 1) * 0.5f;
  const float scale = ReadFloat(data, 3 * sizeof(float));
  for (int i = 0; i < padded_dim_; ++i)
    rotated[i] = centroids_[cluster * padded_dim_ + i] +
                 scale * (code(data, i) - midpoint);
  rotator_->apply_inverse(rotated.data(), raw.data());
  if (metric_ == MetricType::kCosine)
    for (float &v : raw) v *= ReadFloat(data, sizeof(float));
  out->assign(reinterpret_cast<const char *>(raw.data()), dim_ * sizeof(float));
  return 0;
}
#endif

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
  const uint32_t fields[] = {
      kSplitFormat, static_cast<uint32_t>(bits_), num_clusters_,
      static_cast<uint32_t>(centroids_.size() / padded_dim_),
      static_cast<uint32_t>(rotation.size())};
  header.payload_size =
      sizeof(fields) + rotation.size() + centroids_.size() * sizeof(float);
  out->assign(reinterpret_cast<const char *>(&header), sizeof(header));
  out->append(reinterpret_cast<const char *>(fields), sizeof(fields));
  out->append(rotation);
  if (!centroids_.empty())
    out->append(reinterpret_cast<const char *>(centroids_.data()),
                centroids_.size() * sizeof(float));
  return 0;
}

int RabitqQuantizer::deserialize(const void *data, size_t len) {
  if (!data || !rotator_ ||
      len < sizeof(QuantizerSerHeader) + 5 * sizeof(uint32_t))
    return kErrInvalidArgument;
  QuantizerSerHeader header;
  std::memcpy(&header, data, sizeof(header));
  const char *payload = static_cast<const char *>(data) + sizeof(header);
  const uint32_t format = ReadUint(payload), bits = ReadUint(payload, 4);
  const uint32_t clusters = ReadUint(payload, 8),
                 trained = ReadUint(payload, 12);
  const uint32_t rotation_size = ReadUint(payload, 16);
  if (header.magic != kQuantizerMagic ||
      header.version != kQuantizerSerVersion ||
      header.quant_type != static_cast<uint16_t>(type_) ||
      header.dim != static_cast<uint32_t>(dim_) ||
      header.metric != static_cast<uint32_t>(metric_) ||
      header.data_type != static_cast<uint16_t>(DataType::kUint8) ||
      header.reserved != 0 || format != kSplitFormat ||
      bits != static_cast<uint32_t>(bits_) || clusters != num_clusters_ ||
      trained > clusters || header.payload_size != len - sizeof(header) ||
      rotation_size != sizeof(RotatorSerHeader) + 4 * ((padded_dim_ + 7) / 8) ||
      len - sizeof(header) !=
          20 + rotation_size +
              static_cast<size_t>(trained) * padded_dim_ * sizeof(float))
    return kErrInvalidArgument;
  auto rotation = FhtRotator::from_blob(payload + 20, rotation_size);
  if (!rotation || rotation->in_dim() != padded_dim_)
    return kErrInvalidArgument;
  std::vector<float> centroids(static_cast<size_t>(trained) * padded_dim_);
  if (!centroids.empty())
    std::memcpy(centroids.data(), payload + 20 + rotation_size,
                centroids.size() * sizeof(float));
  if (!FiniteInput(centroids.data(), centroids.size()))
    return kErrInvalidArgument;
  rotator_ = std::move(rotation);
  centroids_ = std::move(centroids);
  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(RabitqQuantizer);
}  // namespace zvec::turbo
