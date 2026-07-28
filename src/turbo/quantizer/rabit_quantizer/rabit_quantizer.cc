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

#include "quantizer/rabit_quantizer/rabit_quantizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_features.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_memory.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_stats.h>
#include <zvec/core/framework/index_threads.h>
#include "ailego/pattern/defer.h"

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#endif

#if RABITQ_SUPPORTED

#include <rabitqlib/defines.hpp>
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <rabitqlib/utils/space.hpp>
#include <rabitqlib/utils/warmup_space.hpp>
#include "core/algorithm/hnsw_rabitq/rabitq_params.h"
#include "core/algorithm/hnsw_rabitq/rabitq_utils.h"

namespace zvec {
namespace turbo {

// ---------------------------------------------------------------------------
// Quantized vector layout helpers
// ---------------------------------------------------------------------------

// Datapoint layout: [4B cluster_id][bin_data][ex_data]
static inline uint32_t dp_cluster_id(const void *dp) {
  return *reinterpret_cast<const uint32_t *>(dp);
}

static inline const char *dp_bin_data(const void *dp) {
  return reinterpret_cast<const char *>(dp) + sizeof(uint32_t);
}

static inline const char *dp_ex_data(const void *dp, size_t bin_data_size) {
  return reinterpret_cast<const char *>(dp) + sizeof(uint32_t) + bin_data_size;
}

// Query layout:
// [rotated_query][query_bin][delta|vl|k1xsumq|kbxsumq][q_to_centroids]
struct QueryLayout {
  const float *rotated_query;
  const uint64_t *query_bin;
  const float *delta;
  const float *vl;
  const float *k1xsumq;
  const float *kbxsumq;
  const float *q_to_centroids;

  QueryLayout(const void *query, size_t padded_dim, size_t query_bin_count,
              size_t q_to_centroids_size) {
    const char *p = reinterpret_cast<const char *>(query);
    rotated_query = reinterpret_cast<const float *>(p);
    p += padded_dim * sizeof(float);
    query_bin = reinterpret_cast<const uint64_t *>(p);
    p += query_bin_count * sizeof(uint64_t);
    delta = reinterpret_cast<const float *>(p);
    vl = reinterpret_cast<const float *>(p + sizeof(float));
    k1xsumq = reinterpret_cast<const float *>(p + 2 * sizeof(float));
    kbxsumq = reinterpret_cast<const float *>(p + 3 * sizeof(float));
    q_to_centroids = reinterpret_cast<const float *>(p + 4 * sizeof(float));
    (void)q_to_centroids_size;  // not used here, sizes are in impl
  }
};

// ---------------------------------------------------------------------------
// Pimpl implementation
// ---------------------------------------------------------------------------

struct RabitQuantizer::Impl {
  // RaBitQ parameters
  size_t dimension{0};
  size_t padded_dim{0};
  size_t num_clusters{0};
  size_t ex_bits{0};
  size_t total_bits{0};
  size_t sample_count{0};
  size_t size_bin_data{0};
  size_t size_ex_data{0};
  bool initialized{false};
  bool trained{false};

  // rabitqlib types
  rabitqlib::RotatorType rotator_type{rabitqlib::RotatorType::FhtKacRotator};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  rabitqlib::quant::RabitqConfig query_config;
  rabitqlib::quant::RabitqConfig config;
  rabitqlib::MetricType metric_type{rabitqlib::METRIC_L2};
  rabitqlib::ex_ipfunc ip_func{nullptr};

  // Centroids: original (for centroid search) and rotated (for quantization)
  std::vector<float> centroids;          // num_clusters * dimension
  std::vector<float> rotated_centroids;  // num_clusters * padded_dim

  // Index metadata
  IndexMeta meta;

  // Derived sizes
  size_t query_bin_count{0};      // number of uint64_t values for query bin
  size_t q_to_centroids_size{0};  // num_clusters (L2) or num_clusters*2 (IP)

  void compute_sizes() {
    size_bin_data = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
    size_ex_data = rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
    query_bin_count =
        padded_dim * rabitqlib::SplitSingleQuery<float>::kNumBits / 64;
    q_to_centroids_size =
        (metric_type == rabitqlib::METRIC_IP) ? num_clusters * 2 : num_clusters;
  }

  size_t quantized_dp_length() const {
    return sizeof(uint32_t) + size_bin_data + size_ex_data;
  }

  size_t quantized_query_length() const {
    return padded_dim * sizeof(float) +          // rotated_query
           query_bin_count * sizeof(uint64_t) +  // query_bin
           4 * sizeof(float) +                   // delta, vl, k1xsumq, kbxsumq
           q_to_centroids_size * sizeof(float);  // q_to_centroids
  }

  int train_with_sampler(
      std::shared_ptr<SampleIndexFeatures<CompactIndexFeatures>> sampler);
};

// ---------------------------------------------------------------------------
// Brute-force nearest centroid search (avoids core_knn_cluster dependency)
// ---------------------------------------------------------------------------
static uint32_t find_nearest_centroid(const float *query, size_t dim,
                                      const float *centroids,
                                      size_t num_clusters,
                                      rabitqlib::MetricType metric_type) {
  float best_score = std::numeric_limits<float>::max();
  uint32_t best_id = 0;
  for (uint32_t i = 0; i < num_clusters; ++i) {
    const float *cent = centroids + i * dim;
    float score = 0.0f;
    if (metric_type == rabitqlib::METRIC_L2) {
      for (size_t d = 0; d < dim; ++d) {
        float diff = query[d] - cent[d];
        score += diff * diff;
      }
    } else {
      // InnerProduct: negate so that smaller is better
      for (size_t d = 0; d < dim; ++d) {
        score -= query[d] * cent[d];
      }
    }
    if (score < best_score) {
      best_score = score;
      best_id = i;
    }
  }
  return best_id;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

RabitQuantizer::RabitQuantizer() : impl_(std::make_unique<Impl>()) {
  type_ = QuantizeType::kRabit;
}

RabitQuantizer::~RabitQuantizer() = default;

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int RabitQuantizer::init(const IndexMeta &meta, const ailego::Params &params) {
  impl_->meta = meta;
  impl_->dimension = meta.dimension();

  if (meta.metric_name().empty()) {
    LOG_ERROR("Meta metric is empty");
    return kErrUnsupported;
  }

  // Map metric name to rabitqlib metric type
  std::string metric_name = meta.metric_name();
  if (metric_name == "SquaredEuclidean") {
    impl_->metric_type = rabitqlib::METRIC_L2;
  } else if (metric_name == "InnerProduct" || metric_name == "Cosine") {
    impl_->metric_type = rabitqlib::METRIC_IP;
  } else {
    LOG_ERROR("Unsupported metric: %s", metric_name.c_str());
    return kErrUnsupported;
  }

  // Round up dimension to multiple of 64
  impl_->padded_dim = ((impl_->dimension + 63) / 64) * 64;

  // Parse total_bits (default 7, range [1, 9])
  uint32_t total_bits = 0;
  params.get(PARAM_RABITQ_TOTAL_BITS, &total_bits);
  if (total_bits == 0) {
    total_bits = static_cast<uint32_t>(kDefaultRabitqTotalBits);
  }
  if (total_bits < 1 || total_bits > 9) {
    LOG_ERROR("Invalid total_bits: %u, must be in [1, 9]", total_bits);
    return kErrUnsupported;
  }
  impl_->total_bits = total_bits;
  impl_->ex_bits = total_bits - 1;

  // Parse num_clusters (default 16)
  impl_->num_clusters = 0;
  params.get(PARAM_RABITQ_NUM_CLUSTERS,
             reinterpret_cast<uint32_t *>(&impl_->num_clusters));
  if (impl_->num_clusters == 0) {
    impl_->num_clusters = kDefaultNumClusters;
  }

  // Parse sample_count (0 = all)
  impl_->sample_count = 0;
  params.get(PARAM_RABITQ_SAMPLE_COUNT,
             reinterpret_cast<uint32_t *>(&impl_->sample_count));

  // Parse rotator type
  std::string rotator_type_str;
  params.get(PARAM_RABITQ_ROTATOR_TYPE, &rotator_type_str);
  if (rotator_type_str.empty()) {
    impl_->rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  } else if (strncasecmp(rotator_type_str.c_str(), "fht", 3) == 0) {
    impl_->rotator_type = rabitqlib::RotatorType::FhtKacRotator;
  } else if (strncasecmp(rotator_type_str.c_str(), "matrix", 6) == 0) {
    impl_->rotator_type = rabitqlib::RotatorType::MatrixRotator;
  } else {
    LOG_ERROR("Invalid rotator_type: %s", rotator_type_str.c_str());
    return kErrUnsupported;
  }

  // Create rotator
  impl_->rotator.reset(rabitqlib::choose_rotator<float>(
      impl_->dimension, impl_->rotator_type, impl_->padded_dim));

  // Set up configs
  impl_->query_config = rabitqlib::quant::faster_config(
      impl_->padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits);
  impl_->config =
      rabitqlib::quant::faster_config(impl_->padded_dim, impl_->ex_bits + 1);

  // Select excode ip function
  impl_->ip_func = rabitqlib::select_excode_ipfunc(impl_->ex_bits);

  // Compute derived sizes
  impl_->compute_sizes();

  impl_->initialized = true;

  LOG_INFO(
      "RabitQuantizer initialized: dim=%zu, padded_dim=%zu, "
      "num_clusters=%zu, ex_bits=%zu, total_bits=%zu, rotator_type=%d[%s], "
      "sample_count=%zu, size_bin_data=%zu, size_ex_data=%zu",
      impl_->dimension, impl_->padded_dim, impl_->num_clusters, impl_->ex_bits,
      impl_->total_bits, (int)impl_->rotator_type, rotator_type_str.c_str(),
      impl_->sample_count, impl_->size_bin_data, impl_->size_ex_data);

  return 0;
}

const IndexMeta &RabitQuantizer::meta() const {
  return impl_->meta;
}

DataType RabitQuantizer::input_data_type() const {
  return DataType::kFp32;
}

QuantizeType RabitQuantizer::type() const {
  return type_;
}

int RabitQuantizer::dim() const {
  return static_cast<int>(impl_->dimension);
}

bool RabitQuantizer::require_train() const {
  return true;
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

int RabitQuantizer::train(const void *data, size_t num, size_t stride) {
  if (!impl_->initialized) {
    LOG_ERROR("RabitQuantizer not initialized");
    return kErrNotImplemented;
  }
  if (!data || num == 0) {
    LOG_ERROR("Invalid training data");
    return kErrNotImplemented;
  }

  // Sample data from raw buffer
  size_t sample_count = num;
  if (impl_->sample_count > 0) {
    sample_count = std::min(impl_->sample_count, num);
  }

  auto sampler = std::make_shared<SampleIndexFeatures<CompactIndexFeatures>>(
      impl_->meta, sample_count);

  const uint8_t *ptr = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < num; ++i) {
    sampler->emplace(ptr);
    ptr += stride;
  }

  if (sampler->count() == 0) {
    LOG_ERROR("No samples loaded for training");
    return kErrNotImplemented;
  }

  return impl_->train_with_sampler(sampler);
}

int RabitQuantizer::train(IndexHolder::Pointer holder) {
  if (!impl_->initialized) {
    LOG_ERROR("RabitQuantizer not initialized");
    return kErrNotImplemented;
  }
  if (!holder) {
    LOG_ERROR("Null holder for training");
    return kErrNotImplemented;
  }

  size_t vector_count = holder->count();
  if (vector_count == 0) {
    LOG_ERROR("No vectors for training");
    return kErrNotImplemented;
  }

  size_t sample_count = vector_count;
  if (impl_->sample_count > 0) {
    sample_count = std::min(impl_->sample_count, vector_count);
  }

  LOG_INFO("Training RaBitQ with %zu vectors from %zu", sample_count,
           vector_count);

  auto sampler = std::make_shared<SampleIndexFeatures<CompactIndexFeatures>>(
      impl_->meta, sample_count);
  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Failed to create iterator");
    return kErrNotImplemented;
  }
  for (; iter->is_valid(); iter->next()) {
    sampler->emplace(iter->data());
  }
  holder.reset();

  if (sampler->count() == 0) {
    LOG_ERROR("No samples loaded for training");
    return kErrNotImplemented;
  }

  return impl_->train_with_sampler(sampler);
}

int RabitQuantizer::Impl::train_with_sampler(
    std::shared_ptr<SampleIndexFeatures<CompactIndexFeatures>> sampler) {
  ailego::ElapsedTime timer;

  // Create KmeansCluster for training centroids
  auto cluster = IndexFactory::CreateCluster("OptKmeansCluster");
  if (!cluster) {
    LOG_ERROR("Failed to create OptKmeansCluster");
    return kErrNotImplemented;
  }

  ailego::Params cluster_params;
  int ret = cluster->init(meta, cluster_params);
  if (ret != 0) {
    LOG_ERROR("Failed to init KmeansCluster: %d", ret);
    return ret;
  }

  ret = cluster->mount(sampler);
  if (ret != 0) {
    LOG_ERROR("Failed to mount training data: %d", ret);
    return ret;
  }
  cluster->suggest(num_clusters);

  IndexCluster::CentroidList cents;
  auto threads = std::make_shared<SingleQueueIndexThreads>(0, false);
  ret = cluster->cluster(threads, cents);
  if (ret != 0) {
    LOG_ERROR("Failed to perform clustering: %d", ret);
    return ret;
  }

  if (cents.size() != num_clusters) {
    LOG_WARN("Expected %zu clusters, got %zu", num_clusters, cents.size());
    num_clusters = cents.size();
    compute_sizes();
  }

  // Extract original and rotated centroids
  centroids.resize(num_clusters * dimension);
  rotated_centroids.resize(num_clusters * padded_dim);
  for (uint32_t i = 0; i < num_clusters; ++i) {
    const float *cent_data = static_cast<const float *>(cents[i].feature());
    std::memcpy(&centroids[i * dimension], cent_data,
                dimension * sizeof(float));
    rotator->rotate(cent_data, &rotated_centroids[i * padded_dim]);
  }

  trained = true;

  LOG_INFO("RaBitQ training completed: %zu centroids, cost %zu ms",
           num_clusters, static_cast<size_t>(timer.milli_seconds()));

  return 0;
}

// ---------------------------------------------------------------------------
// quantize_data
// ---------------------------------------------------------------------------

size_t RabitQuantizer::quantized_datapoint_vector_length() const {
  return impl_->quantized_dp_length();
}

void RabitQuantizer::quantize_data(const void *input, void *output) const {
  const float *raw_vector = static_cast<const float *>(input);

  // Rotate the input vector
  std::vector<float> rotated_data(impl_->padded_dim);
  impl_->rotator->rotate(raw_vector, rotated_data.data());

  // Find nearest centroid by brute-force search
  uint32_t cluster_id = find_nearest_centroid(
      raw_vector, impl_->dimension, impl_->centroids.data(),
      impl_->num_clusters, impl_->metric_type);

  // Layout: [4B cluster_id][bin_data][ex_data]
  std::memcpy(output, &cluster_id, sizeof(cluster_id));
  char *bin_data_ptr = reinterpret_cast<char *>(output) + sizeof(cluster_id);
  char *ex_data_ptr = bin_data_ptr + impl_->size_bin_data;

  rabitqlib::quant::quantize_split_single(
      rotated_data.data(),
      impl_->rotated_centroids.data() + (cluster_id * impl_->padded_dim),
      impl_->padded_dim, impl_->ex_bits, bin_data_ptr, ex_data_ptr,
      impl_->metric_type, impl_->config);
}

// ---------------------------------------------------------------------------
// quantize_query
// ---------------------------------------------------------------------------

size_t RabitQuantizer::quantized_query_vector_length() const {
  return impl_->quantized_query_length();
}

void RabitQuantizer::quantize_query(const void *input, void *output) const {
  const float *query_vector = static_cast<const float *>(input);

  // Layout:
  // [rotated_query][query_bin][delta|vl|k1xsumq|kbxsumq][q_to_centroids]
  char *p = reinterpret_cast<char *>(output);

  // 1. Rotate query and store in output
  float *rotated_query = reinterpret_cast<float *>(p);
  impl_->rotator->rotate(query_vector, rotated_query);
  p += impl_->padded_dim * sizeof(float);

  // 2. Create SplitSingleQuery (quantizes to 4-bit)
  rabitqlib::SplitSingleQuery<float> query_obj(
      rotated_query, impl_->padded_dim, impl_->ex_bits, impl_->query_config,
      impl_->metric_type);

  // 3. Copy query_bin to output
  uint64_t *query_bin = reinterpret_cast<uint64_t *>(p);
  std::memcpy(query_bin, query_obj.query_bin(),
              impl_->query_bin_count * sizeof(uint64_t));
  p += impl_->query_bin_count * sizeof(uint64_t);

  // 4. Copy delta, vl, k1xsumq, kbxsumq
  float *delta = reinterpret_cast<float *>(p);
  *delta = query_obj.delta();
  float *vl = reinterpret_cast<float *>(p + sizeof(float));
  *vl = query_obj.vl();
  float *k1xsumq = reinterpret_cast<float *>(p + 2 * sizeof(float));
  *k1xsumq = query_obj.k1xsumq();
  float *kbxsumq = reinterpret_cast<float *>(p + 3 * sizeof(float));
  *kbxsumq = query_obj.kbxsumq();
  p += 4 * sizeof(float);

  // 5. Compute q_to_centroids
  float *q_to_centroids = reinterpret_cast<float *>(p);
  if (impl_->metric_type == rabitqlib::METRIC_L2) {
    for (size_t i = 0; i < impl_->num_clusters; ++i) {
      q_to_centroids[i] = std::sqrt(rabitqlib::euclidean_sqr(
          rotated_query,
          impl_->rotated_centroids.data() + (i * impl_->padded_dim),
          impl_->padded_dim));
    }
  } else {
    // IP: first half = dot_product, second half = euclidean distance
    for (size_t i = 0; i < impl_->num_clusters; ++i) {
      q_to_centroids[i] = rabitqlib::dot_product(
          rotated_query,
          impl_->rotated_centroids.data() + (i * impl_->padded_dim),
          impl_->padded_dim);
      q_to_centroids[i + impl_->num_clusters] =
          std::sqrt(rabitqlib::euclidean_sqr(
              rotated_query,
              impl_->rotated_centroids.data() + (i * impl_->padded_dim),
              impl_->padded_dim));
    }
  }
}

// ---------------------------------------------------------------------------
// calc_distance_dp_query
// ---------------------------------------------------------------------------

float RabitQuantizer::calc_distance_dp_query(const void *dp,
                                             const void *query) const {
  // Parse datapoint
  uint32_t cluster_id = dp_cluster_id(dp);
  const char *bin_data = dp_bin_data(dp);
  const char *ex_data = dp_ex_data(dp, impl_->size_bin_data);

  // Parse query
  QueryLayout ql(query, impl_->padded_dim, impl_->query_bin_count,
                 impl_->q_to_centroids_size);

  // Compute g_add from q_to_centroids
  float g_add;
  if (impl_->metric_type == rabitqlib::METRIC_L2) {
    float norm = ql.q_to_centroids[cluster_id];
    g_add = norm * norm;
  } else {
    // IP
    float ip = ql.q_to_centroids[cluster_id];
    g_add = -ip;
  }

  float est_dist;

  if (impl_->ex_bits > 0) {
    // Full estimation using ex-bits
    rabitqlib::ConstBinDataMap<float> cur_bin(bin_data, impl_->padded_dim);
    rabitqlib::ConstExDataMap<float> cur_ex(ex_data, impl_->padded_dim,
                                            impl_->ex_bits);

    float ip_x0_qr = rabitqlib::mask_ip_x0_q(
        ql.rotated_query, cur_bin.bin_code(), impl_->padded_dim);

    est_dist = cur_ex.f_add_ex() + g_add +
               (cur_ex.f_rescale_ex() *
                (static_cast<float>(1 << impl_->ex_bits) * ip_x0_qr +
                 impl_->ip_func(ql.rotated_query, cur_ex.ex_code(),
                                impl_->padded_dim) +
                 *ql.kbxsumq));
  } else {
    // 1-bit estimation only
    rabitqlib::ConstBinDataMap<float> cur_bin(bin_data, impl_->padded_dim);

    float ip_x0_qr =
        ::warmup_ip_x0_q<rabitqlib::SplitSingleQuery<float>::kNumBits>(
            cur_bin.bin_code(), ql.query_bin, *ql.delta, *ql.vl,
            impl_->padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits);

    est_dist = cur_bin.f_add() + g_add +
               (cur_bin.f_rescale() * (ip_x0_qr + *ql.k1xsumq));
  }

  return est_dist;
}

void RabitQuantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                  int dp_num, const void *query,
                                                  float *dist_list) const {
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = calc_distance_dp_query(dp_list[i], query);
  }
}

float RabitQuantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::string buf(quantized_query_vector_length(), '\0');
  quantize_query(query, &buf[0]);
  return calc_distance_dp_query(dp, buf.data());
}

void RabitQuantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::string buf(quantized_query_vector_length(), '\0');
  quantize_query(query, &buf[0]);
  calc_distance_dp_query_batch(dp_list, dp_num, buf.data(), dist_list);
}

float RabitQuantizer::calc_distance_dp_dp(const void *dp1,
                                          const void *dp2) const {
  // RaBitQ estimates distances between datapoints and queries.
  // dp-dp distance is not directly supported; fall back to treating dp2
  // as a query (only valid if dp2 was produced by quantize_query).
  return calc_distance_dp_query(dp1, dp2);
}

// ---------------------------------------------------------------------------
// serialize / deserialize
// ---------------------------------------------------------------------------

int RabitQuantizer::serialize(std::string *out) const {
  if (!impl_->trained) {
    LOG_ERROR("Quantizer not trained, nothing to serialize");
    return kErrNotImplemented;
  }

  // Build payload: RabitqConverterHeader + rotated_centroids + centroids +
  // rotator
  size_t rotator_size = impl_->rotator->dump_bytes();
  size_t rotated_centroids_size =
      impl_->rotated_centroids.size() * sizeof(float);
  size_t centroids_size = impl_->centroids.size() * sizeof(float);
  size_t payload_size = sizeof(RabitqConverterHeader) + rotated_centroids_size +
                        centroids_size + rotator_size;

  out->resize(sizeof(QuantizerSerHeader) + payload_size);
  char *buf = &(*out)[0];

  // Write QuantizerSerHeader
  QuantizerSerHeader *qhdr = reinterpret_cast<QuantizerSerHeader *>(buf);
  qhdr->magic = kQuantizerMagic;
  qhdr->version = kQuantizerSerVersion;
  qhdr->quant_type = static_cast<uint16_t>(QuantizeType::kRabit);
  qhdr->dim = static_cast<uint32_t>(impl_->dimension);
  qhdr->metric = static_cast<uint32_t>(impl_->metric_type);
  qhdr->payload_size = static_cast<uint32_t>(payload_size);
  qhdr->reserved = 0;
  buf += sizeof(QuantizerSerHeader);

  // Write RabitqConverterHeader
  RabitqConverterHeader rheader;
  rheader.num_clusters = static_cast<uint32_t>(impl_->num_clusters);
  rheader.dim = static_cast<uint32_t>(impl_->dimension);
  rheader.padded_dim = static_cast<uint32_t>(impl_->padded_dim);
  rheader.rotator_size = static_cast<uint32_t>(rotator_size);
  rheader.ex_bits = static_cast<uint8_t>(impl_->ex_bits);
  rheader.rotator_type = static_cast<uint8_t>(impl_->rotator_type);
  std::memcpy(buf, &rheader, sizeof(rheader));
  buf += sizeof(RabitqConverterHeader);

  // Write rotated centroids
  std::memcpy(buf, impl_->rotated_centroids.data(), rotated_centroids_size);
  buf += rotated_centroids_size;

  // Write original centroids
  std::memcpy(buf, impl_->centroids.data(), centroids_size);
  buf += centroids_size;

  // Write rotator
  impl_->rotator->save(buf);

  return 0;
}

int RabitQuantizer::deserialize(const void *data, size_t len) {
  if (!data || len < sizeof(QuantizerSerHeader)) {
    LOG_ERROR("Invalid deserialize data");
    return kErrNotImplemented;
  }

  const char *buf = static_cast<const char *>(data);

  // Read QuantizerSerHeader
  QuantizerSerHeader qhdr;
  std::memcpy(&qhdr, buf, sizeof(qhdr));
  if (qhdr.magic != kQuantizerMagic) {
    LOG_ERROR("Invalid quantizer magic: 0x%x", qhdr.magic);
    return kErrNotImplemented;
  }
  if (qhdr.quant_type != static_cast<uint16_t>(QuantizeType::kRabit)) {
    LOG_ERROR("Quantizer type mismatch: %u", qhdr.quant_type);
    return kErrNotImplemented;
  }
  buf += sizeof(QuantizerSerHeader);

  if (len < sizeof(QuantizerSerHeader) + qhdr.payload_size) {
    LOG_ERROR("Payload size mismatch");
    return kErrNotImplemented;
  }

  // Read RabitqConverterHeader
  RabitqConverterHeader rheader;
  std::memcpy(&rheader, buf, sizeof(rheader));
  buf += sizeof(RabitqConverterHeader);

  impl_->num_clusters = rheader.num_clusters;
  impl_->dimension = rheader.dim;
  impl_->padded_dim = rheader.padded_dim;
  impl_->ex_bits = rheader.ex_bits;
  impl_->total_bits = impl_->ex_bits + 1;
  impl_->rotator_type =
      static_cast<rabitqlib::RotatorType>(rheader.rotator_type);
  impl_->metric_type = static_cast<rabitqlib::MetricType>(qhdr.metric);

  // Read rotated centroids
  size_t rotated_centroids_size =
      impl_->num_clusters * impl_->padded_dim * sizeof(float);
  impl_->rotated_centroids.resize(impl_->num_clusters * impl_->padded_dim);
  std::memcpy(impl_->rotated_centroids.data(), buf, rotated_centroids_size);
  buf += rotated_centroids_size;

  // Read original centroids
  size_t centroids_size =
      impl_->num_clusters * impl_->dimension * sizeof(float);
  impl_->centroids.resize(impl_->num_clusters * impl_->dimension);
  std::memcpy(impl_->centroids.data(), buf, centroids_size);
  buf += centroids_size;

  // Read rotator
  impl_->rotator.reset(rabitqlib::choose_rotator<float>(
      impl_->dimension, impl_->rotator_type, impl_->padded_dim));
  impl_->rotator->load(buf);

  // Recompute configs and sizes
  impl_->query_config = rabitqlib::quant::faster_config(
      impl_->padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits);
  impl_->config =
      rabitqlib::quant::faster_config(impl_->padded_dim, impl_->ex_bits + 1);
  impl_->ip_func = rabitqlib::select_excode_ipfunc(impl_->ex_bits);
  impl_->compute_sizes();

  impl_->initialized = true;
  impl_->trained = true;

  LOG_INFO(
      "RabitQuantizer deserialized: dim=%zu, padded_dim=%zu, "
      "num_clusters=%zu, ex_bits=%zu",
      impl_->dimension, impl_->padded_dim, impl_->num_clusters, impl_->ex_bits);

  return 0;
}

int RabitQuantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

INDEX_FACTORY_REGISTER_QUANTIZER(RabitQuantizer);

}  // namespace turbo
}  // namespace zvec

#else  // !RABITQ_SUPPORTED

namespace zvec {
namespace turbo {

// Empty Impl for stub (non-AVX2 platforms)
struct RabitQuantizer::Impl {};

// ---------------------------------------------------------------------------
// Stub implementations when RaBitQ is not supported (non-AVX2 platforms)
// ---------------------------------------------------------------------------

RabitQuantizer::RabitQuantizer() : impl_(std::make_unique<Impl>()) {
  type_ = QuantizeType::kRabit;
}

RabitQuantizer::~RabitQuantizer() = default;

int RabitQuantizer::init(const IndexMeta &, const ailego::Params &) {
  LOG_ERROR("RaBitQ not supported on this platform");
  return kErrUnsupported;
}

const IndexMeta &RabitQuantizer::meta() const {
  static IndexMeta dummy;
  return dummy;
}

DataType RabitQuantizer::input_data_type() const {
  return DataType::kFp32;
}

QuantizeType RabitQuantizer::type() const {
  return type_;
}

int RabitQuantizer::dim() const {
  return 0;
}

bool RabitQuantizer::require_train() const {
  return true;
}

int RabitQuantizer::train(const void *, size_t, size_t) {
  return kErrUnsupported;
}

int RabitQuantizer::train(IndexHolder::Pointer) {
  return kErrUnsupported;
}

size_t RabitQuantizer::quantized_datapoint_vector_length() const {
  return 0;
}

size_t RabitQuantizer::quantized_query_vector_length() const {
  return 0;
}

void RabitQuantizer::quantize_data(const void *, void *) const {}

void RabitQuantizer::quantize_query(const void *, void *) const {}

float RabitQuantizer::calc_distance_dp_query(const void *, const void *) const {
  return 0.0f;
}

void RabitQuantizer::calc_distance_dp_query_batch(const void *const *, int,
                                                  const void *, float *) const {
}

float RabitQuantizer::calc_distance_dp_query_unquantized(const void *,
                                                         const void *) const {
  return 0.0f;
}

void RabitQuantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *, int, const void *, float *) const {}

float RabitQuantizer::calc_distance_dp_dp(const void *, const void *) const {
  return 0.0f;
}

int RabitQuantizer::serialize(std::string *) const {
  return kErrUnsupported;
}

int RabitQuantizer::deserialize(std::string &) {
  return kErrUnsupported;
}

int RabitQuantizer::deserialize(const void *, size_t) {
  return kErrUnsupported;
}

INDEX_FACTORY_REGISTER_QUANTIZER(RabitQuantizer);

}  // namespace turbo
}  // namespace zvec

#endif  // RABITQ_SUPPORTED
