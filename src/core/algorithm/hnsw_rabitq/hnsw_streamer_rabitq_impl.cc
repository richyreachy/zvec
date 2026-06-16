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
//
// Implementation of HnswStreamer's RaBitQ operations.
// This file is compiled as part of core_knn_hnsw_rabitq (with AVX2 flags)
// and does NOT include hnsw_entity.h, avoiding the type conflict.
#include "../hnsw/hnsw_streamer_rabitq_impl.h"
#include <ailego/pattern/defer.h>
#include "hnsw_rabitq_algorithm.h"
#include "hnsw_rabitq_context.h"
#include "hnsw_rabitq_index_provider.h"
#include "hnsw_rabitq_query_algorithm.h"
#include "hnsw_rabitq_query_entity.h"
#include "hnsw_rabitq_streamer_entity.h"
#include "rabitq_params.h"
#include "rabitq_reformer.h"

static const std::string kRabitqConverterSegId{"rabitq.converter"};

namespace zvec {
namespace core {

// ---------------------------------------------------------------------------
// State definition
// ---------------------------------------------------------------------------
struct HnswStreamerRabitqState {
  std::unique_ptr<HnswRabitqStreamerEntity> entity;
  std::unique_ptr<HnswRabitqAlgorithm> alg;
  std::unique_ptr<HnswRabitqQueryAlgorithm> query_alg;
  IndexMetric::Pointer metric;
  IndexMetric::MatrixDistance add_distance{};
  IndexMetric::MatrixDistance search_distance{};
  IndexMetric::MatrixBatchDistance add_batch_distance{};
  IndexMetric::MatrixBatchDistance search_batch_distance{};
};

void DestroyRabitqState(HnswStreamerRabitqState *p) {
  delete p;
}

// ---------------------------------------------------------------------------
// Helper: update context for staleness
// ---------------------------------------------------------------------------
static int UpdateContextInternal(const HnswStreamerRabitqState *state,
                                 const IndexMeta &meta,
                                 const IndexMetric::Pointer &metric,
                                 size_t max_scan_limit, size_t min_scan_limit,
                                 float max_scan_ratio,
                                 size_t bruteforce_threshold, uint32_t magic,
                                 HnswRabitqContext *ctx) {
  const HnswRabitqEntity::Pointer entity = state->entity->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }
  ctx->set_max_scan_limit(max_scan_limit);
  ctx->set_min_scan_limit(min_scan_limit);
  ctx->set_max_scan_ratio(max_scan_ratio);
  ctx->set_bruteforce_threshold(bruteforce_threshold);
  return ctx->update_context(HnswRabitqContext::kStreamerContext, meta, metric,
                             entity, magic);
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
RabitqStatePtr CreateRabitqState(
    IndexRunner::Stats &stats, bool use_id_map, uint8_t ex_bits,
    uint32_t dimension, uint32_t ef_construction,
    uint32_t upper_max_neighbor_cnt, uint32_t l0_max_neighbor_cnt,
    uint32_t scaling_factor, uint32_t prune_cnt, size_t chunk_size,
    bool filter_same_key, bool get_vector_enabled, uint32_t min_neighbor_cnt,
    size_t docs_hard_limit) {
  auto state = new (std::nothrow) HnswStreamerRabitqState();
  if (!state) {
    LOG_ERROR("Failed to allocate HnswStreamerRabitqState");
    return RabitqStatePtr(nullptr);
  }

  state->entity = std::make_unique<HnswRabitqStreamerEntity>(stats);
  state->entity->set_use_key_info_map(use_id_map);
  state->entity->set_ex_bits(ex_bits);
  state->entity->update_rabitq_params_and_vector_size(dimension);
  state->entity->set_ef_construction(ef_construction);
  state->entity->set_upper_neighbor_cnt(upper_max_neighbor_cnt);
  state->entity->set_l0_neighbor_cnt(l0_max_neighbor_cnt);
  state->entity->set_scaling_factor(scaling_factor);
  state->entity->set_prune_cnt(prune_cnt);
  state->entity->set_chunk_size(chunk_size);
  state->entity->set_filter_same_key(filter_same_key);
  state->entity->set_get_vector(get_vector_enabled);
  state->entity->set_min_neighbor_cnt(min_neighbor_cnt);

  int ret = state->entity->init(docs_hard_limit);
  if (ret != 0) {
    LOG_ERROR("RaBitQ entity init failed for %s", IndexError::What(ret));
    delete state;
    return RabitqStatePtr(nullptr);
  }

  state->alg =
      HnswRabitqAlgorithm::UPointer(new HnswRabitqAlgorithm(*state->entity));
  ret = state->alg->init();
  if (ret != 0) {
    LOG_ERROR("RaBitQ algorithm init failed, ret=%d", ret);
    delete state;
    return RabitqStatePtr(nullptr);
  }

  return RabitqStatePtr(state);
}

// ---------------------------------------------------------------------------
// Open
// ---------------------------------------------------------------------------
int RabitqStateOpen(HnswStreamerRabitqState *state,
                    const std::shared_ptr<RabitqReformer> &reformer,
                    IndexStorage::Pointer stg, size_t max_index_size,
                    bool check_crc_enabled, IndexMeta &meta) {
  // Handle reformer persistence
  if (!stg->has(kRabitqConverterSegId)) {
    int ret = reformer->dump(stg);
    if (ret != 0) {
      LOG_ERROR("Failed to dump reformer, ret=%d", ret);
      return ret;
    }
    LOG_INFO("Dump reformer success.");
  }

  int ret =
      state->entity->open(std::move(stg), max_index_size, check_crc_enabled);
  if (ret != 0) {
    return ret;
  }

  // Verify ex_bits consistency
  if (reformer->ex_bits() != state->entity->ex_bits()) {
    LOG_ERROR(
        "ex_bits mismatch between reformer(%zu) and entity(%zu). "
        "Reformer and entity must use the same total_bits configuration",
        reformer->ex_bits(), (size_t)state->entity->ex_bits());
    return IndexError_Mismatch;
  }

  IndexMeta index_meta;
  ret = state->entity->get_index_meta(&index_meta);
  if (ret == IndexError_NoExist) {
    ret = state->entity->set_index_meta(meta);
    if (ret != 0) {
      LOG_ERROR("Failed to set index meta for %s", IndexError::What(ret));
      return ret;
    }
  } else if (ret != 0) {
    LOG_ERROR("Failed to get index meta for %s", IndexError::What(ret));
    return ret;
  } else {
    if (index_meta.dimension() != meta.dimension() ||
        index_meta.element_size() != meta.element_size() ||
        index_meta.metric_name() != meta.metric_name() ||
        index_meta.data_type() != meta.data_type()) {
      LOG_ERROR("IndexMeta mismatch from the previous in index");
      return IndexError_Mismatch;
    }
    auto metric_params = index_meta.metric_params();
    metric_params.merge(meta.metric_params());
    meta.set_metric(index_meta.metric_name(), 0, metric_params);
  }

  state->metric = IndexFactory::CreateMetric(meta.metric_name());
  if (!state->metric) {
    LOG_ERROR("Failed to create metric %s", meta.metric_name().c_str());
    return IndexError_NoExist;
  }
  ret = state->metric->init(meta, meta.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failed to init metric, ret=%d", ret);
    return ret;
  }

  if (!state->metric->distance()) {
    LOG_ERROR("Invalid metric distance");
    return IndexError_InvalidArgument;
  }
  if (!state->metric->batch_distance()) {
    LOG_ERROR("Invalid metric batch distance");
    return IndexError_InvalidArgument;
  }

  state->add_distance = state->metric->distance();
  state->add_batch_distance = state->metric->batch_distance();
  state->search_distance = state->add_distance;
  state->search_batch_distance = state->add_batch_distance;

  if (state->metric->query_metric() &&
      state->metric->query_metric()->distance() &&
      state->metric->query_metric()->batch_distance()) {
    state->search_distance = state->metric->query_metric()->distance();
    state->search_batch_distance =
        state->metric->query_metric()->batch_distance();
  }

  // Create query algorithm for RaBitQ search
  state->query_alg = HnswRabitqQueryAlgorithm::UPointer(
      new HnswRabitqQueryAlgorithm(*state->entity, reformer->num_clusters(),
                                   reformer->rabitq_metric_type()));

  return 0;
}

// ---------------------------------------------------------------------------
// Close / Flush / Dump
// ---------------------------------------------------------------------------
int RabitqStateClose(HnswStreamerRabitqState *state, IndexMeta &meta,
                     const IndexMetric::Pointer &metric) {
  meta.set_metric(metric->name(), 0, metric->params());
  state->entity->set_index_meta(meta);
  return state->entity->close();
}

int RabitqStateFlush(HnswStreamerRabitqState *state, IndexMeta &meta,
                     const IndexMetric::Pointer &metric, uint64_t checkpoint) {
  meta.set_metric(metric->name(), 0, metric->params());
  state->entity->set_index_meta(meta);
  return state->entity->flush(checkpoint);
}

int RabitqStateDump(HnswStreamerRabitqState *state,
                    const std::shared_ptr<RabitqReformer> &reformer,
                    const IndexDumper::Pointer &dumper) {
  int ret = reformer->dump(dumper);
  if (ret != 0) {
    LOG_ERROR("Failed to dump reformer into dumper.");
    return ret;
  }
  return state->entity->dump(dumper);
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
IndexStreamer::Context::Pointer RabitqStateCreateContext(
    const HnswStreamerRabitqState *state, const IndexMeta &meta,
    const IndexMetric::Pointer &metric, uint32_t ef, size_t max_scan_limit,
    size_t min_scan_limit, float max_scan_ratio, bool bf_enabled,
    float bf_negative_prob, uint32_t magic, bool force_padding_topk_enabled,
    size_t bruteforce_threshold) {
  using Context = IndexStreamer::Context;

  HnswRabitqEntity::Pointer entity = state->entity->clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("CreateContext clone init failed");
    return Context::Pointer();
  }
  HnswRabitqContext *ctx =
      new (std::nothrow) HnswRabitqContext(meta.dimension(), metric, entity);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new HnswRabitqContext");
    return Context::Pointer();
  }
  ctx->set_ef(ef);
  ctx->set_max_scan_limit(max_scan_limit);
  ctx->set_min_scan_limit(min_scan_limit);
  ctx->set_max_scan_ratio(max_scan_ratio);
  ctx->set_filter_mode(bf_enabled ? VisitFilter::BloomFilter
                                  : VisitFilter::ByteMap);
  ctx->set_filter_negative_probability(bf_negative_prob);
  ctx->set_magic(magic);
  ctx->set_force_padding_topk(force_padding_topk_enabled);
  ctx->set_bruteforce_threshold(bruteforce_threshold);

  if (ailego_unlikely(ctx->init(HnswRabitqContext::kStreamerContext)) != 0) {
    LOG_ERROR("Init HnswRabitqContext failed");
    delete ctx;
    return Context::Pointer();
  }
  ctx->check_need_adjuct_ctx(state->entity->doc_cnt());
  return Context::Pointer(ctx);
}

int RabitqStateUpdateContext(const HnswStreamerRabitqState *state,
                             const IndexMeta &meta,
                             const IndexMetric::Pointer &metric,
                             size_t max_scan_limit, size_t min_scan_limit,
                             float max_scan_ratio, size_t bruteforce_threshold,
                             uint32_t magic,
                             IndexStreamer::Context::Pointer &context) {
  HnswRabitqContext *ctx = dynamic_cast<HnswRabitqContext *>(context.get());
  if (!ctx) {
    LOG_ERROR("Cast context to HnswRabitqContext failed");
    return IndexError_Cast;
  }
  return UpdateContextInternal(state, meta, metric, max_scan_limit,
                               min_scan_limit, max_scan_ratio,
                               bruteforce_threshold, magic, ctx);
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------
int RabitqStateAdd(
    HnswStreamerRabitqState *state,
    const std::shared_ptr<RabitqReformer> &reformer,
    const IndexProvider::Pointer &provider, const IndexMetric::Pointer &metric,
    const IndexMeta &meta, size_t docs_soft_limit, size_t docs_hard_limit,
    size_t max_scan_limit, size_t min_scan_limit, float max_scan_ratio,
    size_t bruteforce_threshold, uint32_t magic, std::mutex &mutex,
    ailego::SharedMutex &shared_mutex, std::atomic<size_t> *added_count,
    std::atomic<size_t> *discarded_count, uint64_t pkey, const void *query,
    const IndexQueryMeta &qmeta, IndexStreamer::Context::Pointer &context) {
  if (!provider) {
    LOG_ERROR("Provider is nullptr, cannot add vector in RaBitQ mode");
    return IndexError_InvalidArgument;
  }

  HnswRabitqContext *ctx = dynamic_cast<HnswRabitqContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswRabitqContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic) {
    int ret = UpdateContextInternal(state, meta, state->metric, max_scan_limit,
                                    min_scan_limit, max_scan_ratio,
                                    bruteforce_threshold, magic, ctx);
    if (ret != 0) {
      return ret;
    }
  }

  if (ailego_unlikely(state->entity->doc_cnt() >= docs_soft_limit)) {
    if (state->entity->doc_cnt() >= docs_hard_limit) {
      LOG_ERROR("Current docs %u exceed docs_hard_limit",
                state->entity->doc_cnt());
      (*discarded_count)++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed docs_soft_limit",
               state->entity->doc_cnt());
    }
  }
  if (ailego_unlikely(!shared_mutex.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*discarded_count)++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex.unlock_shared(); });

  ctx->clear();
  ctx->update_dist_caculator_distance(state->add_distance,
                                      state->add_batch_distance);
  ctx->reset_query(query);
  ctx->check_need_adjuct_ctx(state->entity->doc_cnt());
  ctx->set_provider(provider);

  int ret = 0;
  if (metric->support_train()) {
    const std::lock_guard<std::mutex> lk(mutex);
    ret = metric->train(query, meta.dimension());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw streamer metric train failed");
      (*discarded_count)++;
      return ret;
    }
  }

  std::string converted_vector;
  IndexQueryMeta converted_meta;
  ret = reformer->convert(query, qmeta, &converted_vector, &converted_meta);
  if (ret != 0) {
    LOG_ERROR("RaBitQ convert failed, ret=%d", ret);
    (*discarded_count)++;
    return ret;
  }

  level_t level = state->alg->get_random_level();
  node_id_t id;
  ret = state->entity->add_vector(level, pkey, converted_vector.data(), &id);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw streamer add vector failed");
    (*discarded_count)++;
    return ret;
  }

  ret = state->alg->add_node(id, level, ctx);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw streamer add node failed");
    (*discarded_count)++;
    return ret;
  }

  if (ailego_unlikely(ctx->error())) {
    (*discarded_count)++;
    return IndexError_Runtime;
  }
  (*added_count)++;
  return 0;
}

int RabitqStateAddWithId(
    HnswStreamerRabitqState *state,
    const std::shared_ptr<RabitqReformer> &reformer,
    const IndexProvider::Pointer &provider, const IndexMetric::Pointer &metric,
    const IndexMeta &meta, size_t docs_soft_limit, size_t docs_hard_limit,
    size_t max_scan_limit, size_t min_scan_limit, float max_scan_ratio,
    size_t bruteforce_threshold, uint32_t magic, std::mutex &mutex,
    ailego::SharedMutex &shared_mutex, std::atomic<size_t> *added_count,
    std::atomic<size_t> *discarded_count, uint32_t id, const void *query,
    const IndexQueryMeta &qmeta, IndexStreamer::Context::Pointer &context) {
  if (!provider) {
    LOG_ERROR("Provider is nullptr, cannot add vector in RaBitQ mode");
    return IndexError_InvalidArgument;
  }

  HnswRabitqContext *ctx = dynamic_cast<HnswRabitqContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswRabitqContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic) {
    int ret = UpdateContextInternal(state, meta, state->metric, max_scan_limit,
                                    min_scan_limit, max_scan_ratio,
                                    bruteforce_threshold, magic, ctx);
    if (ret != 0) {
      return ret;
    }
  }

  if (ailego_unlikely(state->entity->doc_cnt() >= docs_soft_limit)) {
    if (state->entity->doc_cnt() >= docs_hard_limit) {
      LOG_ERROR("Current docs %u exceed docs_hard_limit",
                state->entity->doc_cnt());
      (*discarded_count)++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed docs_soft_limit",
               state->entity->doc_cnt());
    }
  }
  if (ailego_unlikely(!shared_mutex.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*discarded_count)++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex.unlock_shared(); });

  ctx->clear();
  ctx->update_dist_caculator_distance(state->add_distance,
                                      state->add_batch_distance);
  ctx->reset_query(query);
  ctx->check_need_adjuct_ctx(state->entity->doc_cnt());
  ctx->set_provider(provider);

  int ret = 0;
  if (metric->support_train()) {
    const std::lock_guard<std::mutex> lk(mutex);
    ret = metric->train(query, meta.dimension());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw streamer metric train failed");
      (*discarded_count)++;
      return ret;
    }
  }

  std::string converted_vector;
  IndexQueryMeta converted_meta;
  ret = reformer->convert(query, qmeta, &converted_vector, &converted_meta);
  if (ret != 0) {
    LOG_ERROR("RaBitQ convert failed, ret=%d", ret);
    (*discarded_count)++;
    return ret;
  }

  level_t level = state->alg->get_random_level();
  ret = state->entity->add_vector_with_id(level, id, converted_vector.data());
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw streamer add vector failed");
    (*discarded_count)++;
    return ret;
  }

  ret = state->alg->add_node(id, level, ctx);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw streamer add node failed");
    (*discarded_count)++;
    return ret;
  }

  if (ailego_unlikely(ctx->error())) {
    (*discarded_count)++;
    return IndexError_Runtime;
  }
  (*added_count)++;
  return 0;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
int RabitqStateSearch(const HnswStreamerRabitqState *state,
                      const std::shared_ptr<RabitqReformer> &reformer,
                      const IndexMeta &meta, const IndexMetric::Pointer &metric,
                      size_t max_scan_limit, size_t min_scan_limit,
                      float max_scan_ratio, size_t bruteforce_threshold,
                      uint32_t magic, const void *query,
                      const IndexQueryMeta &qmeta, uint32_t count,
                      IndexStreamer::Context::Pointer &context) {
  HnswRabitqContext *ctx = dynamic_cast<HnswRabitqContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswRabitqContext failed");
    return IndexError_Cast;
  }

  if (ctx->magic() != magic) {
    int ret = UpdateContextInternal(state, meta, metric, max_scan_limit,
                                    min_scan_limit, max_scan_ratio,
                                    bruteforce_threshold, magic, ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(state->search_distance,
                                      state->search_batch_distance);
  ctx->resize_results(count);
  ctx->check_need_adjuct_ctx(state->entity->doc_cnt());
  for (size_t q = 0; q < count; ++q) {
    HnswRabitqQueryEntity entity;
    int ret = reformer->transform_to_entity(query, &entity);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw searcher transform failed");
      return ret;
    }
    ctx->reset_query(query);
    ret = state->query_alg->search(&entity, ctx);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw searcher fast search failed");
      return ret;
    }
    ctx->topk_to_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }
  return 0;
}

int RabitqStateSearchBF(const HnswStreamerRabitqState *state,
                        const std::shared_ptr<RabitqReformer> &reformer,
                        const IndexMeta &meta,
                        const IndexMetric::Pointer &metric,
                        size_t max_scan_limit, size_t min_scan_limit,
                        float max_scan_ratio, size_t bruteforce_threshold,
                        uint32_t magic, const void *query,
                        const IndexQueryMeta &qmeta, uint32_t count,
                        IndexStreamer::Context::Pointer &context) {
  HnswRabitqContext *ctx = dynamic_cast<HnswRabitqContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswRabitqContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic) {
    int ret = UpdateContextInternal(state, meta, metric, max_scan_limit,
                                    min_scan_limit, max_scan_ratio,
                                    bruteforce_threshold, magic, ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(state->search_distance,
                                      state->search_batch_distance);
  ctx->resize_results(count);

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
      return ctx->group_by()(state->entity->get_key(id));
    };

    for (size_t q = 0; q < count; ++q) {
      HnswRabitqQueryEntity entity;
      int ret = reformer->transform_to_entity(query, &entity);
      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Hnsw rabitq streamer transform failed");
        return ret;
      }
      ctx->reset_query(query);
      ctx->group_topk_heaps().clear();

      for (node_id_t id = 0; id < state->entity->doc_cnt(); ++id) {
        if (state->entity->get_key(id) == kInvalidKey) {
          continue;
        }
        if (!ctx->filter().is_valid() ||
            !ctx->filter()(state->entity->get_key(id))) {
          EstimateRecord dist;
          state->query_alg->get_full_est(id, dist, entity);
          std::string group_id = group_by(id);
          auto &topk_heap = ctx->group_topk_heaps()[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(ctx->group_topk());
          }
          topk_heap.emplace_back(id, dist);
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  } else {
    for (size_t q = 0; q < count; ++q) {
      HnswRabitqQueryEntity entity;
      int ret = reformer->transform_to_entity(query, &entity);
      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Hnsw rabitq streamer transform failed");
        return ret;
      }
      ctx->reset_query(query);
      ctx->topk_heap().clear();
      for (node_id_t id = 0; id < state->entity->doc_cnt(); ++id) {
        if (state->entity->get_key(id) == kInvalidKey) {
          continue;
        }
        if (!ctx->filter().is_valid() ||
            !ctx->filter()(state->entity->get_key(id))) {
          EstimateRecord dist;
          state->query_alg->get_full_est(id, dist, entity);
          ctx->topk_heap().emplace(id, dist);
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }
  return 0;
}

int RabitqStateSearchBFByPKeys(const HnswStreamerRabitqState *state,
                               const std::shared_ptr<RabitqReformer> &reformer,
                               const IndexMeta &meta,
                               const IndexMetric::Pointer &metric,
                               size_t max_scan_limit, size_t min_scan_limit,
                               float max_scan_ratio,
                               size_t bruteforce_threshold, uint32_t magic,
                               const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               IndexStreamer::Context::Pointer &context) {
  HnswRabitqContext *ctx = dynamic_cast<HnswRabitqContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswRabitqContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic) {
    int ret = UpdateContextInternal(state, meta, metric, max_scan_limit,
                                    min_scan_limit, max_scan_ratio,
                                    bruteforce_threshold, magic, ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(state->search_distance,
                                      state->search_batch_distance);
  ctx->resize_results(count);

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
      return ctx->group_by()(state->entity->get_key(id));
    };

    for (size_t q = 0; q < count; ++q) {
      HnswRabitqQueryEntity entity;
      int ret = reformer->transform_to_entity(query, &entity);
      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Hnsw rabitq streamer transform failed");
        return ret;
      }
      ctx->reset_query(query);
      ctx->group_topk_heaps().clear();

      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!ctx->filter().is_valid() || !ctx->filter()(pk)) {
          node_id_t id = state->entity->get_id(pk);
          if (id != kInvalidNodeId) {
            EstimateRecord dist;
            state->query_alg->get_full_est(id, dist, entity);
            std::string group_id = group_by(id);
            auto &topk_heap = ctx->group_topk_heaps()[group_id];
            if (topk_heap.empty()) {
              topk_heap.limit(ctx->group_topk());
            }
            topk_heap.emplace_back(id, dist);
          }
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  } else {
    for (size_t q = 0; q < count; ++q) {
      HnswRabitqQueryEntity entity;
      int ret = reformer->transform_to_entity(query, &entity);
      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Hnsw rabitq streamer transform failed");
        return ret;
      }
      ctx->reset_query(query);
      ctx->topk_heap().clear();
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        key_t pk = p_keys[q][idx];
        if (!ctx->filter().is_valid() || !ctx->filter()(pk)) {
          node_id_t id = state->entity->get_id(pk);
          if (id != kInvalidNodeId) {
            EstimateRecord dist;
            state->query_alg->get_full_est(id, dist, entity);
            ctx->topk_heap().emplace(id, dist);
          }
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Provider & Utilities
// ---------------------------------------------------------------------------
IndexProvider::Pointer RabitqStateCreateProvider(
    const HnswStreamerRabitqState *state, const IndexMeta &meta) {
  auto entity = state->entity->clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("Clone HnswRabitqEntity failed");
    return nullptr;
  }
  return IndexProvider::Pointer(
      new HnswRabitqIndexProvider(meta, entity, "HnswStreamer"));
}

uint32_t RabitqStateDocCount(const HnswStreamerRabitqState *state) {
  return state->entity->doc_cnt();
}

}  // namespace core
}  // namespace zvec
