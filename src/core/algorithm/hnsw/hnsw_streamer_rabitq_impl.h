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
// Interface for the RaBitQ-specific implementation of HnswStreamer.
// This header uses only forward declarations and framework types so that it can
// be included from hnsw_streamer.cc (which includes hnsw_entity.h) without
// pulling in hnsw_rabitq_entity.h (which has conflicting type definitions).
//
// The actual implementation lives in hnsw_rabitq/hnsw_streamer_rabitq_impl.cc
// and is compiled as part of core_knn_hnsw_rabitq (with AVX2 flags).
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <ailego/parallel/lock.h>
#include <zvec/core/framework/index_framework.h>

namespace zvec {
namespace core {

class RabitqReformer;

// Opaque state holding all RaBitQ-specific objects.
// Defined in hnsw_rabitq/hnsw_streamer_rabitq_impl.cc.
struct HnswStreamerRabitqState;

void DestroyRabitqState(HnswStreamerRabitqState *p);
struct RabitqStateDeleter {
  void operator()(HnswStreamerRabitqState *p) const {
    DestroyRabitqState(p);
  }
};
using RabitqStatePtr =
    std::unique_ptr<HnswStreamerRabitqState, RabitqStateDeleter>;

// ---------------------------------------------------------------------------
// Factory: create and initialize the rabitq state (entity + build algorithm).
// Called from HnswStreamer::init() in rabitq mode.
// ---------------------------------------------------------------------------
RabitqStatePtr CreateRabitqState(
    IndexRunner::Stats &stats, bool use_id_map, uint8_t ex_bits,
    uint32_t dimension, uint32_t ef_construction,
    uint32_t upper_max_neighbor_cnt, uint32_t l0_max_neighbor_cnt,
    uint32_t scaling_factor, uint32_t prune_cnt, size_t chunk_size,
    bool filter_same_key, bool get_vector_enabled, uint32_t min_neighbor_cnt,
    size_t docs_hard_limit);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
int RabitqStateOpen(HnswStreamerRabitqState *state,
                    const std::shared_ptr<RabitqReformer> &reformer,
                    IndexStorage::Pointer stg, size_t max_index_size,
                    bool check_crc_enabled, IndexMeta &meta);

int RabitqStateClose(HnswStreamerRabitqState *state, IndexMeta &meta,
                     const IndexMetric::Pointer &metric);

int RabitqStateFlush(HnswStreamerRabitqState *state, IndexMeta &meta,
                     const IndexMetric::Pointer &metric, uint64_t checkpoint);

int RabitqStateDump(HnswStreamerRabitqState *state,
                    const std::shared_ptr<RabitqReformer> &reformer,
                    const IndexDumper::Pointer &dumper);

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
IndexStreamer::Context::Pointer RabitqStateCreateContext(
    const HnswStreamerRabitqState *state, const IndexMeta &meta,
    const IndexMetric::Pointer &metric, uint32_t ef, size_t max_scan_limit,
    size_t min_scan_limit, float max_scan_ratio, bool bf_enabled,
    float bf_negative_prob, uint32_t magic, bool force_padding_topk_enabled,
    size_t bruteforce_threshold);

int RabitqStateUpdateContext(const HnswStreamerRabitqState *state,
                             const IndexMeta &meta,
                             const IndexMetric::Pointer &metric,
                             size_t max_scan_limit, size_t min_scan_limit,
                             float max_scan_ratio, size_t bruteforce_threshold,
                             uint32_t magic,
                             IndexStreamer::Context::Pointer &context);

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
    const IndexQueryMeta &qmeta, IndexStreamer::Context::Pointer &context);

int RabitqStateAddWithId(
    HnswStreamerRabitqState *state,
    const std::shared_ptr<RabitqReformer> &reformer,
    const IndexProvider::Pointer &provider, const IndexMetric::Pointer &metric,
    const IndexMeta &meta, size_t docs_soft_limit, size_t docs_hard_limit,
    size_t max_scan_limit, size_t min_scan_limit, float max_scan_ratio,
    size_t bruteforce_threshold, uint32_t magic, std::mutex &mutex,
    ailego::SharedMutex &shared_mutex, std::atomic<size_t> *added_count,
    std::atomic<size_t> *discarded_count, uint32_t id, const void *query,
    const IndexQueryMeta &qmeta, IndexStreamer::Context::Pointer &context);

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
                      IndexStreamer::Context::Pointer &context);

int RabitqStateSearchBF(const HnswStreamerRabitqState *state,
                        const std::shared_ptr<RabitqReformer> &reformer,
                        const IndexMeta &meta,
                        const IndexMetric::Pointer &metric,
                        size_t max_scan_limit, size_t min_scan_limit,
                        float max_scan_ratio, size_t bruteforce_threshold,
                        uint32_t magic, const void *query,
                        const IndexQueryMeta &qmeta, uint32_t count,
                        IndexStreamer::Context::Pointer &context);

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
                               IndexStreamer::Context::Pointer &context);

// ---------------------------------------------------------------------------
// Provider & Utilities
// ---------------------------------------------------------------------------
IndexProvider::Pointer RabitqStateCreateProvider(
    const HnswStreamerRabitqState *state, const IndexMeta &meta);

uint32_t RabitqStateDocCount(const HnswStreamerRabitqState *state);

}  // namespace core
}  // namespace zvec
