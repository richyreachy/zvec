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
#if RABITQ_SUPPORTED
#include <cmath>
#include <set>
#include <shared_mutex>
#include <unordered_set>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/utility/float_helper.h>
#include "algorithm/hnsw_rabitq/rabitq_params.h"
#include "algorithm/vamana/vamana_streamer.h"
#include "turbo/quantizer/rabit_quantizer/rabit_quantizer.h"
#include "diskann_params.h"

namespace zvec {
namespace core {
namespace {
constexpr char kCodec[] = "diskann.stream.v1.rabitq";
constexpr char kPublishedRows[] = "diskann.stream.v1.published";
constexpr uint32_t kRowsPerPage = 4096;
std::string page_name(size_t page) {
  return "diskann.stream.v1.codes." + std::to_string(page);
}
}  // namespace

// Distinct on-disk layout. The writable single-level Vamana graph owns the
// original records and reciprocal edges. Append-only code pages use the same
// dense internal ordinal as the graph; arbitrary external keys stay
// independent.
class DiskAnnMutableStreamer final : public VamanaStreamer {
 public:
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    if (storage_) return IndexError_NoReady;
    dim_ = meta.dimension();
    params.get(PARAM_DISKANN_STREAM_DIMENSION, &dim_);
    if (meta.meta_type() != IndexMeta::MT_DENSE || dim_ < kMinRabitqDimSize ||
        dim_ > kMaxRabitqDimSize ||
        (meta.data_type() != IndexMeta::DataType::DT_FP16 &&
         meta.data_type() != IndexMeta::DataType::DT_FP32))
      return IndexError_Unsupported;
    cosine_ = meta.metric_name() == "Cosine";
    if (!cosine_ && meta.metric_name() != "SquaredEuclidean" &&
        meta.metric_name() != "InnerProduct")
      return IndexError_Unsupported;
    size_t raw = dim_ * meta.unit_size();
    if (meta.element_size() != raw &&
        !(cosine_ && meta.element_size() == raw + sizeof(float)))
      return IndexError_Mismatch;
    IndexMeta codec_meta(IndexMeta::DataType::DT_FP32, dim_);
    codec_meta.set_metric(cosine_ ? "InnerProduct" : meta.metric_name(), 0,
                          ailego::Params());
    ailego::Params options;
    options.set(PARAM_RABITQ_TOTAL_BITS, 2u);
    options.set(PARAM_RABITQ_FIXED_ZERO_CENTER, true);
    codec_ = std::make_unique<turbo::RabitQuantizer>();
    int ret = codec_->init(codec_meta, options);
    if (ret != 0) return ret;
    if (codec_->require_train()) return IndexError_Unsupported;
    code_size_ = codec_->quantized_datapoint_vector_length();
    auto graph_params = params;
    graph_params.set(PARAM_VAMANA_STREAMER_USE_ID_MAP, true);
    graph_params.set(PARAM_VAMANA_STREAMER_TWO_PASS_BUILD_ENABLE, false);
    return VamanaStreamer::init(meta, graph_params);
  }

  int open(IndexStorage::Pointer stg) override {
    return open(stg, false);
  }
  int open(IndexStorage::Pointer stg, bool read_only) override {
    if (!stg || storage_ || !codec_) return IndexError_NoReady;
    read_only = read_only || stg->name() == "MMapFileReadStorage" ||
                stg->name() == "FileReadStorage" ||
                stg->name() == "BufferReadStorage";
    auto metadata = stg->get(kCodec);
    uint32_t published = 0;
    IndexStorage::Segment::Pointer publication;
    if (!metadata) {
      if (read_only || !stg->get_all().empty()) return IndexError_InvalidFormat;
      std::string blob;
      int ret = codec_->serialize(&blob);
      if (ret != 0) return ret;
      if (stg->append(kCodec, blob.size()) != 0 ||
          !(metadata = stg->get(kCodec)) ||
          metadata->write(0, blob.data(), blob.size()) != blob.size() ||
          metadata->resize(blob.size()) != blob.size())
        return IndexError_WriteData;
      metadata->update_data_crc(
          ailego::Crc32c::Hash(blob.data(), blob.size(), 0));
      if (stg->append(kPublishedRows, sizeof(published)) != 0 ||
          !(publication = stg->get(kPublishedRows)) ||
          publication->write(0, &published, sizeof(published)) !=
              sizeof(published) ||
          publication->resize(sizeof(published)) != sizeof(published)) {
        return IndexError_WriteData;
      }
    } else {
      publication = stg->get(kPublishedRows);
      if (!publication || publication->data_size() != sizeof(published) ||
          publication->fetch(0, &published, sizeof(published)) !=
              sizeof(published)) {
        return IndexError_InvalidFormat;
      }
      // Only accept our fixed-center codec descriptor (bounded before
      // allocation).
      size_t n = metadata->data_size();
      if (!n || n > dim_ * 16 + 8192) return IndexError_InvalidFormat;
      std::string blob(n, '\0');
      if (metadata->fetch(0, blob.data(), n) != n ||
          ailego::Crc32c::Hash(blob.data(), n, 0) != metadata->data_crc() ||
          codec_->deserialize(blob.data(), blob.size()) != 0 ||
          codec_->require_train() ||
          codec_->quantized_datapoint_vector_length() != code_size_)
        return IndexError_InvalidFormat;
    }
    int ret = VamanaStreamer::open(stg);
    if (ret != 0) return ret;
    if (published != graph().doc_cnt()) {
      VamanaStreamer::close();
      return IndexError_InvalidFormat;
    }
    pages_.clear();
    for (size_t i = 0; i < graph().doc_cnt(); i += kRowsPerPage) {
      auto seg = stg->get(page_name(i / kRowsPerPage));
      const size_t valid =
          std::min<size_t>(kRowsPerPage, graph().doc_cnt() - i) * code_size_;
      if (!seg || seg->data_size() < valid ||
          seg->capacity() < kRowsPerPage * code_size_) {
        VamanaStreamer::close();
        return IndexError_InvalidFormat;
      }
      pages_.push_back(std::move(seg));
    }
    publication_ = std::move(publication);
    storage_ = std::move(stg);
    read_only_ = read_only;
    failed_ = false;
    return 0;
  }

  int flush(uint64_t cp) override {
    std::unique_lock<std::shared_mutex> lock(lock_);
    if (!storage_ || read_only_ || failed_) return IndexError_NoReady;
    return VamanaStreamer::flush(cp);
  }
  int close() override {
    std::unique_lock<std::shared_mutex> lock(lock_);
    if (!storage_) return IndexError_NoReady;
    int ret = VamanaStreamer::close();
    publication_.reset();
    pages_.clear();
    storage_.reset();
    return ret;
  }
  int cleanup() override {
    if (storage_) {
      int ret = close();
      if (ret != 0) return ret;
    }
    return VamanaStreamer::cleanup();
  }
  int dump(const IndexDumper::Pointer &) override {
    // A Vamana-only dump would silently omit the navigation codes.
    return IndexError_Unsupported;
  }

  int add_with_id_impl(uint32_t id, const void *v, const IndexQueryMeta &m,
                       ContextPointer &ctx) override {
    return add_impl(id, v, m, ctx);
  }
  int add_impl(uint64_t key, const void *v, const IndexQueryMeta &m,
               ContextPointer &ctx) override {
    std::unique_lock<std::shared_mutex> lock(lock_);
    if (!storage_ || read_only_ || failed_) return IndexError_NoReady;
    if (check_params(v, m) != 0 || !ctx || ctx->magic() != graph_magic() ||
        key == kInvalidKey)
      return IndexError_InvalidArgument;
    if (graph().get_id(key) != kInvalidNodeId) return IndexError_Exist;
    auto x = floats(v);
    if (x.empty()) return IndexError_InvalidArgument;
    std::vector<char> code(code_size_);
    codec_->quantize_data(x.data(), code.data());
    const size_t id = graph().doc_cnt();
    const size_t page = id / kRowsPerPage;
    const size_t off = (id % kRowsPerPage) * code_size_;
    if (page == pages_.size()) {
      auto name = page_name(page);
      auto seg = storage_->get(name);
      if (!seg) {
        int ret = storage_->append(name, kRowsPerPage * code_size_);
        if (ret != 0) return ret;
        seg = storage_->get(name);
      }
      if (!seg || seg->capacity() < kRowsPerPage * code_size_)
        return IndexError_WriteData;
      pages_.push_back(std::move(seg));
    }
    auto &seg = pages_[page];
    if (seg->write(off, code.data(), code_size_) != code_size_ ||
        seg->resize(off + code_size_) != off + code_size_)
      return IndexError_WriteData;
    // The graph publication is last. A failed code write cannot expose a node;
    // readers never traverse edges while the reverse lists are being pruned.
    int ret = VamanaStreamer::add_impl(key, v, m, ctx);
    if (ret == 0) {
      const uint32_t count = graph().doc_cnt();
      if (publication_->write(0, &count, sizeof(count)) != sizeof(count)) {
        ret = IndexError_WriteData;
      }
    }
    if (ret != 0) failed_ = true;
    return ret;
  }

  int search_impl(const void *q, const IndexQueryMeta &m,
                  ContextPointer &ctx) const override {
    return search_impl(q, m, 1, ctx);
  }
  int search_impl(const void *q, const IndexQueryMeta &m, uint32_t count,
                  ContextPointer &context) const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    auto *ctx = dynamic_cast<VamanaContext *>(context.get());
    if (!storage_ || failed_) return IndexError_NoReady;
    if (!ctx || ctx->magic() != graph_magic() || check_params(q, m) != 0)
      return IndexError_Mismatch;
    ctx->clear();
    ctx->resize_results(count);
    for (uint32_t i = 0; i < count; ++i) {
      int ret = search_one(
          static_cast<const char *>(q) + size_t(i) * m.element_size(), ctx, i);
      if (ret != 0) return ret;
    }
    return 0;
  }

  int search_candidates_impl(const void *q, const IndexQueryMeta &m,
                             std::vector<uint64_t> &keys, ContextPointer &ctx,
                             std::vector<float> *scores) const override {
    return IndexRunner::search_candidates_impl(q, m, keys, ctx, scores);
  }
  IndexProvider::Pointer create_provider() const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    return storage_ && !failed_ ? VamanaStreamer::create_provider() : nullptr;
  }
  int get_vector_by_id(uint32_t id,
                       IndexStorage::MemoryBlock &block) const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    return storage_ && !failed_ && id < graph().doc_cnt()
               ? graph().get_vector(id, block)
               : IndexError_NoExist;
  }
  const void *get_vector_by_id(uint32_t id) const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    return storage_ && !failed_ && id < graph().doc_cnt()
               ? graph().get_vector(id)
               : nullptr;
  }
  int get_vector(uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    if (!storage_ || failed_) return IndexError_NoReady;
    if (key == kInvalidKey) return IndexError_NoExist;
    return graph().get_vector_by_key(key, block);
  }
  int get_vector_by_key(uint64_t key,
                        IndexStorage::MemoryBlock &block) const override {
    return get_vector(key, block);
  }
  const void *get_vector(uint64_t key) const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    return storage_ && !failed_ && key != kInvalidKey
               ? graph().get_vector_by_key(key)
               : nullptr;
  }
  ContextPointer create_context() const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    return storage_ && !failed_ ? VamanaStreamer::create_context() : nullptr;
  }
  int search_bf_impl(const void *q, const IndexQueryMeta &m, uint32_t count,
                     ContextPointer &ctx) const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    return storage_ && !failed_
               ? VamanaStreamer::search_bf_impl(q, m, count, ctx)
               : IndexError_NoReady;
  }
  int search_bf_by_p_keys_impl(const void *q,
                               const std::vector<std::vector<uint64_t>> &keys,
                               const IndexQueryMeta &m, uint32_t count,
                               ContextPointer &ctx) const override {
    std::shared_lock<std::shared_mutex> lock(lock_);
    return storage_ && !failed_ ? VamanaStreamer::search_bf_by_p_keys_impl(
                                      q, keys, m, count, ctx)
                                : IndexError_NoReady;
  }

 private:
  std::vector<float> floats(const void *data) const {
    std::vector<float> x(dim_);
    if (meta().data_type() == IndexMeta::DataType::DT_FP32)
      std::memcpy(x.data(), data, x.size() * sizeof(float));
    else
      std::copy_n(static_cast<const ailego::Float16 *>(data), dim_, x.begin());
    double norm = 0;
    for (float v : x) {
      if (!std::isfinite(v)) return {};
      norm += double(v) * v;
    }
    if (cosine_ && norm > 0) {
      double inv = 1 / std::sqrt(norm);
      for (auto &v : x) v = static_cast<float>(v * inv);
    }
    return x;
  }
  int search_one(const void *q, VamanaContext *ctx, uint32_t index) const {
    auto *out = ctx->mutable_result(index);
    out->clear();
    if (!graph().doc_cnt() || !ctx->topk()) return 0;
    // Keep the closest L navigation codes; rank expanded, eligible nodes by
    // their original vectors. A filtered node can still lead to a match.
    auto x = floats(q);
    if (x.empty()) return IndexError_InvalidArgument;
    std::vector<char> query(codec_->quantized_query_vector_length());
    codec_->quantize_query(x.data(), query.data());
    using Candidate = std::pair<float, uint32_t>;
    std::set<Candidate> best, frontier, results;
    std::unordered_set<uint32_t> seen;
    const size_t limit = std::max(ctx->topk(), ctx->ef());
    auto enqueue = [&](uint32_t id) -> int {
      if (id >= graph().doc_cnt()) return IndexError_InvalidFormat;
      if (!seen.insert(id).second) return 0;
      IndexStorage::MemoryBlock block;
      if (pages_[id / kRowsPerPage]->read((id % kRowsPerPage) * code_size_,
                                          block, code_size_) != code_size_)
        return IndexError_ReadData;
      float score = codec_->calc_distance_dp_query(block.data(), query.data());
      if (!std::isfinite(score)) return IndexError_InvalidFormat;
      Candidate candidate(score, id);
      if (best.size() == limit && candidate >= *best.rbegin()) return 0;
      best.insert(candidate);
      frontier.insert(candidate);
      if (best.size() > limit) {
        auto last = std::prev(best.end());
        frontier.erase(*last);
        best.erase(last);
      }
      return 0;
    };
    int ret = enqueue(graph().entry_point());
    if (ret != 0) return ret;
    while (!frontier.empty()) {
      uint32_t id = frontier.begin()->second;
      frontier.erase(frontier.begin());
      auto key = graph().get_key(id);
      if (key == kInvalidKey) return IndexError_ReadData;
      if (!ctx->filter()(key)) {
        IndexStorage::MemoryBlock block;
        if (graph().get_vector(id, block) != 0) return IndexError_ReadData;
        float d = exact_graph_distance(q, block.data());
        if (d <= ctx->threshold()) {
          results.emplace(d, id);
          if (results.size() > ctx->topk())
            results.erase(std::prev(results.end()));
        }
      }
      auto neighbors = graph().get_neighbors(id);
      if (!neighbors.data || neighbors.size() > graph().max_degree())
        return IndexError_ReadData;
      for (size_t i = 0; i < neighbors.size(); ++i) {
        ret = enqueue(neighbors[i]);
        if (ret != 0) return ret;
      }
    }
    for (auto [d, id] : results) {
      auto key = graph().get_key(id);
      if (ctx->fetch_vector()) {
        IndexStorage::MemoryBlock b;
        if (graph().get_vector(id, b) != 0) return IndexError_ReadData;
        out->emplace_back(key, d, id, std::move(b));
      } else
        out->emplace_back(key, d, id);
    }
    return 0;
  }

  mutable std::shared_mutex lock_;
  IndexStorage::Pointer storage_;
  IndexStorage::Segment::Pointer publication_;
  std::unique_ptr<turbo::RabitQuantizer> codec_;
  std::vector<IndexStorage::Segment::Pointer> pages_;
  size_t code_size_{0};
  uint32_t dim_{0};
  bool cosine_{false}, read_only_{true}, failed_{false};
};
INDEX_FACTORY_REGISTER_STREAMER(DiskAnnMutableStreamer);
}  // namespace core
}  // namespace zvec
#endif
