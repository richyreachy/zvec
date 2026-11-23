// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include "quantized_graph_column.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <zvec/core/framework/index_factory.h>
#include "algorithm/quantized_graph/quantized_graph_builder.h"

namespace zvec {
#if ZVEC_QUANTIZED_GRAPH_SUPPORTED
namespace {
Status native_status(int code) {
  return code ? Status::InternalError("Quantized graph: ",
                                      core::IndexError::What(code))
              : Status::OK();
}

core::IndexMeta make_meta(const FieldSchema &field) {
  core::IndexMeta meta(core::IndexMeta::DT_FP32, field.dimension());
  if (std::static_pointer_cast<QuantizedGraphIndexParams>(field.index_params())
          ->metric_type() == MetricType::COSINE)
    meta.set_metric("Cosine", 0, ailego::Params{});
  return meta;
}
ailego::Params build_params(const QuantizedGraphIndexParams &p) {
  ailego::Params params;
  params.set("options", p.options());
  return params;
}
}  // namespace
struct QuantizedGraphColumn::State {
  core::IndexRunner::Pointer runner;
  core::IndexQueryMeta query_meta;
};

Status QuantizedGraphColumn::validate_options(
    const QuantizedGraphIndexParams &p) {
  auto builder = core::IndexFactory::CreateBuilder("QuantizedGraphBuilder");
  if (!builder || !core::QuantizedGraphSupported())
    return Status::NotSupported("quantized graph unavailable");
  if (builder->init(core::IndexMeta{}, build_params(p)))
    return Status::InvalidArgument("invalid QuantizedGraph build options");
  return Status::OK();
}
Status QuantizedGraphColumn::open(const FieldSchema &field,
                                  core_interface::Index &raw, bool create,
                                  bool read_only) {
  auto s = std::make_shared<State>();
  auto meta = make_meta(field);
  s->query_meta.set_meta(core::IndexMeta::DT_FP32, meta.dimension());
  if (!create && raw.get_doc_count()) {
    int rc;
    if (read_only) {
      auto reader =
          core::IndexFactory::CreateSearcher("QuantizedGraphSearcher");
      if (!reader) return Status::InternalError("missing QG searcher");
      rc = reader->init({});
      if (!rc) rc = reader->load(raw.storage_, nullptr);
      if (!rc && (reader->meta().dimension() != meta.dimension() ||
                  reader->meta().metric_name() != meta.metric_name()))
        return Status::InternalError("QG schema mismatch");
      s->runner = std::move(reader);
    } else {
      auto reader =
          core::IndexFactory::CreateStreamer("QuantizedGraphStreamer");
      if (!reader) return Status::InternalError("missing QG streamer");
      rc = reader->init(meta, {});
      if (!rc) rc = reader->open(raw.storage_);
      s->runner = std::move(reader);
    }
    if (rc) return native_status(rc);
    if (s->runner->stats().loaded_count() != size_t(raw.get_doc_count()))
      return Status::InternalError("QG row count mismatch");
  }
  state_ = std::move(s);
  return Status::OK();
}
Status QuantizedGraphColumn::build(const FieldSchema &field,
                                   core_interface::Index &raw) {
  auto rows = raw.create_index_provider();
  if (!rows) return Status::InternalError("missing raw row provider");
  if (!rows->count()) return Status::OK();
  auto builder = std::dynamic_pointer_cast<core::QuantizedGraphBuilder>(
      core::IndexFactory::CreateBuilder("QuantizedGraphBuilder"));
  if (!builder) return Status::InternalError("missing QG builder");
  int rc = builder->init(
      make_meta(field),
      build_params(*std::static_pointer_cast<QuantizedGraphIndexParams>(
          field.index_params())));
  if (!rc) rc = builder->build(rows);
  if (!rc) rc = builder->dump(*raw.storage_);
  if (rc) return native_status(rc);
  builder.reset();
  return open(field, raw, false, false);
}
Status QuantizedGraphColumn::search_ids(const void *vector,
                                        const QueryParams::Ptr &params,
                                        uint32_t k, const IndexFilter *filter,
                                        int64_t *ids, float *scores) const {
  const QuantizedGraphQueryParams defaults;
  const auto *p =
      params ? dynamic_cast<const QuantizedGraphQueryParams *>(params.get())
             : &defaults;
  if (!p || !p->capacity)
    return Status::InvalidArgument("invalid QG query parameter");
  if (!k) return Status::OK();
  std::fill_n(ids, k, -1);
  if (scores) std::fill_n(scores, k, std::numeric_limits<float>::quiet_NaN());
  const auto &s = *state_;
  if (!s.runner) return Status::OK();
  // Cache the validated framework parameters by value. QueryParams objects
  // may be mutated or replaced between calls on the same thread.
  const auto settings =
      std::make_tuple(p->capacity, p->margin, p->bucket_width,
                      p->max_expansions, p->route_beam, p->uncertainty_bonus);
  struct Scratch {
    std::optional<std::decay_t<decltype(settings)>> previous;
    std::weak_ptr<core::IndexRunner> owner;
    core::IndexContext::Pointer context;
  };
  thread_local Scratch scratch;
  if (scratch.owner.lock() != s.runner) {
    scratch.context = s.runner->create_context();
    if (!scratch.context)
      return Status::InternalError("failed to create QG context");
    scratch.owner = s.runner;
    scratch.previous.reset();
  }
  auto &ctx = scratch.context;
  ctx->set_topk(k);
  if (!scratch.previous || *scratch.previous != settings) {
    ailego::Params config;
    config.set("capacity", p->capacity);
    config.set("margin", p->margin);
    config.set("bucket_width", p->bucket_width);
    config.set("max_expansions", p->max_expansions);
    config.set("route_beam", p->route_beam);
    config.set("uncertainty_bonus", p->uncertainty_bonus);
    int rc = ctx->update(config);
    if (rc) return native_status(rc);
    scratch.previous = settings;
  }
  if (filter)
    ctx->set_filter(
        [filter](uint64_t key) { return filter->is_filtered(key); });
  int rc = s.runner->search_impl(vector, s.query_meta, ctx);
  ctx->reset_filter();
  if (rc) return native_status(rc);
  size_t i = 0;
  for (const auto &doc : ctx->result()) {
    ids[i] = doc.key();
    if (scores) scores[i] = doc.score();
    ++i;
  }
  return Status::OK();
}

#else
Status QuantizedGraphColumn::validate_options(
    const QuantizedGraphIndexParams &) {
  return Status::NotSupported("QG unavailable on this platform");
}
Status QuantizedGraphColumn::open(const FieldSchema &, core_interface::Index &,
                                  bool, bool) {
  return Status::NotSupported("QG unavailable on this platform");
}
Status QuantizedGraphColumn::build(const FieldSchema &,
                                   core_interface::Index &) {
  return Status::NotSupported("QG unavailable on this platform");
}
Status QuantizedGraphColumn::search_ids(const void *, const QueryParams::Ptr &,
                                        uint32_t, const IndexFilter *,
                                        int64_t *, float *) const {
  return Status::NotSupported("QG unavailable on this platform");
}
#endif

Result<IndexResults::Ptr> QuantizedGraphColumn::search(
    core_interface::Index &raw, const void *vector,
    const vector_column_params::QueryParams &params) const {
  if (params.group_by || params.refiner_param)
    return tl::make_unexpected(
        Status::NotSupported("grouping or external refinement for QG"));
  std::vector<int64_t> ids(params.topk);
  std::vector<float> scores(params.topk);
  auto filter = params.filter;
  std::unique_ptr<EasyIndexFilter> allowed;
  if (!params.bf_pks.empty()) {
    if (params.bf_pks.size() != 1)
      return tl::make_unexpected(
          Status::InvalidArgument("invalid candidate lists"));
    std::unordered_set<uint64_t> keys(params.bf_pks.front().begin(),
                                      params.bf_pks.front().end());
    allowed = std::make_unique<EasyIndexFilter>(
        [keys = std::move(keys), filter](uint64_t k) {
          return !keys.count(k) || (filter && filter->is_filtered(k));
        });
  }
  auto status =
      search_ids(vector, params.query_params, params.topk,
                 allowed ? allowed.get() : filter, ids.data(), scores.data());
  if (!status.ok()) return tl::make_unexpected(status);
  core::IndexDocumentList docs;
  std::vector<std::string> vectors;
  for (size_t i = 0; i < ids.size() && ids[i] >= 0; ++i) {
    docs.emplace_back(ids[i], scores[i]);
    if (params.fetch_vector) {
      core_interface::VectorDataBuffer row;
      if (raw.fetch(ids[i], &row) != 0)
        return tl::make_unexpected(
            Status::InternalError("QG row fetch failed"));
      vectors.emplace_back(std::move(
          std::get<core_interface::DenseVectorBuffer>(row.vector_buffer).data));
    }
  }
  return std::make_unique<VectorIndexResults>(
      false, std::move(docs), std::move(vectors), std::vector<std::string>{});
}
}  // namespace zvec
