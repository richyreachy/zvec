// Copyright 2026-present the zvec project
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

#include <zvec/core/interface/index.h>
#include "algorithm/quantized_graph/quantized_graph_search_context.h"

namespace zvec::core_interface {

class QuantizedGraphIndex final : public Index {
 public:
  int open(const std::string &path, StorageOptions options) override {
    if (!options.read_only || options.create_new) {
      return core::IndexError_Unsupported;
    }
    int ret = Index::open(path, options);
    if (ret == 0) is_trained_ = true;
    return ret;
  }

 protected:
  int create_and_init_converter_reformer(const QuantizerParam &quantizer,
                                         const BaseIndexParam &param) override {
    return quantizer.type == QuantizerType::kNone && !param.use_external_vector
               ? 0
               : core::IndexError_Unsupported;
  }

  int create_and_init_streamer(const BaseIndexParam &) override {
    streamer_ = core::IndexFactory::CreateStreamer("QuantizedGraphStreamer");
    if (!streamer_) return core::IndexError_Unsupported;
    // The QG searcher returns final L2 squared or cosine distances.
    metric_.reset();
    return streamer_->init(proxima_index_meta_, {});
  }

  int _dense_fetch(uint32_t, VectorDataBuffer *) override {
    return core::IndexError_Unsupported;
  }

  int _prepare_for_search(const VectorData &,
                          const BaseIndexQueryParam::Pointer &param,
                          core::IndexContext::Pointer &ctx) override {
    const auto *p = dynamic_cast<const QuantizedGraphQueryParam *>(param.get());
    if (!p) return core::IndexError_InvalidArgument;
    if (p->fetch_vector || p->is_linear || p->bf_pks ||
        (p->group_by_param && p->group_by_param->group_by))
      return core::IndexError_Unsupported;
    auto *qg = dynamic_cast<core::QuantizedGraphSearchContext *>(ctx.get());
    if (!qg) return core::IndexError_InvalidArgument;
    int ret = qg->configure(p->capacity, p->margin, p->bucket_width,
                            p->max_expansions, p->route_beam,
                            p->uncertainty_bonus);
    if (ret) return ret;
    ctx->set_topk(p->topk);
    if (p->filter) ctx->set_filter(*p->filter);
    else ctx->reset_filter();
    if (p->radius > 0) ctx->set_threshold(p->radius);
    else ctx->reset_threshold();
    return 0;
  }
};
}  // namespace zvec::core_interface
