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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <ailego/pattern/defer.h>
#include <core/quantizer/quantizer_params.h>
#include <core/quantizer/turbo_quant_engine.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {

/*! TurboQuant Converter
 *
 * Implements both MSE-optimal (Algorithm 1) and inner-product-optimal
 * (Algorithm 2) TurboQuant quantization, following the IndexConverter
 * abstract interface.  The converter is data-oblivious: the codebook is
 * precomputed from the Beta distribution and does not require training
 * on the input data.
 */
class TurboQuantConverter : public IndexConverter {
 public:
  TurboQuantConverter(IndexMeta::DataType /*dst_type*/) {}
  ~TurboQuantConverter() override = default;

  int init(const IndexMeta &index_meta, const ailego::Params &params) override {
    meta_ = index_meta;
    original_dimension_ = index_meta.dimension();

    int bit_width = 4;
    params.get(TURBO_QUANT_CONVERTER_BIT_WIDTH, &bit_width);
    if (bit_width < 1 || bit_width > 8) {
      LOG_ERROR("TurboQuantConverter: invalid bit_width=%d (must be 1-8)",
                bit_width);
      return IndexError_InvalidArgument;
    }

    std::string mode = TURBO_QUANT_MODE_MSE;
    params.get(TURBO_QUANT_CONVERTER_MODE, &mode);
    bool prod_mode = (mode == TURBO_QUANT_MODE_PROD);

    enable_rotate_ = true;
    params.get(TURBO_QUANT_CONVERTER_ENABLE_ROTATE, &enable_rotate_);

    seed_ = 42;
    params.get(TURBO_QUANT_CONVERTER_SEED, &seed_);

    engine_ = std::make_shared<TurboQuantEngine>(
        original_dimension_, bit_width, prod_mode, enable_rotate_, seed_);

    meta_.set_converter("TurboQuantConverter", 0, params);
    meta_.set_meta(IndexMeta::DataType::DT_INT8, engine_->total_bytes());

    ailego::Params metric_params;
    metric_params.set(TURBO_QUANT_REFORMER_BIT_WIDTH, bit_width);
    metric_params.set(TURBO_QUANT_REFORMER_MODE, mode);
    metric_params.set(TURBO_QUANT_REFORMER_ENABLE_ROTATE, enable_rotate_);
    metric_params.set(TURBO_QUANT_REFORMER_SEED, static_cast<int64_t>(seed_));
    metric_params.set(TURBO_QUANT_REFORMER_DIMENSION,
                      static_cast<int64_t>(original_dimension_));
    metric_params.set("turbo_quant.metric.origin_metric_name",
                      index_meta.metric_name());
    meta_.set_metric("TurboQuant", 0, metric_params);

    ailego::Params reformer_params;
    reformer_params.set(TURBO_QUANT_REFORMER_BIT_WIDTH, bit_width);
    reformer_params.set(TURBO_QUANT_REFORMER_MODE, mode);
    reformer_params.set(TURBO_QUANT_REFORMER_ENABLE_ROTATE, enable_rotate_);
    reformer_params.set(TURBO_QUANT_REFORMER_SEED, static_cast<int64_t>(seed_));
    reformer_params.set(TURBO_QUANT_REFORMER_DIMENSION,
                        static_cast<int64_t>(original_dimension_));
    meta_.set_reformer("TurboQuantReformer", 0, reformer_params);

    ailego::Params conv_params = params;
    conv_params.set(TURBO_QUANT_CONVERTER_BIT_WIDTH, bit_width);
    conv_params.set(TURBO_QUANT_CONVERTER_MODE, mode);
    conv_params.set(TURBO_QUANT_CONVERTER_ENABLE_ROTATE, enable_rotate_);
    conv_params.set(TURBO_QUANT_CONVERTER_SEED, static_cast<int64_t>(seed_));
    meta_.set_converter("TurboQuantConverter", 0, conv_params);

    return 0;
  }

  int cleanup(void) override {
    *stats_.mutable_trained_count() = 0;
    *stats_.mutable_transformed_count() = 0;
    return 0;
  }

  int train(IndexHolder::Pointer /*holder*/) override {
    return 0;
  }

  int transform(IndexHolder::Pointer holder) override {
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != original_dimension_) {
      LOG_ERROR(
          "TurboQuantConverter: type/dimension mismatch (type=%d, dim=%zu, "
          "expected dim=%zu)",
          holder->data_type(), holder->dimension(), original_dimension_);
      return IndexError_Mismatch;
    }
    *stats_.mutable_transformed_count() += holder->count();
    holder_ = std::make_shared<TurboQuantHolder>(holder, engine_);
    return 0;
  }

  int dump(const IndexDumper::Pointer &dumper) override {
    if (enable_rotate_ && engine_ && engine_->rotator()) {
      return engine_->rotator()->dump(dumper, TURBO_QUANT_SEG_ROTATOR);
    }
    return 0;
  }

  int dump_to_storage(const IndexStorage::Pointer &storage) override {
    if (enable_rotate_ && engine_ && engine_->rotator()) {
      return engine_->rotator()->dump(storage, TURBO_QUANT_SEG_ROTATOR);
    }
    return 0;
  }

  const Stats &stats(void) const override {
    return stats_;
  }
  IndexHolder::Pointer result(void) const override {
    return holder_;
  }
  const IndexMeta &meta(void) const override {
    return meta_;
  }

 private:
  class TurboQuantHolder : public IndexHolder {
   public:
    class Iterator : public IndexHolder::Iterator {
     public:
      Iterator(const TurboQuantHolder *owner,
               IndexHolder::Iterator::Pointer &&iter)
          : owner_(owner),
            buffer_(owner->element_size(), 0),
            front_iter_(std::move(iter)) {
        encode_record();
      }

      ~Iterator() override = default;

      const void *data(void) const override {
        return buffer_.data();
      }
      bool is_valid(void) const override {
        return front_iter_->is_valid();
      }
      uint64_t key(void) const override {
        return front_iter_->key();
      }

      void next(void) override {
        front_iter_->next();
        encode_record();
      }

     private:
      void encode_record(void) {
        if (!front_iter_->is_valid()) return;
        const float *vec = reinterpret_cast<const float *>(front_iter_->data());
        owner_->engine_->quantize(vec, buffer_.data());
      }

      const TurboQuantHolder *owner_;
      std::vector<uint8_t> buffer_;
      IndexHolder::Iterator::Pointer front_iter_;
    };

    TurboQuantHolder(IndexHolder::Pointer front,
                     std::shared_ptr<TurboQuantEngine> engine)
        : front_(std::move(front)), engine_(std::move(engine)) {}

    size_t count(void) const override {
      return front_->count();
    }
    size_t dimension(void) const override {
      return engine_->total_bytes();
    }
    IndexMeta::DataType data_type(void) const override {
      return IndexMeta::DataType::DT_INT8;
    }
    size_t element_size(void) const override {
      return engine_->total_bytes();
    }
    bool multipass(void) const override {
      return front_->multipass();
    }

    IndexHolder::Iterator::Pointer create_iterator(void) override {
      auto iter = front_->create_iterator();
      return iter ? IndexHolder::Iterator::Pointer(
                        new Iterator(this, std::move(iter)))
                  : IndexHolder::Iterator::Pointer();
    }

   private:
    IndexHolder::Pointer front_;
    std::shared_ptr<TurboQuantEngine> engine_;
  };

  IndexMeta meta_{};
  Stats stats_{};
  IndexHolder::Pointer holder_{};
  size_t original_dimension_{0};
  bool enable_rotate_{true};
  uint64_t seed_{42};
  std::shared_ptr<TurboQuantEngine> engine_{};
};

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(TurboQuantConverter, TurboQuantConverter,
                                       IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
