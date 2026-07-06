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
#include <string>
#include <vector>
#include <core/quantizer/quantizer_params.h>
#include <core/quantizer/turbo_quant_engine.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {

/*! Reformer for TurboQuant
 *
 * Transforms query vectors into the TurboQuant quantized format (same
 * packing as the converter) and provides revert() to dequantize
 * stored vectors back to FP32 for exact re-ranking.
 */
class TurboQuantReformer : public IndexReformer {
 public:
  TurboQuantReformer(IndexMeta::DataType /*dst_type*/) {}
  ~TurboQuantReformer() override = default;

  int init(const ailego::Params &params) override {
    int bit_width = 4;
    if (!params.get(TURBO_QUANT_REFORMER_BIT_WIDTH, &bit_width)) {
      LOG_ERROR("TurboQuantReformer: missing bit_width param");
      return IndexError_InvalidArgument;
    }

    std::string mode = TURBO_QUANT_MODE_MSE;
    params.get(TURBO_QUANT_REFORMER_MODE, &mode);
    bool prod_mode = (mode == TURBO_QUANT_MODE_PROD);

    enable_rotate_ = true;
    params.get(TURBO_QUANT_REFORMER_ENABLE_ROTATE, &enable_rotate_);

    int64_t seed_val = 42;
    params.get(TURBO_QUANT_REFORMER_SEED, &seed_val);
    seed_ = static_cast<uint64_t>(seed_val);

    int64_t dim_val = 0;
    if (!params.get(TURBO_QUANT_REFORMER_DIMENSION, &dim_val) || dim_val <= 0) {
      LOG_ERROR("TurboQuantReformer: missing dimension param");
      return IndexError_InvalidArgument;
    }
    dimension_ = static_cast<size_t>(dim_val);

    engine_ = std::make_shared<TurboQuantEngine>(
        dimension_, bit_width, prod_mode, enable_rotate_, seed_);

    initialized_ = true;
    return 0;
  }

  int cleanup(void) override {
    engine_.reset();
    initialized_ = false;
    return 0;
  }

  int load(IndexStorage::Pointer storage) override {
    if (!enable_rotate_ || !storage) return 0;
    int ret = Rotator::open(&rotator_, storage, TURBO_QUANT_SEG_ROTATOR);
    if (ret != 0) {
      LOG_ERROR("TurboQuantReformer: failed to load rotator, ret=%d", ret);
      return ret;
    }
    if (engine_ && rotator_) {
      engine_->set_rotator(rotator_);
    }
    return 0;
  }

  int unload(void) override {
    rotator_.reset();
    return 0;
  }

  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(query, qmeta, 1, out, ometa);
  }

  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(query, qmeta, count, out, ometa);
  }

  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    return do_quantize(record, rmeta, 1, out, ometa);
  }

  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(records, rmeta, count, out, ometa);
  }

  bool need_revert() const override {
    return true;
  }

  int revert(const void *in, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    if (!initialized_ || !engine_) {
      LOG_ERROR("TurboQuantReformer::revert called before init");
      return IndexError_Runtime;
    }
    size_t dim = dimension_;
    out->resize(dim * sizeof(float));
    float *out_buf = reinterpret_cast<float *>(&(*out)[0]);
    engine_->dequantize(reinterpret_cast<const uint8_t *>(in), out_buf);
    (void)qmeta;
    return 0;
  }

 private:
  int do_quantize(const void *src, const IndexQueryMeta &smeta, uint32_t count,
                  std::string *out, IndexQueryMeta *ometa) const {
    if (!initialized_ || !engine_) {
      LOG_ERROR("TurboQuantReformer: quantize called before init");
      return IndexError_Runtime;
    }
    if (smeta.data_type() != IndexMeta::DataType::DT_FP32 ||
        smeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = smeta;
    ometa->set_meta(IndexMeta::DataType::DT_INT8, engine_->total_bytes());
    size_t out_stride = engine_->total_bytes();
    out->resize(static_cast<size_t>(count) * out_stride);

    const float *vec = reinterpret_cast<const float *>(src);
    uint8_t *ovec = reinterpret_cast<uint8_t *>(&(*out)[0]);
    size_t dim = smeta.dimension();
    for (uint32_t i = 0; i < count; ++i) {
      engine_->quantize(vec + i * dim, ovec + i * out_stride);
    }
    return 0;
  }

  size_t dimension_{0};
  bool enable_rotate_{true};
  uint64_t seed_{42};
  bool initialized_{false};
  std::shared_ptr<TurboQuantEngine> engine_{};
  std::shared_ptr<Rotator> rotator_{};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(TurboQuantReformer, TurboQuantReformer,
                                      IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
