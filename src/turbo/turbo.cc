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

#include <ailego/internal/cpu_features.h>
#include <zvec/turbo/turbo.h>
#include "avx/float32/cosine.h"
#include "avx/float32/inner_product.h"
#include "avx/float32/squared_euclidean.h"
#include "avx/half_float/cosine.h"
#include "avx/half_float/inner_product.h"
#include "avx/half_float/squared_euclidean.h"
#include "avx2/record_quantized_int4/cosine.h"
#include "avx2/record_quantized_int4/inner_product.h"
#include "avx2/record_quantized_int4/squared_euclidean.h"
#include "avx2/record_quantized_int8/cosine.h"
#include "avx2/record_quantized_int8/inner_product.h"
#include "avx2/record_quantized_int8/squared_euclidean.h"
#include "avx512/float32/cosine.h"
#include "avx512/float32/inner_product.h"
#include "avx512/float32/squared_euclidean.h"
#include "avx512/half_float/cosine.h"
#include "avx512/half_float/inner_product.h"
#include "avx512/half_float/squared_euclidean.h"
#include "avx512_vnni/record_quantized_int8/cosine.h"
#include "avx512_vnni/record_quantized_int8/squared_euclidean.h"
#include "avx512fp16/half_float/cosine.h"
#include "avx512fp16/half_float/inner_product.h"
#include "avx512fp16/half_float/squared_euclidean.h"
#include "scalar/float32/cosine.h"
#include "scalar/float32/inner_product.h"
#include "scalar/float32/squared_euclidean.h"
#include "scalar/half_float/cosine.h"
#include "scalar/half_float/inner_product.h"
#include "scalar/half_float/squared_euclidean.h"
#include "scalar/record_quantized_int4/cosine.h"
#include "scalar/record_quantized_int4/inner_product.h"
#include "scalar/record_quantized_int4/squared_euclidean.h"
#include "scalar/record_quantized_int8/cosine.h"
#include "scalar/record_quantized_int8/inner_product.h"
#include "scalar/record_quantized_int8/squared_euclidean.h"
#include "sse/record_quantized_int4/cosine.h"
#include "sse/record_quantized_int4/inner_product.h"
#include "sse/record_quantized_int4/squared_euclidean.h"
#include "sse/record_quantized_int8/cosine.h"
#include "sse/record_quantized_int8/inner_product.h"
#include "sse/record_quantized_int8/squared_euclidean.h"

namespace zvec::turbo {

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type,
                               CpuArchType cpu_arch_type) {
  // INT8
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512VNNI)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_distance;
        }
      }

      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2 &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX2)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx2::squared_euclidean_int8_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx2::cosine_int8_distance;
        }

        if (metric_type == MetricType::kInnerProduct) {
          return avx2::inner_product_int8_distance;
        }
      }

      if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kSSE)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return sse::squared_euclidean_int8_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return sse::cosine_int8_distance;
        }

        if (metric_type == MetricType::kInnerProduct) {
          return sse::inner_product_int8_distance;
        }
      }

      if (metric_type == MetricType::kSquaredEuclidean) {
        return scalar::squared_euclidean_int8_distance;
      }
      if (metric_type == MetricType::kCosine) {
        return scalar::cosine_int8_distance;
      }

      if (metric_type == MetricType::kInnerProduct) {
        return scalar::inner_product_int8_distance;
      }
    }
  }

  // INT4
  if (data_type == DataType::kInt4) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2 &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX2)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx2::squared_euclidean_int4_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx2::cosine_int4_distance;
        }
        if (metric_type == MetricType::kInnerProduct) {
          return avx2::inner_product_int4_distance;
        }
      }

      if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kSSE)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return sse::squared_euclidean_int4_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return sse::cosine_int4_distance;
        }
        if (metric_type == MetricType::kInnerProduct) {
          return sse::inner_product_int4_distance;
        }
      }

      if (metric_type == MetricType::kSquaredEuclidean) {
        return scalar::squared_euclidean_int4_distance;
      } else if (metric_type == MetricType::kCosine) {
        return scalar::cosine_int4_distance;
      } else if (metric_type == MetricType::kInnerProduct) {
        return scalar::inner_product_int4_distance;
      }
    }
  }

  // FP32
  if (data_type == DataType::kFp32) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512::squared_euclidean_fp32_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512::cosine_fp32_distance;
        }
        if (metric_type == MetricType::kInnerProduct) {
          return avx512::inner_product_fp32_distance;
        }
      }

      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx::squared_euclidean_fp32_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx::cosine_fp32_distance;
        }
        if (metric_type == MetricType::kInnerProduct) {
          return avx::inner_product_fp32_distance;
        }
      }

      if (metric_type == MetricType::kSquaredEuclidean) {
        return scalar::squared_euclidean_fp32_distance;
      }
      if (metric_type == MetricType::kCosine) {
        return scalar::cosine_fp32_distance;
      }
      if (metric_type == MetricType::kInnerProduct) {
        return scalar::inner_product_fp32_distance;
      }
    }
  }

  // FP16
  if (data_type == DataType::kFp16) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16 &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512FP16)) {
        if (metric_type == MetricType::kInnerProduct) {
          return avx512fp16::inner_product_fp16_distance;
        }
      }

      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512::squared_euclidean_fp16_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512::cosine_fp16_distance;
        }
        if (metric_type == MetricType::kInnerProduct) {
          return avx512::inner_product_fp16_distance;
        }
      }

      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx::squared_euclidean_fp16_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx::cosine_fp16_distance;
        }
        if (metric_type == MetricType::kInnerProduct) {
          return avx::inner_product_fp16_distance;
        }
      }

      if (metric_type == MetricType::kSquaredEuclidean) {
        return scalar::squared_euclidean_fp16_distance;
      }
      if (metric_type == MetricType::kCosine) {
        return scalar::cosine_fp16_distance;
      }
      if (metric_type == MetricType::kInnerProduct) {
        return scalar::inner_product_fp16_distance;
      }
    }
  }
  return nullptr;
}

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type,
                                          CpuArchType cpu_arch_type) {
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512VNNI)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_batch_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_batch_distance;
        }
      }
    }
  }

  if (data_type == DataType::kInt4) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2 &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX2)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx2::squared_euclidean_int4_batch_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx2::cosine_int4_batch_distance;
        }
        if (metric_type == MetricType::kInnerProduct) {
          return avx2::inner_product_int4_batch_distance;
        }
      }
    }
  }

  return nullptr;
}

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type,
                                              CpuArchType cpu_arch_type) {
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512VNNI)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_query_preprocess;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_query_preprocess;
        }
      }
    }
  }
  return nullptr;
}

}  // namespace zvec::turbo
