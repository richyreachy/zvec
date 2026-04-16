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
#pragma once

#include <functional>
#include <zvec/ailego/math_batch/utils.h>

namespace zvec::turbo {

using DistanceFunc =
    std::function<void(const void *m, const void *q, size_t dim, float *out)>;
using BatchDistanceFunc = std::function<void(
    const void **m, const void *q, size_t num, size_t dim, float *out)>;
using QueryPreprocessFunc =
    zvec::ailego::DistanceBatch::DistanceBatchQueryPreprocessFunc;

enum class MetricType {
  kSquaredEuclidean,
  kCosine,
  kInnerProduct,
  kMipsSquaredEuclidean,
  kUnknown,
};

enum class DataType {
  kInt4,
  kInt8,
  kFp16,
  kFp32,
  kUnknown,
};

enum class QuantizeType {
  kDefault,
  kRecordInt8,
  kRecordInt4,
  kInt8,
  kInt4,
  kFp16,
  kPQ,
  kRabit
};

enum class CpuArchType {
  kAuto,
  kScalar,
  kSSE,
  kAVX,
  kAVX2,
  kAVX512,
  kAVX512VNNI,
  kAVX512FP16
};

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type,
                               CpuArchType cpu_arch_type = CpuArchType::kAuto);

BatchDistanceFunc get_batch_distance_func(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

QueryPreprocessFunc get_query_preprocess_func(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

}  // namespace zvec::turbo
