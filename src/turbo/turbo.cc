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

#include <turbo/turbo.h>
#include <zvec/turbo/turbo.h>

namespace zvec::turbo {
namespace {
static_assert(
    static_cast<int>(MetricType::kSquaredEuclidean) ==
        static_cast<int>(::turbo::MetricType::kSquaredEuclidean),
    "Turbo MetricType::kSquaredEuclidean changed; review the adapter");
static_assert(static_cast<int>(MetricType::kCosine) ==
                  static_cast<int>(::turbo::MetricType::kCosine),
              "Turbo MetricType::kCosine changed; review the adapter");
static_assert(static_cast<int>(MetricType::kInnerProduct) ==
                  static_cast<int>(::turbo::MetricType::kInnerProduct),
              "Turbo MetricType::kInnerProduct changed; review the adapter");
static_assert(
    static_cast<int>(MetricType::kMipsSquaredEuclidean) ==
        static_cast<int>(::turbo::MetricType::kMipsSquaredEuclidean),
    "Turbo MetricType::kMipsSquaredEuclidean changed; review the adapter");
static_assert(static_cast<int>(MetricType::kUnknown) ==
                  static_cast<int>(::turbo::MetricType::kUnknown),
              "Turbo MetricType::kUnknown changed; review the adapter");
::turbo::MetricType ToKernel(MetricType value) {
  return static_cast<::turbo::MetricType>(value);
}
static_assert(static_cast<int>(DataType::kInt4) ==
                  static_cast<int>(::turbo::DataType::kInt4),
              "Turbo DataType::kInt4 changed; review the adapter");
static_assert(static_cast<int>(DataType::kInt8) ==
                  static_cast<int>(::turbo::DataType::kInt8),
              "Turbo DataType::kInt8 changed; review the adapter");
static_assert(static_cast<int>(DataType::kFp16) ==
                  static_cast<int>(::turbo::DataType::kFp16),
              "Turbo DataType::kFp16 changed; review the adapter");
static_assert(static_cast<int>(DataType::kFp32) ==
                  static_cast<int>(::turbo::DataType::kFp32),
              "Turbo DataType::kFp32 changed; review the adapter");
static_assert(static_cast<int>(DataType::kUint8) ==
                  static_cast<int>(::turbo::DataType::kUint8),
              "Turbo DataType::kUint8 changed; review the adapter");
static_assert(static_cast<int>(DataType::kUnknown) ==
                  static_cast<int>(::turbo::DataType::kUnknown),
              "Turbo DataType::kUnknown changed; review the adapter");
static_assert(static_cast<int>(DataType::kUint4) ==
                  static_cast<int>(::turbo::DataType::kUint4),
              "Turbo DataType::kUint4 changed; review the adapter");
static_assert(static_cast<int>(DataType::kUint7) ==
                  static_cast<int>(::turbo::DataType::kUint7),
              "Turbo DataType::kUint7 changed; review the adapter");
::turbo::DataType ToKernel(DataType value) {
  return static_cast<::turbo::DataType>(value);
}
static_assert(static_cast<int>(QuantizeType::kUniform) ==
                  static_cast<int>(::turbo::QuantizeType::kUniform),
              "Turbo QuantizeType::kUniform changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kRecord) ==
                  static_cast<int>(::turbo::QuantizeType::kRecord),
              "Turbo QuantizeType::kRecord changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kFp16) ==
                  static_cast<int>(::turbo::QuantizeType::kFp16),
              "Turbo QuantizeType::kFp16 changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kFp32) ==
                  static_cast<int>(::turbo::QuantizeType::kFp32),
              "Turbo QuantizeType::kFp32 changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kPQ) ==
                  static_cast<int>(::turbo::QuantizeType::kPQ),
              "Turbo QuantizeType::kPQ changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kRabit) ==
                  static_cast<int>(::turbo::QuantizeType::kRabit),
              "Turbo QuantizeType::kRabit changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kUniformUint8) ==
                  static_cast<int>(::turbo::QuantizeType::kUniformUint8),
              "Turbo QuantizeType::kUniformUint8 changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kRaw) ==
                  static_cast<int>(::turbo::QuantizeType::kRaw),
              "Turbo QuantizeType::kRaw changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kPQFast) ==
                  static_cast<int>(::turbo::QuantizeType::kPQFast),
              "Turbo QuantizeType::kPQFast changed; review the adapter");
static_assert(static_cast<int>(QuantizeType::kUniformUint4) ==
                  static_cast<int>(::turbo::QuantizeType::kUniformUint4),
              "Turbo QuantizeType::kUniformUint4 changed; review the adapter");
::turbo::QuantizeType ToKernel(QuantizeType value) {
  return static_cast<::turbo::QuantizeType>(value);
}
static_assert(static_cast<int>(RotateType::kFht) ==
                  static_cast<int>(::turbo::RotateType::kFht),
              "Turbo RotateType::kFht changed; review the adapter");
static_assert(static_cast<int>(RotateType::kOpq) ==
                  static_cast<int>(::turbo::RotateType::kOpq),
              "Turbo RotateType::kOpq changed; review the adapter");
::turbo::RotateType ToKernel(RotateType value) {
  return static_cast<::turbo::RotateType>(value);
}
static_assert(static_cast<int>(CpuArchType::kAuto) ==
                  static_cast<int>(::turbo::CpuArchType::kAuto),
              "Turbo CpuArchType::kAuto changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kScalar) ==
                  static_cast<int>(::turbo::CpuArchType::kScalar),
              "Turbo CpuArchType::kScalar changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kSSE2) ==
                  static_cast<int>(::turbo::CpuArchType::kSSE2),
              "Turbo CpuArchType::kSSE2 changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kAVX) ==
                  static_cast<int>(::turbo::CpuArchType::kAVX),
              "Turbo CpuArchType::kAVX changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kAVX2) ==
                  static_cast<int>(::turbo::CpuArchType::kAVX2),
              "Turbo CpuArchType::kAVX2 changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kAVX512) ==
                  static_cast<int>(::turbo::CpuArchType::kAVX512),
              "Turbo CpuArchType::kAVX512 changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kAVX512VNNI) ==
                  static_cast<int>(::turbo::CpuArchType::kAVX512VNNI),
              "Turbo CpuArchType::kAVX512VNNI changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kAVX512FP16) ==
                  static_cast<int>(::turbo::CpuArchType::kAVX512FP16),
              "Turbo CpuArchType::kAVX512FP16 changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kNEON) ==
                  static_cast<int>(::turbo::CpuArchType::kNEON),
              "Turbo CpuArchType::kNEON changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kSVE) ==
                  static_cast<int>(::turbo::CpuArchType::kSVE),
              "Turbo CpuArchType::kSVE changed; review the adapter");
static_assert(static_cast<int>(CpuArchType::kSVE2) ==
                  static_cast<int>(::turbo::CpuArchType::kSVE2),
              "Turbo CpuArchType::kSVE2 changed; review the adapter");
::turbo::CpuArchType ToKernel(CpuArchType value) {
  return static_cast<::turbo::CpuArchType>(value);
}
}  // namespace

DistanceKernels get_distance_kernels(MetricType metric, DataType data,
                                     QuantizeType quant, CpuArchType arch) {
  const auto kernels = ::turbo::get_distance_kernels(
      ToKernel(metric), ToKernel(data), ToKernel(quant), ToKernel(arch));
  // std::function retains support for the capturing distance adapters in zvec.
  return {kernels.dist, kernels.batch, kernels.preprocess};
}
DistanceFunc get_distance_func(MetricType metric, DataType data,
                               QuantizeType quant, CpuArchType arch) {
  return get_distance_kernels(metric, data, quant, arch).dist;
}
BatchDistanceFunc get_batch_distance_func(MetricType metric, DataType data,
                                          QuantizeType quant,
                                          CpuArchType arch) {
  return get_distance_kernels(metric, data, quant, arch).batch;
}
QueryPreprocessFunc get_query_preprocess_func(MetricType metric, DataType data,
                                              QuantizeType quant,
                                              CpuArchType arch) {
  return get_distance_kernels(metric, data, quant, arch).preprocess;
}
UniformQuantizeFunc get_uniform_quantize_func(DataType data) {
  return ::turbo::get_uniform_quantize_func(ToKernel(data));
}
UniformUint4QuantizeFunc get_uniform_uint4_quantize_func(DataType data) {
  return ::turbo::get_uniform_uint4_quantize_func(ToKernel(data));
}
ConvertFunc get_convert_func(DataType data) {
  return ::turbo::get_convert_func(ToKernel(data));
}
RotatorKernels get_rotator_kernels(RotateType rotate, CpuArchType arch) {
  const auto kernels =
      ::turbo::get_rotator_kernels(ToKernel(rotate), ToKernel(arch));
  return {kernels.rotate, kernels.unrotate};
}
CodebookKernels get_pq_kernels(DataType data, QuantizeType quant,
                               CpuArchType arch) {
  const auto kernels =
      ::turbo::get_pq_kernels(ToKernel(data), ToKernel(quant), ToKernel(arch));
  return {kernels.asymmetric_distance, kernels.symmetric_distance,
          kernels.batch_asymmetric_distance, kernels.fast_scan};
}
}  // namespace zvec::turbo
