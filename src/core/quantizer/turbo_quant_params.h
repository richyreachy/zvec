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

#include <string>

namespace zvec {
namespace core {

//! TurboQuantConverter / TurboQuantReformer parameter names

//! Bit-width per coordinate (1-8).  For Prod mode the effective MSE
//! bit-width is bits-1 and the remaining bit is used by the QJL stage.
static const std::string TURBO_QUANT_CONVERTER_BIT_WIDTH =
    "turbo_quant.converter.bit_width";
//! Mode: "mse" (MSE-optimal only) or "prod" (unbiased inner product).
static const std::string TURBO_QUANT_CONVERTER_MODE =
    "turbo_quant.converter.mode";
//! Enable random rotation (default: true).
static const std::string TURBO_QUANT_CONVERTER_ENABLE_ROTATE =
    "turbo_quant.converter.enable_rotate";
//! Random seed for rotation and QJL matrix (default: 42).
static const std::string TURBO_QUANT_CONVERTER_SEED =
    "turbo_quant.converter.seed";

//! Reformer: bit-width (persisted from converter)
static const std::string TURBO_QUANT_REFORMER_BIT_WIDTH =
    "turbo_quant.reformer.bit_width";
//! Reformer: mode (persisted from converter)
static const std::string TURBO_QUANT_REFORMER_MODE =
    "turbo_quant.reformer.mode";
//! Reformer: enable rotation (persisted from converter)
static const std::string TURBO_QUANT_REFORMER_ENABLE_ROTATE =
    "turbo_quant.reformer.enable_rotate";
//! Reformer: random seed (persisted from converter)
static const std::string TURBO_QUANT_REFORMER_SEED =
    "turbo_quant.reformer.seed";
//! Reformer: dimension
static const std::string TURBO_QUANT_REFORMER_DIMENSION =
    "turbo_quant.reformer.dimension";

//! Segment IDs for persistence
static const std::string TURBO_QUANT_SEG_ROTATOR{"turbo_quant_rotator"};
static const std::string TURBO_QUANT_SEG_QJL_MATRIX{"turbo_quant_qjl_matrix"};

//! Mode constants
static const std::string TURBO_QUANT_MODE_MSE = "mse";
static const std::string TURBO_QUANT_MODE_PROD = "prod";

}  // namespace core
}  // namespace zvec
