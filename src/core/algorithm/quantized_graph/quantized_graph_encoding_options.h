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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace zvec::core {

// Build-time options only. The resulting codec and graph blocks are embedded
// in the snapshot; search never trains or reads an external codec file.
struct QuantizedGraphEncodingOptions {
  // Native remains an alias for source compatibility. Keep the enum values.
  enum class Codec { RaBitQ = 0, Global = 1, LowRank = 2, Native = RaBitQ };
  enum class Codebook { Hadamard, Gaussian };

  size_t max_neighbors = 32;
  size_t threads = 8;

  // Independent signed rotations; persisted with the codec (1 is legacy).
  size_t rotation_views = 1;
  Codec codec = Codec::RaBitQ;
  Codebook codebook = Codebook::Hadamard;
  size_t block_dimension = 6;
  size_t rank = 8;

  size_t training_samples = 1024;  // Maximum number of sampled graph centers.
  uint32_t seed = 20261031;

  // Shrink the fitted center displacement toward the graph node. This is a
  // dataset-independent hyperparameter, not the fitted lambda itself.
  double center_weight = 1;

  static bool parse_name(std::string_view name, Codec &out) {
    if (name == "rabitq") out = Codec::RaBitQ;
    else if (name == "global") out = Codec::Global;
    else if (name == "low_rank") out = Codec::LowRank;
    else return false;
    return true;
  }

  static bool parse_name(std::string_view name, Codebook &out) {
    if (name == "hadamard") out = Codebook::Hadamard;
    else if (name == "gaussian") out = Codebook::Gaussian;
    else return false;
    return true;
  }

  bool valid() const {
    return max_neighbors >= 32 && max_neighbors <= 256 &&
           max_neighbors % 32 == 0 && threads <= 1024 &&
           (rotation_views == 1 || rotation_views == 2 ||
            rotation_views == 4) &&
           (codec == Codec::RaBitQ || codec == Codec::Global ||
            codec == Codec::LowRank) &&
           (codebook == Codebook::Hadamard || codebook == Codebook::Gaussian) &&
           (block_dimension == 2 || block_dimension == 3 ||
            block_dimension == 4 || block_dimension == 6 ||
            block_dimension == 8) &&
           (codebook != Codebook::Hadamard || block_dimension >= 6) &&
           (rank == 8 || rank == 16 || rank == 32) && training_samples > 0 &&
           training_samples <= 16384 && std::isfinite(center_weight) &&
           center_weight >= 0 && center_weight <= 1;
  }
};

}  // namespace zvec::core
