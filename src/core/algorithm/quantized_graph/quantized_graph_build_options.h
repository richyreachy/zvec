// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace zvec::core {

// Build-time options only. The resulting codec and graph blocks are embedded
// in the snapshot; search never trains or reads an external codec file.
struct QuantizedGraphBuildOptions {
  enum class Codec { Native, Global, LowRank };
  enum class Codebook { Hadamard, Gaussian };
  size_t max_neighbors = 32;
  size_t threads = 8;
  Codec codec = Codec::Native;
  Codebook codebook = Codebook::Hadamard;
  size_t block_dimension = 6;
  size_t rank = 8;
  size_t training_samples = 1024;  // Maximum number of sampled graph centers.
  uint32_t seed = 20261031;
  // Shrink the fitted center displacement toward the graph node. This is a
  // dataset-independent hyperparameter, not the fitted lambda itself.
  double center_weight = 1;

  bool valid() const {
    return max_neighbors >= 32 && max_neighbors <= 256 &&
           max_neighbors % 32 == 0 && threads <= 1024 &&
           (codec == Codec::Native || codec == Codec::Global ||
            codec == Codec::LowRank) &&
           (codebook == Codebook::Hadamard || codebook == Codebook::Gaussian) &&
           (block_dimension == 4 || block_dimension == 6 ||
            block_dimension == 8) &&
           (codebook != Codebook::Hadamard || block_dimension != 4) &&
           (rank == 8 || rank == 16 || rank == 32) && training_samples > 0 &&
           training_samples <= 16384 && std::isfinite(center_weight) &&
           center_weight >= 0 && center_weight <= 1;
  }
};
}  // namespace zvec::core
